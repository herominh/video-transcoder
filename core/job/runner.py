"""Run one claimed execution to its result (contract README 15.1, from rule 3 on).

JobRunner.run() is the job runner core.protocol.handler.handle() hands a claimed execution to. It
returns the execution's one ending, sealed: `transcode.result.completed` once the generation and its
manifest are stored under the granted prefix, `transcode.result.failed` for every other ending, with
the typed failure and the stage it ended in.

The stages, in order (the `stage` of a progress message and of a failed result; null before the first):

    (none)        what is left of the start's wall time becomes the execution's deadline (the handler
                  counts that time from the moment the start began, so the claim has already spent
                  some of it; with none left the execution ends here, before anything else); both
                  storage access profiles, the media profile, the grant's key, the ffmpeg version and
                  both storage clients are settled before anything is read: what cannot work is found
                  before a byte is downloaded or encoded
    probing       the preflight: the source's facts from a bounded conditional read and its admission;
                  then the plan: the renditions and the room they may take, so that a source that
                  cannot be planned or does not fit fails before it is downloaded
    downloading   the whole source into the scratch directory
    transcoding   the planned renditions, one at a time, within their output allowance
    finalizing    the thumbnail, the master playlist, the manifest, and the manifest's self-check
    uploading     the files the manifest lists, then the manifest

Everything the execution writes locally lives in one scratch directory of its own, removed whatever
the ending. The media key of the grant goes to the encode and nowhere else, and the storage credentials
to the two client factories (one for the source, one for the output) and nowhere else: neither is
logged, put in a message or written by this module (README 15.1, rules 4 and 5). A bug is logged by its
class and its stack only: an unexpected error's own text can quote anything, so it reaches no log line.

A stage that runs out of time ends the execution for good (`deadline_exceeded` is not retryable there: a
limit that is too small must not become a paid retry loop). One exception lives in the preflight: when
its own time budget ran out while the storage was the slow party, it fails retryable instead
(core/probe.py, source_unavailable / probe_source_slow). So that the Hub and an operator can tell a
limit that is too small from a start whose claim took most of the time, the detail of a failed
`deadline_exceeded` result ends with how much of the start's wall time was gone before the execution began, in
whole seconds: the failure's own detail first, never cut, so one that leaves no room goes without.
"""

from __future__ import annotations

import errno
import logging
import shutil
import tempfile
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from core.admission import RequestLimits
from core.bounded_process import MS_PER_SECOND, Deadline
from core.encode import ArtifactFile, OutputAllowance, RenditionOutput, encode_rendition
from core.encode_command import EncodeSource
from core.failure import DETAIL_MAX_CHARS, Diagnostic, Failure, WorkerFailure, make_detail
from core.hls_key import KEY_BYTES
from core.package import (
    CONTRACT_MAX_DIAGNOSTICS,
    CONTRACT_MAX_DURATION_MS,
    ENCRYPTION_AES_128,
    MANIFEST_VERSION,
    EncryptionRef,
    ManifestIdentity,
    ManifestOutput,
    ProfileRef,
    manifest_max_bytes,
    reserve_package,
    write_manifest,
    write_master_playlist,
)
from core.probe import run_preflight
from core.profile import MediaProfile, supported_profile
from core.protocol import messages, pipeline
from core.protocol.context import TrustedContext
from core.protocol.handler import ClaimedExecution, Runtime
from core.protocol.messages import Encoder, ManifestLocator, Media, SourceRead
from core.protocol.outbox import SealedMessage, SelfCheckFailed, seal_message
from core.protocol.registry import Channel, MessageKind, Role
from core.protocol.settings import StorageProfile, UnknownCredentialRef
from core.renditions import RenditionPlan, plan_renditions
from core.source_facts import SourceFacts
from core.source_reader import RangeSource, S3RangeSource
from core.thumbnail import ThumbnailResult, make_thumbnail

from .config import JobSettings
from .ffmpeg_version import installed_ffmpeg_version
from .progress import ProgressReporter
from .storage import DeadlineSource, download_source, storage_client, upload_client, upload_generation

logger = logging.getLogger(__name__)

# The contract's stage names, in the order this runner goes through them.
STAGE_PROBING = "probing"
STAGE_DOWNLOADING = "downloading"
STAGE_TRANSCODING = "transcoding"
STAGE_FINALIZING = "finalizing"
STAGE_UPLOADING = "uploading"

SCRATCH_PREFIX = "vh-job-"
SOURCE_DIR_NAME = "source"
# A fixed name without an extension: nothing of the source's own name or type reaches the scratch disk.
SOURCE_FILE_NAME = "source"
OUTPUT_DIR_NAME = "output"
WORK_DIR_NAME = "work"
# The two sides of an execution's storage, as a log line and a failure's detail name them.
SIDE_SOURCE = "source"
SIDE_OUTPUT = "output"
# What a failure's detail calls the errno of an OSError that carries none the platform names.
UNKNOWN_ERRNO_NAME = "unknown"
# How many errors of a chain (an error, its cause or the error it interrupted, and so on) a bug's log follows.
MAX_CHAINED_ERRORS = 8
# The class of the failures whose detail says what the start had spent before the execution began, and
# what that detail is made of: the failure's own, then the two numbers in whole seconds.
DEADLINE_EXCEEDED_CLASS = "deadline_exceeded"
SPENT_BEFORE_START_DETAIL = "{detail}; the start had spent {spent_s} s of its {limit_s} s before the execution began"

UNSUPPORTED_PROFILE = Failure(
    "configuration_error", "unsupported_profile", False,
    "the dispatch names a media profile revision this worker does not hold",
)
GRANT_KEY_MISMATCH = Failure(
    "internal_error", "grant_key_mismatch", False,
    "the grant's media key does not fit the dispatch's encryption",
)
# Retryable, unlike a deadline that passes during a stage: only the claim phase can spend a start's wall
# time before the execution begins, which is a passing fault (a Hub that was slow to grant), and the
# execution did nothing (no read, no encode, no write). A new dispatch has a fresh budget, and the
# attempt it repeats cost nothing.
WALL_TIME_SPENT_BEFORE_START = Failure(
    "deadline_exceeded", "wall_time_spent_before_start", True,
    "the start's wall time was spent before the execution could begin",
)


@dataclass(frozen=True)
class _Scratch:
    """The execution's own directories: the downloaded source, the output to upload, the stages' transient files."""

    source_path: Path
    output_root: Path
    work_dir: Path


@dataclass(frozen=True)
class _Plan:
    """What the probing stage settles before anything is downloaded: the source's facts, the renditions
    to encode, the room those may take together and the room the manifest may take."""

    facts: SourceFacts
    renditions: tuple[RenditionPlan, ...]
    allowance: OutputAllowance
    manifest_max_bytes: int


@dataclass(frozen=True)
class _Encoded:
    """What the transcoding stage hands to the package: the renditions and what the package is sized after."""

    source: EncodeSource
    largest_plan: RenditionPlan
    renditions: tuple[RenditionOutput, ...]
    manifest_max_bytes: int


@dataclass(frozen=True)
class _Generation:
    """The finished generation: the manifest, every file it lists, and the encoder that made them."""

    manifest: ManifestOutput
    master: ArtifactFile
    thumbnail: ThumbnailResult
    renditions: tuple[RenditionOutput, ...]
    encoder: Encoder

    @property
    def artifacts(self) -> tuple[ArtifactFile, ...]:
        """Exactly what the manifest lists, in its order: the master, the thumbnail if any, each rendition's files."""
        package = (self.master,) if self.thumbnail.artifact is None else (self.master, self.thumbnail.artifact)
        return package + tuple(artifact for rendition in self.renditions for artifact in rendition.artifacts)


def _media(renditions: Sequence[RenditionOutput]) -> Media:
    """The generation's media facts, computed as the manifest's own `media` block is (core/package.py)."""
    return Media(
        # The cut lets a rendition run up to one frame past an admitted 6 hours.
        duration_ms=min(max(rendition.duration_ms for rendition in renditions), CONTRACT_MAX_DURATION_MS),
        max_width=max(rendition.width for rendition in renditions),
        max_height=max(rendition.height for rendition in renditions),
        has_audio=renditions[0].has_audio,
        rendition_count=len(renditions),
    )


def _chained_to(error: BaseException) -> BaseException | None:
    """The error Python reports before this one: its cause, else the error it interrupted unless suppressed."""
    if error.__cause__ is not None:
        return error.__cause__
    return None if error.__suppress_context__ else error.__context__


def _stack_report(error: BaseException) -> str:
    """Where an unexpected error came from, without a word of what it says.

    The error's class name and its stack (file names, line numbers and source lines), then the same for
    each error it is chained to. No error's own text is part of it: that text can quote anything the
    error choked on (a URL with its query string, a header, a path).
    """
    links: list[str] = []
    seen: set[int] = set()
    link: BaseException | None = error
    while link is not None and id(link) not in seen and len(seen) < MAX_CHAINED_ERRORS:
        seen.add(id(link))
        links.append(f"{type(link).__name__}\n{''.join(traceback.format_tb(link.__traceback__))}")
        link = _chained_to(link)
    return "chained to ".join(links).rstrip("\n")


def _is_passing_shortage(error: Exception) -> bool:
    """Whether a start failed for want of something of the worker's own (memory, a descriptor), which the
    next start would not meet again: the line core/job/ffmpeg_version.py draws for a failed start.

    A MemoryError, and any OSError but a FileNotFoundError or a PermissionError: a file that is missing
    or may not be read is the installation's, and every start meets it again.
    """
    if isinstance(error, MemoryError):
        return True
    return isinstance(error, OSError) and not isinstance(error, (FileNotFoundError, PermissionError))


def _shortage_name(error: Exception) -> str:
    """A shortage by its class and, for an OSError, its errno's symbolic name: never the error's own
    message or file name, which may carry a path."""
    name = type(error).__name__
    if not isinstance(error, OSError):
        return name
    return f"{name} ({errno.errorcode.get(error.errno or 0, UNKNOWN_ERRNO_NAME)})"


class JobRunner:
    """Runs claimed executions with one job configuration; `run` is a core.protocol.handler.JobRunner."""

    def __init__(
        self,
        job_settings: JobSettings,
        *,
        source_client_factory: Callable[[StorageProfile], Any] = storage_client,
        output_client_factory: Callable[[StorageProfile], Any] = upload_client,
        ffmpeg_version: Callable[[Deadline], str] = installed_ffmpeg_version,
    ) -> None:
        """`source_client_factory` makes the S3 client the source is read with and `output_client_factory`
        the one the generation is stored with, each from an access profile; `ffmpeg_version` names the
        installed ffmpeg within the execution's deadline, which it is handed, or raises WorkerFailure
        when it cannot."""
        if not isinstance(job_settings, JobSettings):
            raise TypeError("job_settings must be a JobSettings")
        if not all(callable(given) for given in (source_client_factory, output_client_factory, ffmpeg_version)):
            raise TypeError("source_client_factory, output_client_factory and ffmpeg_version must be callable")
        self._job_settings = job_settings
        self._source_client_factory = source_client_factory
        self._output_client_factory = output_client_factory
        self._ffmpeg_version = ffmpeg_version

    def run(self, claimed: ClaimedExecution, runtime: Runtime) -> SealedMessage:
        """Run the granted execution and return its sealed `transcode.result.completed` or `.failed`.

        The execution has what is left of its start's wall time: up to `claimed.wall_time_ends_at`, a
        reading of `runtime.monotonic`, so the time the claim took is already counted. With none left
        it does nothing and returns a failed result. Raises only for arguments of the wrong type and
        when the failed result itself cannot be built or sealed (a wiring bug).
        """
        if not isinstance(claimed, ClaimedExecution):
            raise TypeError("claimed must be a ClaimedExecution")
        if not isinstance(runtime, Runtime):
            raise TypeError("runtime must be a Runtime")
        return _Execution(
            claimed,
            runtime,
            self._job_settings,
            self._source_client_factory,
            self._output_client_factory,
            self._ffmpeg_version,
        ).run()


class _Execution:
    """One execution on its way to its result; it remembers how far it came, for a failed result."""

    def __init__(
        self,
        claimed: ClaimedExecution,
        runtime: Runtime,
        job_settings: JobSettings,
        source_client_factory: Callable[[StorageProfile], Any],
        output_client_factory: Callable[[StorageProfile], Any],
        ffmpeg_version: Callable[[Deadline], str],
    ) -> None:
        self._dispatch = claimed.dispatch
        self._grant = claimed.grant
        self._runtime = runtime
        self._job_settings = job_settings
        self._source_client_factory = source_client_factory
        self._output_client_factory = output_client_factory
        self._ffmpeg_version = ffmpeg_version
        # A reading of the runtime's clock is only ever compared with another reading of it: what is left
        # of the start's wall time is taken as a duration, and the Deadline (on time.monotonic) is built
        # once, from that duration.
        left_ms = int((claimed.wall_time_ends_at - runtime.monotonic()) * MS_PER_SECOND)
        self._wall_time_spent = left_ms < 1
        # Kept for a deadline failure's detail, in whole seconds rounded down: the dispatch's limit, and
        # what the start had spent of it by now (more than the limit when the claim overran it).
        limit_ms = claimed.dispatch.limits.max_wall_time_ms
        self._wall_time_limit_s = limit_ms // MS_PER_SECOND
        self._spent_before_start_s = max(0, limit_ms - left_ms) // MS_PER_SECOND
        # With no time left the execution ends at its first step, and its deadline is one that has passed.
        self._deadline = Deadline(time.monotonic()) if self._wall_time_spent else Deadline.after_ms(left_ms)
        self._progress = ProgressReporter(claimed, runtime)
        self._stage: str | None = None
        self._bytes_read = 0
        # True once the preflight read the source: every read was conditional on the recorded ETag.
        self._etag_observed = False
        self._encoder: Encoder | None = None
        self._refusal_diagnostics: tuple[Diagnostic, ...] = ()
        self._scratch_root: Path | None = None

    def run(self) -> SealedMessage:
        try:
            return self._ending()
        finally:
            self._remove_scratch()

    def _ending(self) -> SealedMessage:
        execution_id = self._grant.identity.execution_id
        try:
            return self._completed(self._stored_generation())
        except WorkerFailure as ended:
            failure = ended.failure
        except Exception as error:  # noqa: BLE001 - a bug still ends the execution with a result
            # No `exc_info`: a traceback ends with the error's own text. The class and the stack say where.
            logger.error("execution %s ended on an unexpected %s", execution_id, _stack_report(error))
            failure = Failure("internal_error", "unexpected_error", True, make_detail(type(error).__name__))
        failure = self._with_the_time_spent_before_start(failure)
        logger.warning(
            "execution %s failed at stage %s: %s/%s", execution_id, self._stage, failure.error_class, failure.code
        )
        return self._failed(failure)

    def _with_the_time_spent_before_start(self, failure: Failure) -> Failure:
        """A deadline failure, its detail followed by how much of the start's wall time was gone before
        the execution began; every other failure as it is.

        Whatever ran out of time (a stage, a transfer, the ffmpeg check, or the start before the execution
        could begin), the two numbers tell a limit that is too small for the job from a start whose claim
        took most of it. (Where a stage's own budget ran out with wall time left, the code names that
        budget, and the numbers only show that the wall time was not what ran out.) The failure's own
        detail is never cut: one that leaves no room for the numbers within what a detail may hold stays
        as it is.
        """
        if failure.error_class != DEADLINE_EXCEEDED_CLASS:
            return failure
        detail = SPENT_BEFORE_START_DETAIL.format(
            detail=failure.detail, spent_s=self._spent_before_start_s, limit_s=self._wall_time_limit_s
        )
        if len(detail) > DETAIL_MAX_CHARS:
            return failure
        return replace(failure, detail=detail)

    # --- the stages --------------------------------------------------------------------------------

    def _stored_generation(self) -> _Generation:
        """Every stage in order; returns once the generation and its manifest are stored."""
        if self._wall_time_spent:
            # Before anything else: no profile is looked up, no client made, nothing read, written or reported.
            raise WorkerFailure(WALL_TIME_SPENT_BEFORE_START)
        source_access, output_access = self._storage_profiles()
        profile = self._media_profile()
        self._require_grant_key_fits()
        encoder = self._identified_encoder(profile)
        source = self._source(self._client(SIDE_SOURCE, self._source_client_factory, source_access))
        output_client = self._client(SIDE_OUTPUT, self._output_client_factory, output_access)
        # The preflight makes its own reads of the source: this holds each of them to the execution's
        # deadline while it is in flight, as the download and the upload hold theirs.
        plan = self._probe(DeadlineSource(source, self._deadline), profile)
        scratch = self._download(source)
        encoded = self._transcode(scratch, plan, profile)
        generation = self._finalize(scratch, encoded, profile, encoder)
        self._upload(output_client, scratch, generation)
        return generation

    def _enter(self, stage: str) -> None:
        self._stage = stage
        logger.info("execution %s: %s", self._grant.identity.execution_id, stage)
        self._progress.stage_started(stage)

    def _storage_profiles(self) -> tuple[StorageProfile, StorageProfile]:
        """The access profiles of the source and of the output, both resolved before either is used (README 7)."""
        settings = self._runtime.settings
        try:
            return (
                settings.storage_profile(self._dispatch.source.credential_ref),
                settings.storage_profile(self._dispatch.output.credential_ref),
            )
        except UnknownCredentialRef as unknown:
            # A credential_ref is a name the contract's schema bounds, never a secret.
            detail = make_detail(f"no storage profile is configured for the credential_ref {unknown.credential_ref}")
            raise WorkerFailure(
                Failure(UnknownCredentialRef.ERROR_CLASS, UnknownCredentialRef.ERROR_CODE, False, detail)
            ) from None

    def _media_profile(self) -> MediaProfile:
        named = self._dispatch.profile
        profile = supported_profile(named.id, named.version, named.sha256)
        if profile is None:
            raise WorkerFailure(UNSUPPORTED_PROFILE)
        return profile

    def _require_grant_key_fits(self) -> None:
        """Defence in depth: the validator already checked the grant against the dispatch's encryption (S31)."""
        encryption, grant = self._dispatch.encryption, self._grant
        if encryption.mode == ENCRYPTION_AES_128:
            key = grant.media_key
            fits = isinstance(key, bytes) and len(key) == KEY_BYTES and grant.media_key_id == encryption.media_key_id
        else:
            fits = grant.media_key is None
        if not fits:
            raise WorkerFailure(GRANT_KEY_MISMATCH)

    def _identified_encoder(self, profile: MediaProfile) -> Encoder:
        """What encodes this execution; kept, so that a failed result names it too."""
        encoder = Encoder(
            image_digest=self._job_settings.image_digest,
            ffmpeg_version=self._ffmpeg_version(self._deadline),
            video_encoder=profile.encode.video_encoder,
            preset=profile.encode.encoder_preset,
        )
        self._encoder = encoder
        return encoder

    def _client(self, side: str, factory: Callable[[StorageProfile], Any], access: StorageProfile) -> Any:
        """The S3 client of one side (the source's or the output's), made before the first stage: making
        one sends no request, and a profile no client can be made of must not cost a download and an encode.

        Whatever the factory raises ends the execution typed, by the question core/job/ffmpeg_version.py
        asks of a failed start: would the next start, on a healthy host, meet it again? A passing shortage
        of the worker's own (no memory, no descriptor left: _is_passing_shortage) would not, and is a
        retryable resource_exhausted/storage_client_start_failed. Everything else would, a missing or
        unreadable file included, and is configuration_error/storage_client_unavailable, not retryable.
        """
        try:
            return factory(access)
        except Exception as error:  # noqa: BLE001 - whatever the factory raises, the execution ends typed
            shortage = _is_passing_shortage(error)
            # The class name only, and a shortage's errno by its symbolic name, here and in the result:
            # such an error's own text can quote the endpoint.
            named = _shortage_name(error) if shortage else type(error).__name__
            detail = make_detail(f"the {side} storage client could not be made: {named}")
            if shortage:
                logger.warning("%s", detail)
                failure = Failure("resource_exhausted", "storage_client_start_failed", True, detail)
            else:
                logger.error("%s", detail)
                failure = Failure("configuration_error", "storage_client_unavailable", False, detail)
            raise WorkerFailure(failure) from None

    def _source(self, client: Any) -> S3RangeSource:
        located = self._dispatch.source
        return S3RangeSource(
            client,
            bucket=located.bucket,
            key=located.object_key,
            size_bytes=located.size_bytes,
            etag=located.etag,
        )

    def _probe(self, source: RangeSource, profile: MediaProfile) -> _Plan:
        self._enter(STAGE_PROBING)
        limits = self._dispatch.limits
        preflight = run_preflight(
            source,
            profile=profile,
            limits=RequestLimits(
                max_source_bytes=limits.max_source_bytes, max_source_duration_ms=limits.max_source_duration_ms
            ),
            deadline=self._deadline,
        )
        self._bytes_read = preflight.read.bytes_read
        self._etag_observed = True
        decision = preflight.decision
        if not decision.admissible:
            # The first refusal is the failure; the Hub still learns the others, as far as a result holds them.
            self._refusal_diagnostics = tuple(
                Diagnostic(code=refusal.code, detail=make_detail(refusal.detail))
                for refusal in decision.refusals[1 : 1 + CONTRACT_MAX_DIAGNOSTICS]
            )
            raise WorkerFailure(decision.refusals[0])
        return self._planned(preflight.facts, profile)

    def _planned(self, facts: SourceFacts, profile: MediaProfile) -> _Plan:
        """The renditions and their room, from the facts and the limits alone: whatever cannot be planned
        or does not fit fails here, in the probing stage, before the source is downloaded."""
        dispatch = self._dispatch
        renditions = plan_renditions(
            facts.video_streams[0],
            has_audio=bool(facts.audio_streams),
            requested=dispatch.renditions,
            profile=profile,
        )
        job = OutputAllowance.for_job(
            max_output_bytes=dispatch.limits.max_output_bytes,
            max_artifact_count=dispatch.limits.max_artifact_count,
            source_size_bytes=dispatch.source.size_bytes,
            profile=profile,
        )
        plan = _Plan(
            facts=facts,
            renditions=renditions,
            allowance=reserve_package(job, profile),
            manifest_max_bytes=manifest_max_bytes(job.max_artifacts),
        )
        self._progress.renditions_planned(len(renditions))
        return plan

    def _download(self, source: S3RangeSource) -> _Scratch:
        self._enter(STAGE_DOWNLOADING)
        scratch = self._make_scratch()
        probed_bytes = self._bytes_read

        def downloaded(total_bytes: int) -> None:
            self._bytes_read = probed_bytes + total_bytes
            self._progress.downloaded(total_bytes, source.size_bytes)

        download_source(
            source,
            destination=scratch.source_path,
            deadline=self._deadline,
            sleep=self._runtime.sleep,
            on_bytes=downloaded,
        )
        return scratch

    def _make_scratch(self) -> _Scratch:
        root = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=self._job_settings.scratch_root))
        # Recorded before anything is made inside it, so the cleanup finds it whatever fails next.
        self._scratch_root = root
        scratch = _Scratch(
            source_path=root / SOURCE_DIR_NAME / SOURCE_FILE_NAME,
            output_root=root / OUTPUT_DIR_NAME,
            work_dir=root / WORK_DIR_NAME,
        )
        for directory in (scratch.source_path.parent, scratch.output_root, scratch.work_dir):
            directory.mkdir()
        return scratch

    def _transcode(self, scratch: _Scratch, plan: _Plan, profile: MediaProfile) -> _Encoded:
        self._enter(STAGE_TRANSCODING)
        source = EncodeSource.from_facts(scratch.source_path, plan.facts, profile)
        allowance = plan.allowance
        renditions: list[RenditionOutput] = []
        # One at a time, never in parallel: on the shared scratch disk a failed sibling's removal could
        # free the space a rendition's free-disk check relies on. Each takes what the earlier ones left.
        for planned in plan.renditions:
            output = encode_rendition(
                source,
                planned,
                profile=profile,
                output_root=scratch.output_root,
                work_dir=scratch.work_dir,
                media_key=self._grant.media_key,
                allowance=allowance,
                deadline=self._deadline,
            )
            allowance = allowance.after(output)
            renditions.append(output)
            self._progress.rendition_done()
        return _Encoded(
            source=source,
            largest_plan=plan.renditions[0],
            renditions=tuple(renditions),
            manifest_max_bytes=plan.manifest_max_bytes,
        )

    def _finalize(
        self, scratch: _Scratch, encoded: _Encoded, profile: MediaProfile, encoder: Encoder
    ) -> _Generation:
        self._enter(STAGE_FINALIZING)
        dispatch, identity = self._dispatch, self._grant.identity
        named_profile, encryption = dispatch.profile, dispatch.encryption
        thumbnail = make_thumbnail(
            encoded.source,
            encoded.largest_plan,
            profile=profile,
            output_root=scratch.output_root,
            work_dir=scratch.work_dir,
            deadline=self._deadline,
        )
        master = write_master_playlist(encoded.renditions, output_root=scratch.output_root)
        manifest = write_manifest(
            identity=ManifestIdentity(
                org_uuid=identity.org_uuid,
                video_uuid=identity.video_uuid,
                source_id=dispatch.source.source_id,
                attempt_id=identity.attempt_id,
                dispatch_id=identity.dispatch_id,
                execution_id=identity.execution_id,
            ),
            profile_ref=ProfileRef(id=named_profile.id, version=named_profile.version, sha256=named_profile.sha256),
            profile=profile,
            encryption=EncryptionRef(mode=encryption.mode, media_key_id=encryption.media_key_id),
            requested_renditions=dispatch.renditions,
            renditions=encoded.renditions,
            master=master,
            thumbnail=thumbnail,
            created_at=self._runtime.clock(),
            output_root=scratch.output_root,
            max_bytes=encoded.manifest_max_bytes,
        )
        self._require_valid_manifest(scratch.output_root / manifest.path)
        return _Generation(
            manifest=manifest, master=master, thumbnail=thumbnail, renditions=encoded.renditions, encoder=encoder
        )

    def _require_valid_manifest(self, manifest_path: Path) -> None:
        """The manifest's self-check: the worker's own validator must accept the file before anything is uploaded."""
        dispatch, identity = self._dispatch, self._grant.identity
        context = TrustedContext.from_dict(
            {
                "context_version": messages.CONTEXT_VERSION,
                "role": Role.WORKER_SENDER.value,
                "channel": Channel.STORAGE.value,
                "accepted_manifest_versions": [MANIFEST_VERSION],
                "accepted_message_kinds": [MessageKind.GENERATION_MANIFEST.value],
                "expect": {
                    "org_uuid": identity.org_uuid,
                    "video_uuid": identity.video_uuid,
                    "attempt_id": identity.attempt_id,
                    "dispatch_id": identity.dispatch_id,
                    "execution_id": identity.execution_id,
                    "source_id": dispatch.source.source_id,
                    "profile": dispatch.profile.to_dict(),
                    "encryption": dispatch.encryption.to_dict(),
                    "renditions": list(dispatch.renditions),
                    "max_artifact_count": dispatch.limits.max_artifact_count,
                },
            }
        )
        verdict = pipeline.validate(manifest_path.read_bytes(), context)
        if verdict.accepted:
            return
        # The layer and the reason only: the pointer and the manifest's content stay here.
        refused_at = f"{verdict.layer.value}/{verdict.reason.value}"  # type: ignore[union-attr]
        raise WorkerFailure(
            Failure(
                "internal_error", "manifest_self_check_failed", False,
                f"the worker's validator refuses the manifest it wrote at {refused_at}",
            )
        )

    def _upload(self, client: Any, scratch: _Scratch, generation: _Generation) -> None:
        self._enter(STAGE_UPLOADING)
        manifest = generation.manifest
        generation_bytes = manifest.total_bytes + manifest.size_bytes

        def uploaded(total_bytes: int) -> None:
            self._progress.uploaded(total_bytes, generation_bytes)

        upload_generation(
            client,
            bucket=self._dispatch.output.bucket,
            prefix=self._grant.output_prefix,
            output_root=scratch.output_root,
            artifacts=generation.artifacts,
            manifest=manifest,
            deadline=self._deadline,
            sleep=self._runtime.sleep,
            on_bytes=uploaded,
        )

    # --- the two endings ---------------------------------------------------------------------------

    def _source_read(self) -> SourceRead:
        located = self._dispatch.source
        # How much of the object was read, not how often: the preflight and the download read the same
        # bytes, and the contract bounds `bytes_read` by the largest source's size.
        bytes_read = min(self._bytes_read, located.size_bytes)
        return SourceRead(source_id=located.source_id, bytes_read=bytes_read, etag_observed=located.etag)

    def _completed(self, generation: _Generation) -> SealedMessage:
        dispatch, grant, settings = self._dispatch, self._grant, self._runtime.settings
        manifest = generation.manifest
        now = self._runtime.clock()
        thumbnail_diagnostic = generation.thumbnail.diagnostic
        message = messages.result_completed(
            protocol_version=dispatch.protocol_version,
            message_id=self._runtime.new_uuid(),
            sent_at=now,
            key_id=settings.worker_to_hub_key_id,
            identity=grant.identity,
            generation_id=grant.generation_id,
            manifest=ManifestLocator(
                location_id=grant.output_location_id,
                path=manifest.path,
                sha256=manifest.sha256,
                size_bytes=manifest.size_bytes,
                artifact_count=manifest.artifact_count,
                total_bytes=manifest.total_bytes,
            ),
            source=self._source_read(),
            profile=dispatch.profile,
            media=_media(generation.renditions),
            encoder=generation.encoder,
            diagnostics=() if thumbnail_diagnostic is None else (thumbnail_diagnostic,),
        )
        context = messages.worker_sender_context(
            MessageKind.TRANSCODE_RESULT_COMPLETED,
            grant.identity,
            settings=settings,
            now=now,
            source_id=dispatch.source.source_id,
            output_location_id=grant.output_location_id,
            profile=dispatch.profile,
        )
        try:
            sealed = seal_message(message, context, settings=settings)
        except SelfCheckFailed as refused:
            # Its text is the outbox's own: the kind, the layer and the reason, nothing of the message.
            raise WorkerFailure(
                Failure("internal_error", "result_self_check_failed", False, make_detail(str(refused)))
            ) from None
        logger.info(
            "execution %s completed: %d artifacts, %d bytes",
            grant.identity.execution_id, manifest.artifact_count, manifest.total_bytes,
        )
        return sealed

    def _failed(self, failure: Failure) -> SealedMessage:
        dispatch, grant, settings = self._dispatch, self._grant, self._runtime.settings
        now = self._runtime.clock()
        message = messages.result_failed(
            protocol_version=dispatch.protocol_version,
            message_id=self._runtime.new_uuid(),
            sent_at=now,
            key_id=settings.worker_to_hub_key_id,
            identity=grant.identity,
            failure=failure,
            stage=self._stage,
            source=self._source_read() if self._etag_observed else None,
            encoder=self._encoder,
            diagnostics=self._refusal_diagnostics,
        )
        context = messages.worker_sender_context(
            MessageKind.TRANSCODE_RESULT_FAILED,
            grant.identity,
            settings=settings,
            now=now,
            source_id=dispatch.source.source_id,
        )
        return seal_message(message, context, settings=settings)

    def _remove_scratch(self) -> None:
        if self._scratch_root is None:
            return
        try:
            shutil.rmtree(self._scratch_root)
        except Exception as error:  # noqa: BLE001 - a leftover directory never changes the execution's result
            logger.warning("the execution's scratch directory could not be removed: %s", type(error).__name__)
