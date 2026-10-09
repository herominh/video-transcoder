from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from core.admission import AdmissionDecision, RequestLimits, decide_admission
from core.profile import GIB, PILOT_PROFILE, MediaProfile
from core.source_facts import AudioStreamFacts, SourceFacts, VideoStreamFacts

GENEROUS_LIMITS = RequestLimits(max_source_bytes=100 * GIB, max_source_duration_ms=100_000_000)
ORDINARY_SIZE_BYTES = 500_000_000
PROFILE_MAX_BYTES = PILOT_PROFILE.source.max_source_bytes
PROFILE_MAX_DURATION_MS = PILOT_PROFILE.source.max_duration_ms


def _video(**overrides: Any) -> VideoStreamFacts:
    fields: dict[str, Any] = {
        "index": 0,
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "sample_aspect_num": 1,
        "sample_aspect_den": 1,
        "rotation_degrees": 0,
        "display_width": 1920,
        "display_height": 1080,
        "frame_rate_num": 30000,
        "frame_rate_den": 1001,
        "field_order": "progressive",
        "color_transfer": "bt709",
        "pix_fmt": "yuv420p",
        "dolby_vision_profile": None,
    }
    fields.update(overrides)
    return VideoStreamFacts(**fields)


def _audio(index: int = 1) -> AudioStreamFacts:
    return AudioStreamFacts(index=index, codec_name="aac", channels=2)


def _facts(**overrides: Any) -> SourceFacts:
    fields: dict[str, Any] = {
        "demuxer": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration_ms": 600_000,
        "duration_estimated": False,
        "video_streams": (_video(),),
        "audio_streams": (_audio(),),
        "attached_picture_count": 0,
        "total_stream_count": 2,
    }
    fields.update(overrides)
    return SourceFacts(**fields)


def _decide(
    facts: SourceFacts,
    *,
    size_bytes: int = ORDINARY_SIZE_BYTES,
    profile: MediaProfile = PILOT_PROFILE,
    limits: RequestLimits = GENEROUS_LIMITS,
) -> AdmissionDecision:
    return decide_admission(facts, size_bytes=size_bytes, profile=profile, limits=limits)


def _classes_and_codes(decision: AdmissionDecision) -> list[tuple[str, str]]:
    return [(refusal.error_class, refusal.code) for refusal in decision.refusals]


def test_decide_when_the_source_fits_everything_should_admit_it() -> None:
    # Arrange
    facts = _facts()

    # Act
    decision = _decide(facts)

    # Assert
    assert decision.refusals == ()
    assert decision.admissible is True


@pytest.mark.parametrize(
    ("facts", "size_bytes", "expected"),
    [
        pytest.param(_facts(), PROFILE_MAX_BYTES + 1, ("input_limits_exceeded", "source_too_large"), id="1-size"),
        pytest.param(
            _facts(demuxer="flv"), ORDINARY_SIZE_BYTES, ("input_invalid", "unsupported_container"), id="2-container"
        ),
        pytest.param(
            _facts(duration_estimated=True),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "duration_unknown"),
            id="3-estimated-duration",
        ),
        pytest.param(
            _facts(duration_ms=PROFILE_MAX_DURATION_MS + 1),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "duration_exceeded"),
            id="4-duration",
        ),
        pytest.param(
            _facts(video_streams=(), total_stream_count=1),
            ORDINARY_SIZE_BYTES,
            ("input_invalid", "no_video_stream"),
            id="5-no-video",
        ),
        pytest.param(
            _facts(video_streams=(_video(), _video(index=2)), total_stream_count=3),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "too_many_video_streams"),
            id="6-video-streams",
        ),
        pytest.param(
            _facts(audio_streams=tuple(_audio(index) for index in range(1, 6)), total_stream_count=6),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "too_many_audio_streams"),
            id="7-audio-streams",
        ),
        pytest.param(
            _facts(total_stream_count=17),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "too_many_streams"),
            id="8-total-streams",
        ),
        pytest.param(
            _facts(video_streams=(_video(codec_name="mjpeg"),)),
            ORDINARY_SIZE_BYTES,
            ("input_invalid", "unsupported_video_codec"),
            id="9-codec",
        ),
        pytest.param(
            _facts(video_streams=(_video(display_width=3841, display_height=2000),)),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "resolution_exceeded"),
            id="10-resolution",
        ),
        pytest.param(
            _facts(video_streams=(_video(display_width=3840, display_height=2400),)),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "display_pixels_exceeded"),
            id="11-pixels",
        ),
        pytest.param(
            _facts(
                video_streams=(
                    _video(
                        width=8192,
                        height=2160,
                        sample_aspect_num=1,
                        sample_aspect_den=4,
                        display_width=2048,
                        display_height=2160,
                    ),
                )
            ),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "frame_size_exceeded"),
            id="11b-stored-frame-edge",
        ),
        pytest.param(
            _facts(
                video_streams=(
                    _video(
                        width=3840,
                        height=3840,
                        sample_aspect_num=1,
                        sample_aspect_den=4,
                        display_width=960,
                        display_height=3840,
                    ),
                )
            ),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "frame_size_exceeded"),
            id="11c-stored-frame-pixels",
        ),
        pytest.param(
            _facts(video_streams=(_video(frame_rate_num=120, frame_rate_den=1),)),
            ORDINARY_SIZE_BYTES,
            ("input_limits_exceeded", "frame_rate_exceeded"),
            id="12-frame-rate",
        ),
        pytest.param(
            _facts(video_streams=(_video(field_order="tt"),)),
            ORDINARY_SIZE_BYTES,
            ("input_invalid", "interlaced_unsupported"),
            id="13-interlaced",
        ),
        pytest.param(
            _facts(video_streams=(_video(color_transfer="smpte2084"),)),
            ORDINARY_SIZE_BYTES,
            ("input_invalid", "hdr_unsupported"),
            id="14-hdr",
        ),
        pytest.param(
            _facts(video_streams=(_video(dolby_vision_profile=8),)),
            ORDINARY_SIZE_BYTES,
            ("input_invalid", "hdr_unsupported"),
            id="14b-dolby-vision",
        ),
    ],
)
def test_decide_when_exactly_one_rule_is_broken_should_return_exactly_that_refusal(
    facts: SourceFacts, size_bytes: int, expected: tuple[str, str]
) -> None:
    # Arrange (facts and size come from the parameters)

    # Act
    decision = _decide(facts, size_bytes=size_bytes)

    # Assert
    assert _classes_and_codes(decision) == [expected]
    assert decision.refusals[0].retryable is False
    assert decision.admissible is False


def test_decide_when_the_request_size_limit_is_below_the_profile_should_refuse_by_the_request() -> None:
    # Arrange
    limits = RequestLimits(max_source_bytes=1_000, max_source_duration_ms=PROFILE_MAX_DURATION_MS)

    # Act
    at_limit = _decide(_facts(), size_bytes=1_000, limits=limits)
    over_limit = _decide(_facts(), size_bytes=1_001, limits=limits)

    # Assert
    assert at_limit.admissible is True
    assert _classes_and_codes(over_limit) == [("input_limits_exceeded", "source_too_large")]
    assert over_limit.refusals[0].detail == "source size 1001 bytes exceeds the limit of 1000 bytes"


def test_decide_when_the_profile_size_cap_is_below_the_request_should_refuse_by_the_profile() -> None:
    # Arrange
    limits = RequestLimits(max_source_bytes=PROFILE_MAX_BYTES * 2, max_source_duration_ms=PROFILE_MAX_DURATION_MS)

    # Act
    at_cap = _decide(_facts(), size_bytes=PROFILE_MAX_BYTES, limits=limits)
    over_cap = _decide(_facts(), size_bytes=PROFILE_MAX_BYTES + 1, limits=limits)

    # Assert
    assert at_cap.admissible is True
    assert _classes_and_codes(over_cap) == [("input_limits_exceeded", "source_too_large")]


def test_decide_when_the_request_duration_limit_is_below_the_profile_should_refuse_by_the_request() -> None:
    # Arrange
    limits = RequestLimits(max_source_bytes=PROFILE_MAX_BYTES, max_source_duration_ms=3_600_000)

    # Act
    at_limit = _decide(_facts(duration_ms=3_600_000), limits=limits)
    over_limit = _decide(_facts(duration_ms=7_260_000), limits=limits)

    # Assert
    assert at_limit.admissible is True
    assert _classes_and_codes(over_limit) == [("input_limits_exceeded", "duration_exceeded")]
    assert over_limit.refusals[0].detail == "duration 7260000 ms exceeds the limit of 3600000 ms"


def test_decide_when_the_profile_duration_cap_is_below_the_request_should_refuse_by_the_profile() -> None:
    # Arrange
    limits = RequestLimits(max_source_bytes=PROFILE_MAX_BYTES, max_source_duration_ms=PROFILE_MAX_DURATION_MS * 2)

    # Act
    at_cap = _decide(_facts(duration_ms=PROFILE_MAX_DURATION_MS), limits=limits)
    over_cap = _decide(_facts(duration_ms=PROFILE_MAX_DURATION_MS + 1), limits=limits)

    # Assert
    assert at_cap.admissible is True
    assert _classes_and_codes(over_cap) == [("input_limits_exceeded", "duration_exceeded")]


def test_decide_when_many_rules_are_broken_should_return_every_refusal_in_table_order() -> None:
    # Arrange
    video = _video(
        codec_name="mjpeg",
        width=7680,
        height=4320,
        display_width=7680,
        display_height=4320,
        frame_rate_num=120,
        frame_rate_den=1,
        field_order="bb",
        color_transfer="arib-std-b67",
    )
    facts = _facts(
        demuxer="flv",
        duration_estimated=True,
        duration_ms=PROFILE_MAX_DURATION_MS + 1,
        video_streams=(video, _video(index=1)),
        audio_streams=tuple(_audio(index) for index in range(2, 7)),
        total_stream_count=20,
    )

    # Act
    decision = _decide(facts, size_bytes=PROFILE_MAX_BYTES + 1)

    # Assert
    assert [refusal.code for refusal in decision.refusals] == [
        "source_too_large",
        "unsupported_container",
        "duration_unknown",
        "duration_exceeded",
        "too_many_video_streams",
        "too_many_audio_streams",
        "too_many_streams",
        "unsupported_video_codec",
        "resolution_exceeded",
        "display_pixels_exceeded",
        "frame_size_exceeded",
        "frame_rate_exceeded",
        "interlaced_unsupported",
        "hdr_unsupported",
    ]
    assert all(refusal.retryable is False for refusal in decision.refusals)


def test_decide_when_there_is_no_video_stream_should_refuse_only_for_the_missing_stream() -> None:
    # Arrange
    facts = _facts(video_streams=(), attached_picture_count=1, total_stream_count=2)

    # Act
    decision = _decide(facts)

    # Assert
    assert _classes_and_codes(decision) == [("input_invalid", "no_video_stream")]


def _two_video_streams_profile() -> MediaProfile:
    return dataclasses.replace(PILOT_PROFILE, source=dataclasses.replace(PILOT_PROFILE.source, max_video_streams=2))


def test_decide_when_a_second_video_stream_has_a_disallowed_codec_should_refuse_it_naming_that_stream() -> None:
    # Arrange
    facts = _facts(
        video_streams=(_video(index=0), _video(index=1, codec_name="mjpeg")),
        audio_streams=(_audio(index=2),),
        total_stream_count=3,
    )

    # Act
    decision = _decide(facts, profile=_two_video_streams_profile())

    # Assert
    assert _classes_and_codes(decision) == [("input_invalid", "unsupported_video_codec")]
    assert decision.refusals[0].detail.startswith("video stream 1: video codec mjpeg ")


def test_decide_when_several_video_streams_break_rules_should_order_refusals_by_stream_then_by_row() -> None:
    # Arrange
    first = _video(index=0, frame_rate_num=120, frame_rate_den=1, color_transfer="smpte2084")
    second = _video(index=1, codec_name="mjpeg", field_order="tt")
    facts = _facts(video_streams=(first, second), audio_streams=(_audio(index=2),), total_stream_count=3)

    # Act
    decision = _decide(facts, profile=_two_video_streams_profile())

    # Assert
    assert [(refusal.code, refusal.detail.split(":")[0]) for refusal in decision.refusals] == [
        ("frame_rate_exceeded", "video stream 0"),
        ("hdr_unsupported", "video stream 0"),
        ("unsupported_video_codec", "video stream 1"),
        ("interlaced_unsupported", "video stream 1"),
    ]


def test_decide_when_the_stream_carries_dolby_vision_should_refuse_it_naming_the_profile() -> None:
    # Arrange
    facts = _facts(video_streams=(_video(dolby_vision_profile=5),))

    # Act
    decision = _decide(facts)

    # Assert
    assert _classes_and_codes(decision) == [("input_invalid", "hdr_unsupported")]
    assert "Dolby Vision profile 5" in decision.refusals[0].detail


def test_decide_when_pq_and_dolby_vision_both_signal_hdr_should_return_one_refusal_naming_both() -> None:
    # Arrange
    facts = _facts(video_streams=(_video(color_transfer="smpte2084", dolby_vision_profile=8),))

    # Act
    decision = _decide(facts)

    # Assert
    assert _classes_and_codes(decision) == [("input_invalid", "hdr_unsupported")]
    assert "color transfer smpte2084" in decision.refusals[0].detail
    assert "Dolby Vision profile 8" in decision.refusals[0].detail


@pytest.mark.parametrize("demuxer", ["mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "mpegts", "avi"])
def test_decide_when_any_listed_demuxer_name_is_allowed_should_accept_the_container(demuxer: str) -> None:
    # Arrange
    facts = _facts(demuxer=demuxer)

    # Act
    decision = _decide(facts)

    # Assert
    assert decision.admissible is True


@pytest.mark.parametrize(("frame_rate_num", "frame_rate_den"), [(61, 1), (60, 1), (60000, 1001), (122, 2)])
def test_decide_when_the_frame_rate_is_at_or_below_the_cap_should_accept_it(
    frame_rate_num: int, frame_rate_den: int
) -> None:
    # Arrange
    facts = _facts(video_streams=(_video(frame_rate_num=frame_rate_num, frame_rate_den=frame_rate_den),))

    # Act
    decision = _decide(facts)

    # Assert
    assert decision.admissible is True


def test_decide_when_the_frame_rate_is_a_hair_above_the_cap_should_refuse_it() -> None:
    # Arrange
    facts = _facts(video_streams=(_video(frame_rate_num=61_000_001, frame_rate_den=1_000_000),))

    # Act
    decision = _decide(facts)

    # Assert
    assert _classes_and_codes(decision) == [("input_limits_exceeded", "frame_rate_exceeded")]


@pytest.mark.parametrize("field_order", ["tt", "bb", "tb", "bt"])
def test_decide_when_the_field_order_is_interlaced_should_refuse_it(field_order: str) -> None:
    # Arrange
    facts = _facts(video_streams=(_video(field_order=field_order),))

    # Act
    decision = _decide(facts)

    # Assert
    assert _classes_and_codes(decision) == [("input_invalid", "interlaced_unsupported")]


@pytest.mark.parametrize("field_order", ["progressive", "unknown"])
def test_decide_when_the_field_order_is_progressive_or_unknown_should_accept_it(field_order: str) -> None:
    # Arrange
    facts = _facts(video_streams=(_video(field_order=field_order),))

    # Act
    decision = _decide(facts)

    # Assert
    assert decision.admissible is True


@pytest.mark.parametrize("color_transfer", ["bt709", "smpte170m", "iec61966-2-1", "unknown", None])
def test_decide_when_the_transfer_is_sdr_or_unstated_should_accept_it(color_transfer: str | None) -> None:
    # Arrange
    facts = _facts(video_streams=(_video(color_transfer=color_transfer),))

    # Act
    decision = _decide(facts)

    # Assert
    assert decision.admissible is True


def test_decide_when_the_profile_allows_interlace_and_hdr_should_accept_both() -> None:
    # Arrange
    profile = dataclasses.replace(
        PILOT_PROFILE,
        source=dataclasses.replace(PILOT_PROFILE.source, reject_interlaced=False, reject_hdr=False),
    )
    facts = _facts(video_streams=(_video(field_order="tt", color_transfer="smpte2084", dolby_vision_profile=8),))

    # Act
    decision = _decide(facts, profile=profile)

    # Assert
    assert decision.admissible is True


@pytest.mark.parametrize("size_bytes", [0, -1, True])
def test_decide_when_the_size_is_not_a_positive_int_should_raise(size_bytes: int) -> None:
    # Arrange
    facts = _facts()

    # Act / Assert
    with pytest.raises(ValueError):
        _decide(facts, size_bytes=size_bytes)


@pytest.mark.parametrize(("max_bytes", "max_duration_ms"), [(0, 1), (1, 0), (-1, 1), (True, 1), (1, 1.5)])
def test_request_limits_when_a_limit_is_not_a_positive_int_should_raise(max_bytes: Any, max_duration_ms: Any) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        RequestLimits(max_source_bytes=max_bytes, max_source_duration_ms=max_duration_ms)
