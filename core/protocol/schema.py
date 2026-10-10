"""Layer L3: JSON Schema 2020-12 validation against the local contract tree only.

Every schema of schemas/ and carriers/ is registered under its $id. jsonschema combines this
registry with its bundled copies of the json-schema.org metaschemas, which therefore resolve
without the network; a $ref that names anything else fails because the registry refuses to
retrieve. The verdict uses the FIRST error jsonschema reports (bounded work on hostile input);
error_pointers() lists every error, sub-errors flattened, for tests and diagnostics only.
This module is the only one that imports jsonschema/referencing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Iterator, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource, Unresolvable
from referencing.jsonschema import DRAFT202012

from . import files

SCHEMA_BASE_URI = "https://schemas.video-hub.invalid/transcode/v2/"
SCHEMA_DIRECTORIES = ("schemas", "carriers")
SCHEMA_SUFFIX = ".schema.json"


class UnresolvableReference(Exception):
    """A $ref that names nothing inside the contract tree."""


def _refuse_retrieval(uri: str) -> Resource:
    raise NoSuchResource(ref=uri)


def schema_files() -> list[str]:
    """Relative paths of every schema file of the tree."""
    return [
        relative_path
        for relative_path in files.list_tree_files()
        if relative_path.split("/")[0] in SCHEMA_DIRECTORIES and relative_path.endswith(SCHEMA_SUFFIX)
    ]


@lru_cache(maxsize=1)
def local_registry() -> Registry:
    """Every schema of the tree, keyed by its $id, which must equal the base URI plus its relative path."""
    resources: list[tuple[str, Resource]] = []
    for relative_path in schema_files():
        contents = files.read_json(relative_path)
        expected_id = SCHEMA_BASE_URI + relative_path
        if not isinstance(contents, dict) or contents.get("$id") != expected_id:
            raise ValueError(f"{relative_path} must declare $id {expected_id}")
        resources.append((expected_id, Resource.from_contents(contents, default_specification=DRAFT202012)))
    return Registry(retrieve=_refuse_retrieval).with_resources(resources).crawl()


@lru_cache(maxsize=None)
def validator_for(relative_path: str) -> Draft202012Validator:
    """A validator for one schema file of the tree."""
    schema_uri = SCHEMA_BASE_URI + relative_path
    try:
        contents = local_registry().contents(schema_uri)
    except (NoSuchResource, Unresolvable, LookupError) as error:
        raise UnresolvableReference(f"no schema {relative_path} in the contract tree") from error
    return Draft202012Validator(contents, registry=local_registry())


def json_pointer(path: Iterable[Any]) -> str:
    """RFC 6901 pointer of an instance path ("" for the root)."""
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def _flatten(errors: Iterable[ValidationError]) -> Iterator[ValidationError]:
    for error in errors:
        yield error
        yield from _flatten(error.context or ())


def _pointers(validator: Draft202012Validator, instance: Any) -> list[str]:
    try:
        return [json_pointer(error.absolute_path) for error in _flatten(validator.iter_errors(instance))]
    except Unresolvable as error:
        raise UnresolvableReference(str(error)) from error


def first_error_pointer(relative_path: str, instance: Any) -> str | None:
    """The instance pointer of the first error jsonschema reports, or None when the instance is valid.

    Validation stops at that error: nothing after it is evaluated.
    """
    try:
        for error in validator_for(relative_path).iter_errors(instance):
            return json_pointer(error.absolute_path)
    except Unresolvable as error:
        raise UnresolvableReference(str(error)) from error
    return None


def error_pointers(relative_path: str, instance: Any) -> list[str]:
    """Pointers of every reported error, sub-errors flattened, in report order; empty when valid."""
    return _pointers(validator_for(relative_path), instance)


def error_pointers_for_schema(schema: Mapping[str, Any], instance: Any) -> list[str]:
    """Validate against an ad-hoc schema that may $ref the tree: one definition of it (the settings and the
    message builders), or a lint test's own schema."""
    return _pointers(Draft202012Validator(dict(schema), registry=local_registry()), instance)


def resolve_reference(base_uri: str, reference: str) -> Any:
    """The contents a $ref names, resolved against base_uri inside the local registry only."""
    try:
        return local_registry().resolver(base_uri=base_uri).lookup(reference).contents
    except Unresolvable as error:
        raise UnresolvableReference(f"{reference!r} from {base_uri!r} names nothing in the tree") from error
