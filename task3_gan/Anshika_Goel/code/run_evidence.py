"""Raw run-evidence utilities for Task 3 CycleGAN.

This module owns durable experiment evidence only:

- one atomic JSON manifest per run;
- one append-only JSONL event stream per run;
- strict finite JSON serialization;
- monotonically increasing event sequence numbers;
- safe continuation of an existing event stream after epoch-boundary
  training resume;
- runtime and hardware disclosure without usernames or hostnames.

It does not discover data, build models, train networks, save model
checkpoints, generate images, invoke Kaggle, or choose experiment
hyperparameters.

The JSONL stream is intended to remain raw evidence. Downstream summary
tables and plots should be derived from it rather than replacing it.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


__all__ = [
    "RunEvidenceWriter",
    "atomic_write_json",
    "collect_runtime_environment",
    "read_run_events",
    "utc_now_iso",
]


_SCHEMA_VERSION = 1

_RUN_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)


def utc_now_iso() -> str:
    """Return a timezone-explicit UTC timestamp."""

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat(
            timespec="milliseconds"
        )
        .replace(
            "+00:00",
            "Z",
        )
    )


def _validate_run_id(
    run_id: str,
) -> str:

    if not isinstance(
        run_id,
        str,
    ):
        raise TypeError(
            "run_id must be str"
        )

    if not _RUN_ID_PATTERN.fullmatch(
        run_id
    ):
        raise ValueError(
            "run_id must contain only letters, numbers, '.', '_', or '-' "
            "and must begin with a letter or number"
        )

    return run_id


def _normalize_json(
    value: Any,
    *,
    path: str = "$",
) -> Any:
    """Return a strict JSON-safe copy and reject non-finite numbers."""

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(
            value
        ):
            raise ValueError(
                "Non-finite float is not permitted at "
                + path
            )

        return value

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        Mapping,
    ):

        normalized: dict[
            str,
            Any,
        ] = {}

        for key, item in value.items():

            if not isinstance(
                key,
                str,
            ):
                raise TypeError(
                    "JSON mapping keys must be strings at "
                    + path
                )

            normalized[
                key
            ] = _normalize_json(
                item,
                path=(
                    path
                    + "."
                    + key
                ),
            )

        return normalized

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):

        return [
            _normalize_json(
                item,
                path=(
                    path
                    + "["
                    + str(
                        index
                    )
                    + "]"
                ),
            )
            for index, item
            in enumerate(
                value
            )
        ]

    raise TypeError(
        "Unsupported JSON evidence type at "
        + path
        + ": "
        + type(
            value
        ).__name__
    )


def _json_bytes(
    value: Any,
    *,
    newline: bool,
) -> bytes:

    normalized = _normalize_json(
        value
    )

    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    if newline:
        text += "\n"

    return text.encode(
        "utf-8"
    )


def _atomic_write_bytes(
    path: Path,
    data: bytes,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=(
            path.name
            + "."
        ),
        suffix=".tmp",
        dir=str(
            path.parent
        ),
    )

    temporary_path = Path(
        temporary_name
    )

    try:

        with os.fdopen(
            descriptor,
            "wb",
        ) as handle:

            handle.write(
                data
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary_path,
            path,
        )

    except BaseException:

        try:
            temporary_path.unlink(
                missing_ok=True
            )
        finally:
            raise


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """Atomically write one strict JSON object."""

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            "payload must be a mapping"
        )

    destination = Path(
        path
    )

    _atomic_write_bytes(
        destination,
        _json_bytes(
            payload,
            newline=True,
        ),
    )


def _create_empty_file_exclusive(
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "xb",
    ) as handle:

        handle.flush()

        os.fsync(
            handle.fileno()
        )


def _append_jsonl(
    path: Path,
    record: Mapping[str, Any],
) -> None:

    data = _json_bytes(
        record,
        newline=True,
    )

    with path.open(
        "ab",
    ) as handle:

        handle.write(
            data
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )


def _read_json_object(
    path: Path,
) -> dict[str, Any]:

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Invalid JSON file: "
            + str(
                path
            )
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            "Expected JSON object: "
            + str(
                path
            )
        )

    return value


def read_run_events(
    path: str | Path,
    *,
    expected_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read and structurally validate an append-only run-event stream."""

    source = Path(
        path
    )

    if not source.is_file():
        raise FileNotFoundError(
            "Run event log does not exist: "
            + str(
                source
            )
        )

    if expected_run_id is not None:
        expected_run_id = _validate_run_id(
            expected_run_id
        )

    records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    with source.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):

            if not raw_line.endswith(
                "\n"
            ):
                raise RuntimeError(
                    "Run event log ends with an incomplete record at line "
                    + str(
                        line_number
                    )
                )

            line = raw_line[:-1]

            if not line:
                raise RuntimeError(
                    "Blank JSONL record at line "
                    + str(
                        line_number
                    )
                )

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSONL record at line "
                    + str(
                        line_number
                    )
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise RuntimeError(
                    "JSONL record must be an object at line "
                    + str(
                        line_number
                    )
                )

            required = {
                "schema_version",
                "run_id",
                "sequence",
                "timestamp_utc",
                "event_type",
                "payload",
            }

            if set(
                record
            ) != required:
                raise RuntimeError(
                    "Unexpected JSONL record schema at line "
                    + str(
                        line_number
                    )
                )

            if record[
                "schema_version"
            ] != _SCHEMA_VERSION:
                raise RuntimeError(
                    "Unsupported event schema version at line "
                    + str(
                        line_number
                    )
                )

            run_id = _validate_run_id(
                record[
                    "run_id"
                ]
            )

            if (
                expected_run_id is not None
                and run_id
                != expected_run_id
            ):
                raise RuntimeError(
                    "Run ID mismatch at line "
                    + str(
                        line_number
                    )
                )

            expected_sequence = (
                len(
                    records
                )
                + 1
            )

            if record[
                "sequence"
            ] != expected_sequence:
                raise RuntimeError(
                    "Non-contiguous event sequence at line "
                    + str(
                        line_number
                    )
                )

            if not isinstance(
                record[
                    "timestamp_utc"
                ],
                str,
            ) or not record[
                "timestamp_utc"
            ]:
                raise RuntimeError(
                    "Invalid timestamp at line "
                    + str(
                        line_number
                    )
                )

            if not isinstance(
                record[
                    "event_type"
                ],
                str,
            ) or not record[
                "event_type"
            ]:
                raise RuntimeError(
                    "Invalid event_type at line "
                    + str(
                        line_number
                    )
                )

            if not isinstance(
                record[
                    "payload"
                ],
                dict,
            ):
                raise RuntimeError(
                    "Event payload must be an object at line "
                    + str(
                        line_number
                    )
                )

            # Re-normalization catches NaN/Infinity parsed by Python's
            # permissive JSON decoder as well as unexpected nested types.
            _normalize_json(
                record
            )

            records.append(
                record
            )

    return records


def collect_runtime_environment(
    *,
    selected_device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Collect reproducibility-relevant runtime/hardware facts."""

    cuda_available = bool(
        torch.cuda.is_available()
    )

    cuda_device_count = (
        int(
            torch.cuda.device_count()
        )
        if cuda_available
        else 0
    )

    cuda_devices = []

    if cuda_available:

        for index in range(
            cuda_device_count
        ):

            properties = (
                torch.cuda.get_device_properties(
                    index
                )
            )

            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_mib": float(
                        properties.total_memory
                        / 1024**2
                    ),
                    "compute_capability": (
                        str(
                            properties.major
                        )
                        + "."
                        + str(
                            properties.minor
                        )
                    ),
                }
            )

    selected = (
        None
        if selected_device is None
        else str(
            torch.device(
                selected_device
            )
        )
    )

    return {
        "python_version": sys.version.split()[0],
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_devices": cuda_devices,
        "selected_device": selected,
    }


@dataclass
class RunEvidenceWriter:
    """Durable manifest + append-only raw JSONL event writer."""

    run_dir: Path
    run_id: str
    manifest_path: Path
    events_path: Path
    next_sequence: int

    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        *,
        run_id: str,
        manifest: Mapping[str, Any],
    ) -> "RunEvidenceWriter":
        """Create a new evidence stream and refuse accidental overwrite."""

        validated_run_id = _validate_run_id(
            run_id
        )

        if not isinstance(
            manifest,
            Mapping,
        ):
            raise TypeError(
                "manifest must be a mapping"
            )

        destination = Path(
            run_dir
        )

        manifest_path = (
            destination
            / "run_manifest.json"
        )

        events_path = (
            destination
            / "events.jsonl"
        )

        if manifest_path.exists():
            raise FileExistsError(
                "Run manifest already exists"
            )

        if events_path.exists():
            raise FileExistsError(
                "Run event log already exists"
            )

        manifest_envelope = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": validated_run_id,
            "created_at_utc": utc_now_iso(),
            "manifest": dict(
                manifest
            ),
        }

        # Validate the whole envelope before creating either durable file.
        _json_bytes(
            manifest_envelope,
            newline=True,
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        atomic_write_json(
            manifest_path,
            manifest_envelope,
        )

        try:
            _create_empty_file_exclusive(
                events_path
            )
        except BaseException:
            manifest_path.unlink(
                missing_ok=True
            )
            raise

        return cls(
            run_dir=destination,
            run_id=validated_run_id,
            manifest_path=manifest_path,
            events_path=events_path,
            next_sequence=1,
        )

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        run_id: str,
    ) -> "RunEvidenceWriter":
        """Resume an existing evidence stream without rewriting history."""

        validated_run_id = _validate_run_id(
            run_id
        )

        destination = Path(
            run_dir
        )

        manifest_path = (
            destination
            / "run_manifest.json"
        )

        events_path = (
            destination
            / "events.jsonl"
        )

        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Run manifest does not exist"
            )

        if not events_path.is_file():
            raise FileNotFoundError(
                "Run event log does not exist"
            )

        manifest = _read_json_object(
            manifest_path
        )

        # Python's JSON decoder accepts NaN and Infinity by default.
        # Re-apply the same strict normalization used during creation so
        # a corrupted existing manifest cannot bypass the evidence contract.
        _normalize_json(
            manifest
        )

        required_manifest_keys = {
            "schema_version",
            "run_id",
            "created_at_utc",
            "manifest",
        }

        if set(
            manifest
        ) != required_manifest_keys:
            raise RuntimeError(
                "Run manifest schema is invalid"
            )

        if manifest[
            "schema_version"
        ] != _SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported run manifest schema version"
            )

        if manifest[
            "run_id"
        ] != validated_run_id:
            raise RuntimeError(
                "Run manifest ID does not match requested run_id"
            )

        if not isinstance(
            manifest[
                "manifest"
            ],
            dict,
        ):
            raise RuntimeError(
                "Run manifest payload must be an object"
            )

        records = read_run_events(
            events_path,
            expected_run_id=(
                validated_run_id
            ),
        )

        return cls(
            run_dir=destination,
            run_id=validated_run_id,
            manifest_path=manifest_path,
            events_path=events_path,
            next_sequence=(
                len(
                    records
                )
                + 1
            ),
        )

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append exactly one durable raw event."""

        if not isinstance(
            event_type,
            str,
        ) or not event_type:
            raise ValueError(
                "event_type must be a non-empty string"
            )

        if "\n" in event_type or "\r" in event_type:
            raise ValueError(
                "event_type cannot contain a newline"
            )

        if not isinstance(
            payload,
            Mapping,
        ):
            raise TypeError(
                "payload must be a mapping"
            )

        record = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "sequence": self.next_sequence,
            "timestamp_utc": utc_now_iso(),
            "event_type": event_type,
            "payload": dict(
                payload
            ),
        }

        # Fully validate before the append operation begins.
        _json_bytes(
            record,
            newline=True,
        )

        _append_jsonl(
            self.events_path,
            record,
        )

        self.next_sequence += 1

        return _normalize_json(
            record
        )
