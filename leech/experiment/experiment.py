import json
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union
import re
import sys

from .migrations import (
    CONFIG_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    SYSTEM_DEVICE_ID,
    SYSTEM_DEVICE_TYPE,
    MigrationError,
    UnsupportedSchemaError,
    is_system_device_name,
    migrate_config_data,
    migrate_device_config,
    migrate_run_data,
    new_device_id,
)


def _atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


@dataclass
class ExperimentMetadata:
    experiment_name: str = ""
    version: str = "1.0.0"
    author: str = ""
    created_at: str = ""
    description: str = ""
    cloned_from: Optional[str] = None
    extra: dict = field(default_factory=dict, repr=False, compare=False)


@dataclass
class ExecutionControl:
    required_devices: List[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict, repr=False, compare=False)


@dataclass
class SequenceStep:
    step_id: int = 1
    action: str = ""
    parameters: dict = field(default_factory=dict)
    device_name: str = ""
    device_id: str = ""
    operation_id: str = ""
    extra: dict = field(default_factory=dict, repr=False, compare=False)


@dataclass
class PostProcessingScript:
    script_id: int = 1
    name: str = ""
    environment: str = "Python"
    script_path: str = ""
    enabled_by_default: bool = True
    extra: dict = field(default_factory=dict, repr=False, compare=False)


@dataclass
class ExperimentConfig:
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    execution_control: ExecutionControl = field(default_factory=ExecutionControl)
    sequence: List[SequenceStep] = field(default_factory=list)
    post_processing: List[PostProcessingScript] = field(default_factory=list)
    devices: List[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict, repr=False, compare=False)
    migration_changed: bool = field(default=False, repr=False, compare=False)
    migration_warnings: List[str] = field(default_factory=list, repr=False, compare=False)


def _extra_fields(data, known_fields):
    return deepcopy({
        key: value for key, value in data.items() if key not in known_fields
    })


def _merge_fields(extra, known_fields):
    result = deepcopy(extra or {})
    result.update(deepcopy(known_fields))
    return result


def _default_config(name: str, author: str = "", description: str = "") -> dict:
    return _config_to_dict(ExperimentConfig(
        metadata=ExperimentMetadata(
            experiment_name=name,
            author=author,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    ))


def _config_to_dataclass(data: dict) -> ExperimentConfig:
    metadata = data.get("metadata", {})
    execution_control = data.get("execution_control", {})
    sequence = data.get("sequence", [])
    post_processing = data.get("post_processing", [])
    if not isinstance(metadata, dict):
        raise MigrationError("config metadata must be an object")
    if not isinstance(execution_control, dict):
        raise MigrationError("config execution_control must be an object")
    if not isinstance(post_processing, list):
        raise MigrationError("config post_processing must be a list")
    metadata_fields = {
        "experiment_name", "version", "author", "created_at", "description", "cloned_from"
    }
    execution_fields = {"required_devices"}
    step_fields = {"step_id", "action", "parameters", "device_name", "device_id", "operation_id"}
    post_processing_fields = {
        "script_id", "name", "environment", "script_path", "enabled_by_default"
    }
    return ExperimentConfig(
        metadata=ExperimentMetadata(
            experiment_name=metadata.get("experiment_name", ""),
            version=metadata.get("version", "1.0.0"),
            author=metadata.get("author", ""),
            created_at=metadata.get("created_at", ""),
            description=metadata.get("description", ""),
            cloned_from=metadata.get("cloned_from"),
            extra=_extra_fields(metadata, metadata_fields),
        ),
        execution_control=ExecutionControl(
            required_devices=list(execution_control.get("required_devices", [])),
            extra=_extra_fields(execution_control, execution_fields),
        ),
        sequence=[
            SequenceStep(
                step_id=step.get("step_id", index + 1),
                action=step.get("action", ""),
                parameters=deepcopy(step.get("parameters", {})),
                device_name=step.get("device_name", ""),
                device_id=step.get("device_id", ""),
                operation_id=step.get("operation_id", step.get("action", "")),
                extra=_extra_fields(step, step_fields),
            )
            for index, step in enumerate(sequence)
        ],
        post_processing=[
            PostProcessingScript(
                script_id=script.get("script_id", index + 1),
                name=script.get("name", ""),
                environment=script.get("environment", "Python"),
                script_path=script.get("script_path", ""),
                enabled_by_default=script.get("enabled_by_default", True),
                extra=_extra_fields(script, post_processing_fields),
            )
            for index, script in enumerate(post_processing)
        ],
        devices=deepcopy(data.get("devices", [])),
        extra=_extra_fields(data, {
            "schema_version", "metadata", "execution_control", "sequence",
            "post_processing", "devices",
        }),
    )


def normalize_device_record(record: dict, index: int = 0) -> dict:
    if not isinstance(record, dict):
        raise MigrationError(f"devices[{index}] must be an object")
    item = deepcopy(record)
    name = item.get("name", "")
    device_type = item.get("device_type", "unknown")
    if not isinstance(name, str) or not name:
        name = f"Device {index + 1}"
    if not isinstance(device_type, str) or not device_type:
        device_type = "unknown"
    device_id = item.get("device_id") or item.get("id") or new_device_id()
    config_data = item.get("config", {})
    if config_data is None:
        config_data = {}
    if not isinstance(config_data, dict):
        raise MigrationError(f"devices[{index}].config must be an object")
    config_version = item.get("config_version", 1)
    if isinstance(config_version, bool) or not isinstance(config_version, int) or config_version < 0:
        config_version = 1
    item.update({
        "device_id": device_id,
        "name": name,
        "device_type": device_type,
        "config": deepcopy(config_data),
        "config_version": config_version,
    })
    return item


def _config_to_dict(config: ExperimentConfig) -> dict:
    devices = [
        normalize_device_record(device, index)
        for index, device in enumerate(config.devices)
    ]
    name_ids = {}
    for device in devices:
        name_ids.setdefault(device["name"], []).append(device["device_id"])

    sequence = []
    for step in config.sequence:
        device_name = step.device_name
        device_id = step.device_id
        if device_id == SYSTEM_DEVICE_ID or (
            not device_id and is_system_device_name(device_name)
        ):
            device_id = SYSTEM_DEVICE_ID
            device_name = ""
        elif not device_id and len(name_ids.get(device_name, [])) == 1:
            device_id = name_ids[device_name][0]
        item = _merge_fields(step.extra, {
            "step_id": step.step_id,
            "action": step.action,
            "operation_id": step.operation_id or step.action,
            "parameters": deepcopy(step.parameters),
            "device_name": device_name,
            "device_id": device_id,
        })
        sequence.append(item)

    metadata = _merge_fields(config.metadata.extra, {
        "experiment_name": config.metadata.experiment_name,
        "version": config.metadata.version,
        "author": config.metadata.author,
        "created_at": config.metadata.created_at,
        "description": config.metadata.description,
        "cloned_from": config.metadata.cloned_from,
    })
    execution_control = _merge_fields(config.execution_control.extra, {
        "required_devices": list(config.execution_control.required_devices),
    })
    post_processing = [
        _merge_fields(script.extra, {
            "script_id": script.script_id,
            "name": script.name,
            "environment": script.environment,
            "script_path": script.script_path,
            "enabled_by_default": script.enabled_by_default,
        })
        for script in config.post_processing
    ]
    return _merge_fields(config.extra, {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "metadata": metadata,
        "execution_control": execution_control,
        "sequence": sequence,
        "post_processing": post_processing,
        "devices": devices,
    })


def _device_storage_format(device) -> str:
    if not isinstance(device, dict):
        return "csv"
    device_type = str(device.get("device_type", "")).lower()
    if "mcs" in device_type:
        return "hdf5"
    config = device.get("config") if isinstance(device.get("config"), dict) else {}
    mode = config.get("storage_mode")
    if isinstance(mode, str) and mode in ("rhs", "csv", "hdf5"):
        return mode
    return "csv"


def _device_data_dir(device) -> str:
    name = str(device.get("name", "device"))
    device_id = device.get("device_id")
    key = f"{name}_{device_id}" if device_id else name
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._") or "device"
    return f"raw/{key}"


def _storage_format(devices) -> str:
    for device in devices:
        if isinstance(device, dict):
            return _device_storage_format(device)
    return "csv"


def resolve_storage(device_record, run_path, device_dir=None):
    """Resolve (format, chunks_dir) per device; first hit wins."""
    run_path = Path(run_path)
    metadata = {}
    for name in ("run.json", "metadata.json"):
        path = run_path / name
        if path.exists():
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metadata = {}
            break
    run_storage = metadata.get("storage") if isinstance(metadata.get("storage"), dict) else {}
    record = device_record if isinstance(device_record, dict) else {}
    device_storage = record.get("storage") if isinstance(record.get("storage"), dict) else {}

    fmt = device_storage.get("format")
    if not isinstance(fmt, str) or not fmt:
        fmt = _device_storage_format(record) if record else None

    if device_dir is not None:
        candidates = [Path(device_dir)]
    else:
        data_dir = device_storage.get("data_dir") or run_storage.get("data_dir")
        candidates = []
        if isinstance(data_dir, str) and data_dir:
            candidates.append(run_path / data_dir)
        candidates.extend([run_path / "raw", run_path / "raw_chunks"])

    chunks_dir = None
    for candidate in candidates:
        if candidate.is_dir():
            if any(candidate.rglob("chunk_*.h5")) or any(candidate.rglob("chunk_*.csv")):
                chunks_dir = candidate
                break
    if chunks_dir is None:
        for candidate in candidates:
            if candidate.is_dir():
                chunks_dir = candidate
                break
    if chunks_dir is None:
        chunks_dir = run_path / "raw"

    has_h5 = any(chunks_dir.rglob("chunk_*.h5"))
    has_csv = any(chunks_dir.rglob("chunk_*.csv"))
    # Files on disk outrank metadata: mixed runs used to record the first
    # device's format at run level and mislabel the rest.
    if has_h5 and not has_csv:
        fmt = "hdf5"
    elif has_csv and not has_h5:
        fmt = "csv"
    elif not fmt:
        fmt = run_storage.get("format") if isinstance(run_storage.get("format"), str) else None
    if not fmt:
        raise FileNotFoundError(f"No storage format or chunk files in {chunks_dir}")
    return fmt, chunks_dir


class ExperimentManager:

    @staticmethod
    def create(
        experiments_dir: str | Path,
        name: str,
        author: str = "",
        description: str = "",
    ) -> Path:
        experiments_dir = Path(experiments_dir).resolve()
        experiments_dir.mkdir(parents=True, exist_ok=True)
        experiment_path = experiments_dir / name
        config_path = experiment_path / "config.json"
        if config_path.exists():
            raise FileExistsError(f"Experiment already exists: {experiment_path}")
        experiment_path.mkdir(parents=True, exist_ok=True)
        config_data = _default_config(name, author, description)
        _atomic_write_json(config_path, config_data)
        return experiment_path

    @staticmethod
    def load(experiment_path: str | Path) -> ExperimentConfig:
        config_path = Path(experiment_path) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        migrated, report = migrate_config_data(data)
        config = _config_to_dataclass(migrated)
        changed = report.changed
        from leech.device.registry import DEVICE_CLASSES
        for record in config.devices:
            device_class = DEVICE_CLASSES.get(record.get("device_type"))
            if device_class is None:
                continue
            record["config"], record["config_version"], device_changed = migrate_device_config(
                device_class,
                record.get("config", {}),
                record.get("config_version", 1),
            )
            changed = changed or device_changed
        config.migration_changed = changed
        config.migration_warnings = list(report.warnings)
        return config

    @staticmethod
    def save(experiment_path: str | Path, config: ExperimentConfig, backup: bool = False):
        config_path = Path(experiment_path) / "config.json"
        data, _ = migrate_config_data(_config_to_dict(config))
        backup_path = config_path.with_name(f"{config_path.name}.bak")
        if backup and config_path.exists() and not backup_path.exists():
            shutil.copy2(config_path, backup_path)
        _atomic_write_json(config_path, data)

    @staticmethod
    def list_experiments(experiments_dir: str | Path) -> list[Path]:
        base = Path(experiments_dir)
        if not base.exists():
            return []
        return sorted(
            p for p in base.iterdir()
            if p.is_dir() and (p / "config.json").exists()
        )

    @staticmethod
    def delete(experiment_path: str | Path):
        path = Path(experiment_path)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)

    @staticmethod
    def start_run(experiment_path: str | Path, run_name: str) -> Path:
        exp_path = Path(experiment_path)
        runs_dir = exp_path / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_path = runs_dir / run_name
        run_path.mkdir(parents=True, exist_ok=True)
        if any(run_path.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {run_path}")
        return run_path

    @staticmethod
    def init_run(run_path: str | Path, config_data: dict, devices: list, sequence: list) -> Path:
        run_path = Path(run_path)
        devices = deepcopy(devices)
        for device in devices:
            if isinstance(device, dict) and "storage" not in device:
                device["storage"] = {
                    "format": _device_storage_format(device),
                    "data_dir": _device_data_dir(device),
                }
        run_json = {
            "experiment": {
                "name": config_data.get("metadata", {}).get("experiment_name", ""),
                "version": config_data.get("metadata", {}).get("version", "1.0.0"),
                "author": config_data.get("metadata", {}).get("author", ""),
                "description": config_data.get("metadata", {}).get("description", ""),
                "created_at": config_data.get("metadata", {}).get("created_at", ""),
            },
            "run": {
                "name": run_path.name,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": None,
                "status": "running",
                "device_count": len(devices),
                "step_count": len(sequence),
            },
            "devices": devices,
            "sequence": sequence,
            "storage": {
                "format": _storage_format(devices),
                "data_dir": "raw",
            },
        }
        migrated, _ = migrate_run_data(run_json)
        path = run_path / "run.json"
        _atomic_write_json(path, migrated)
        return path

    @staticmethod
    def _run_metadata_path(run_path: str | Path) -> Path:
        run_path = Path(run_path)
        for name in ("run.json", "metadata.json"):
            path = run_path / name
            if path.exists():
                return path
        raise FileNotFoundError(f"No run metadata in {run_path}")

    @staticmethod
    def update_run(run_path: str | Path, status: str, steps: list = None, error_count: int = 0):
        path = ExperimentManager._run_metadata_path(run_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data, _ = migrate_run_data(data)
        if isinstance(data.get("run"), dict):
            run = data["run"]
        else:
            run = data
        if status != "running":
            run["end_time"] = datetime.now(timezone.utc).isoformat()
        run["status"] = status
        if error_count:
            run["error_count"] = error_count
        if steps:
            run["steps"] = steps
        _atomic_write_json(path, data)

    @staticmethod
    def load_run(run_path: str | Path) -> dict:
        path = ExperimentManager._run_metadata_path(run_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        migrated, _ = migrate_run_data(data)
        return migrated

    @staticmethod
    def list_runs(experiment_path: str | Path) -> list[Path]:
        runs_dir = Path(experiment_path) / "runs"
        if not runs_dir.exists():
            return []
        return sorted(p for p in runs_dir.iterdir() if p.is_dir())

    @staticmethod
    def clone_experiment(experiment_path: str | Path, new_name: str) -> Path:
        src = Path(experiment_path)
        dst = src.parent / new_name
        shutil.copytree(src, dst)
        config = ExperimentManager.load(dst)
        config.metadata.experiment_name = new_name
        config.metadata.cloned_from = str(src.name)
        ExperimentManager.save(dst, config, backup=config.migration_changed)
        return dst

    @staticmethod
    def rename_run(run_path: str | Path, new_name: str) -> Path:
        src = Path(run_path)
        dst = src.parent / new_name
        if dst.exists():
            raise FileExistsError(f"Run '{new_name}' already exists")
        meta_file = ExperimentManager._run_metadata_path(src)
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        data, _ = migrate_run_data(data)
        if isinstance(data.get("run"), dict) and "name" in data["run"]:
            data["run"]["name"] = new_name
        elif "name" in data:
            data["name"] = new_name
        src.rename(dst)
        _atomic_write_json(dst / meta_file.name, data)
        return dst

    @staticmethod
    def delete_run(run_path: str | Path):
        path = Path(run_path)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
