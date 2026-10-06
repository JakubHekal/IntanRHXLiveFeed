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
    is_locked: bool = False
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


def migrate_system_device_names(config):
    changed = False
    for step in config.sequence:
        if step.device_id == SYSTEM_DEVICE_ID or (
            not step.device_id and is_system_device_name(step.device_name)
        ):
            step.device_name = ""
            step.device_id = SYSTEM_DEVICE_ID
            changed = True
        if not step.operation_id:
            step.operation_id = step.action
    return changed


def _default_config(name: str, author: str = "", description: str = "") -> dict:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "metadata": {
            "experiment_name": name,
            "version": "1.0.0",
            "author": author,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "cloned_from": None,
        },
        "execution_control": {
            "is_locked": False,
            "required_devices": [],
        },
        "sequence": [],
        "post_processing": [],
        "devices": [],
    }


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
    execution_fields = {"is_locked", "required_devices"}
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
            is_locked=execution_control.get("is_locked", False),
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


def _config_to_dict(config: ExperimentConfig) -> dict:
    devices = []
    name_ids = {}
    for index, device in enumerate(config.devices):
        if not isinstance(device, dict):
            raise MigrationError(f"devices[{index}] must be an object")
        item = deepcopy(device)
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
        devices.append(item)
        name_ids.setdefault(name, []).append(device_id)

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
        "is_locked": config.execution_control.is_locked,
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
        experiment_path.mkdir(parents=True, exist_ok=True)
        config_data = _default_config(name, author, description)
        config_path = experiment_path / "config.json"
        _atomic_write_json(config_path, config_data)
        return experiment_path

    @staticmethod
    def load(experiment_path: str | Path) -> ExperimentConfig:
        config_path = Path(experiment_path) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        migrated, report = migrate_config_data(data)
        config = _config_to_dataclass(migrated)
        config.migration_changed = report.changed
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
        return run_path

    @staticmethod
    def init_run(run_path: str | Path, config_data: dict, devices: list, sequence: list) -> Path:
        run_path = Path(run_path)
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
        src.rename(dst)
        meta_file = ExperimentManager._run_metadata_path(dst)
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        data, _ = migrate_run_data(data)
        if isinstance(data.get("run"), dict) and "name" in data["run"]:
            data["run"]["name"] = new_name
        elif "name" in data:
            data["name"] = new_name
        _atomic_write_json(meta_file, data)
        return dst

    @staticmethod
    def delete_run(run_path: str | Path):
        path = Path(run_path)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
