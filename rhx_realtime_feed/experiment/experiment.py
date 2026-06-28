import json
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentMetadata:
    experiment_name: str = ""
    version: str = "1.0.0"
    author: str = ""
    created_at: str = ""
    description: str = ""
    cloned_from: Optional[str] = None


@dataclass
class ExecutionControl:
    is_locked: bool = False
    required_devices: list[str] = field(default_factory=list)


@dataclass
class SequenceStep:
    step_id: int = 1
    action: str = ""
    parameters: dict = field(default_factory=dict)


@dataclass
class PostProcessingScript:
    script_id: int = 1
    name: str = ""
    environment: str = "Python"
    script_path: str = ""
    enabled_by_default: bool = True


@dataclass
class ExperimentConfig:
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    execution_control: ExecutionControl = field(default_factory=ExecutionControl)
    sequence: list[SequenceStep] = field(default_factory=list)
    post_processing: list[PostProcessingScript] = field(default_factory=list)


def _default_config(name: str, author: str = "", description: str = "") -> dict:
    return {
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
            "required_devices": ["Intan RHX"],
        },
        "sequence": [],
        "post_processing": [],
    }


def _config_to_dataclass(data: dict) -> ExperimentConfig:
    m = data.get("metadata", {})
    ec = data.get("execution_control", {})
    seq = data.get("sequence", [])
    pp = data.get("post_processing", [])
    return ExperimentConfig(
        metadata=ExperimentMetadata(
            experiment_name=m.get("experiment_name", ""),
            version=m.get("version", "1.0.0"),
            author=m.get("author", ""),
            created_at=m.get("created_at", ""),
            description=m.get("description", ""),
            cloned_from=m.get("cloned_from"),
        ),
        execution_control=ExecutionControl(
            is_locked=ec.get("is_locked", False),
            required_devices=ec.get("required_devices", []),
        ),
        sequence=[
            SequenceStep(
                step_id=s.get("step_id", i + 1),
                action=s.get("action", ""),
                parameters=s.get("parameters", {}),
            )
            for i, s in enumerate(seq)
        ],
        post_processing=[
            PostProcessingScript(
                script_id=p.get("script_id", i + 1),
                name=p.get("name", ""),
                environment=p.get("environment", "Python"),
                script_path=p.get("script_path", ""),
                enabled_by_default=p.get("enabled_by_default", True),
            )
            for i, p in enumerate(pp)
        ],
    )


def _config_to_dict(config: ExperimentConfig) -> dict:
    return {
        "metadata": {
            "experiment_name": config.metadata.experiment_name,
            "version": config.metadata.version,
            "author": config.metadata.author,
            "created_at": config.metadata.created_at,
            "description": config.metadata.description,
            "cloned_from": config.metadata.cloned_from,
        },
        "execution_control": {
            "is_locked": config.execution_control.is_locked,
            "required_devices": list(config.execution_control.required_devices),
        },
        "sequence": [
            {
                "step_id": s.step_id,
                "action": s.action,
                "parameters": dict(s.parameters),
            }
            for s in config.sequence
        ],
        "post_processing": [
            {
                "script_id": p.script_id,
                "name": p.name,
                "environment": p.environment,
                "script_path": p.script_path,
                "enabled_by_default": p.enabled_by_default,
            }
            for p in config.post_processing
        ],
    }


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
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        return experiment_path

    @staticmethod
    def load(experiment_path: str | Path) -> ExperimentConfig:
        config_path = Path(experiment_path) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _config_to_dataclass(data)

    @staticmethod
    def save(experiment_path: str | Path, config: ExperimentConfig):
        config_path = Path(experiment_path) / "config.json"
        data = _config_to_dict(config)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

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
