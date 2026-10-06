from copy import deepcopy
from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

CONFIG_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 2
SYSTEM_DEVICE_ID = "system"
SYSTEM_DEVICE_TYPE = "__system__"
_SYSTEM_DEVICE_NAMES = frozenset(("__System__", "System actions", SYSTEM_DEVICE_TYPE))
_SYSTEM_OPERATION_IDS = frozenset(("wait_input", "log_event", "pause", "start_recording", "stop_recording"))


class MigrationError(ValueError):
    pass


class UnsupportedSchemaError(MigrationError):
    pass


@dataclass(frozen=True)
class MigrationReport:
    changed: bool
    from_version: int
    to_version: int
    warnings: tuple[str, ...] = ()


def new_device_id() -> str:
    return str(uuid4())


def migrate_device_config(device_class, config, config_version):
    target_version = getattr(device_class, "config_version", 1)
    if isinstance(target_version, bool) or not isinstance(target_version, int) or target_version < 0:
        raise MigrationError(f"Invalid config_version for {device_class.__name__}")
    if (
        isinstance(config_version, bool)
        or not isinstance(config_version, int)
        or config_version < 0
    ):
        raise MigrationError(f"Invalid device config_version: {config_version!r}")
    current = deepcopy(config if config is not None else {})
    if not isinstance(current, dict):
        raise MigrationError("device config must be an object")
    if config_version > target_version:
        raise UnsupportedSchemaError(
            f"Device config version {config_version} is newer than supported version {target_version}"
        )
    if config_version == target_version:
        return current, target_version, False
    try:
        migrated = device_class.migrate_config(current, config_version)
    except Exception as exc:
        raise MigrationError(f"Could not migrate device config: {exc}") from exc
    if not isinstance(migrated, dict):
        raise MigrationError("Device config migration must return an object")
    return migrated, target_version, True


def is_system_device_name(name) -> bool:
    return name in _SYSTEM_DEVICE_NAMES


def _read_version(data, target, kind):
    value = data.get("schema_version", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MigrationError(f"Invalid {kind} schema_version: {value!r}")
    if value > target:
        raise UnsupportedSchemaError(
            f"Unsupported {kind} schema version {value}; current version is {target}"
        )
    return value


def _run_chain(data, target, migrations, kind):
    if not isinstance(data, dict):
        raise MigrationError(f"{kind.capitalize()} data must be an object")
    result = deepcopy(data)
    from_version = _read_version(result, target, kind)
    version = from_version
    warnings = []
    while version < target:
        result, step_warnings = migrations[version](result)
        warnings.extend(step_warnings)
        next_version = _read_version(result, target, kind)
        if next_version != version + 1:
            raise MigrationError(
                f"Invalid {kind} migration step {version} -> {next_version}"
            )
        version = next_version
    validated_from = deepcopy(result)
    _validate_current(result, kind)
    return result, MigrationReport(
        changed=(
            from_version != target
            or bool(warnings)
            or result != validated_from
        ),
        from_version=from_version,
        to_version=target,
        warnings=tuple(warnings),
    )


def _validate_device_id(value, kind, index):
    if not isinstance(value, str) or not value:
        raise MigrationError(f"{kind} devices[{index}].device_id must be a UUID")
    if value == SYSTEM_DEVICE_ID:
        raise MigrationError(f"{kind} devices[{index}].device_id is reserved")
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise MigrationError(
            f"{kind} devices[{index}].device_id must be a UUID: {value!r}"
        ) from exc
    if value.lower() != str(parsed):
        raise MigrationError(
            f"{kind} devices[{index}].device_id must be canonical: {value!r}"
        )
    return value.casefold()


def _migrated_device_id(kind, index, name, device_type):
    seed = f"leech:{kind}:{index}:{device_type}:{name}"
    return str(uuid5(NAMESPACE_URL, seed))


def _device_action(step):
    return step.get("operation_id") or step.get("action", "")


def _is_legacy_system_step(step):
    if step.get("device_id") == SYSTEM_DEVICE_ID:
        return True
    name = step.get("device_name", "")
    action = _device_action(step)
    return action in _SYSTEM_OPERATION_IDS or (
        not step.get("device_id") and is_system_device_name(name)
    )


def _normalize_system_names(data):
    result = deepcopy(data)
    sequence = result.get("sequence", [])
    if isinstance(sequence, list):
        for step in sequence:
            if isinstance(step, dict) and _is_legacy_system_step(step):
                step["device_id"] = SYSTEM_DEVICE_ID
                step["device_name"] = ""
    return result, []


def _validate_current(data, kind):
    if kind == "run" and "devices" not in data and "sequence" not in data:
        return
    records = data.get("devices")
    if not isinstance(records, list):
        raise MigrationError(f"{kind} devices must be a list")
    known_ids = set()
    canonical_ids = {}
    names = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise MigrationError(f"{kind} devices[{index}] must be an object")
        device_id = _validate_device_id(record.get("device_id"), kind, index)
        if device_id in known_ids:
            raise MigrationError(f"{kind} has duplicate device_id {record['device_id']!r}")
        known_ids.add(device_id)
        canonical_ids[device_id] = record["device_id"]
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise MigrationError(f"{kind} devices[{index}].name must be non-empty text")
        if name in names:
            raise MigrationError(f"{kind} has duplicate device name {name!r}")
        names.add(name)
        device_type = record.get("device_type")
        if not isinstance(device_type, str) or not device_type:
            raise MigrationError(f"{kind} devices[{index}].device_type must be non-empty text")
        if device_type == SYSTEM_DEVICE_TYPE:
            raise MigrationError(f"{kind} devices[{index}].device_type is reserved")
        if not isinstance(record.get("config"), dict):
            raise MigrationError(f"{kind} devices[{index}].config must be an object")
        version = record.get("config_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise MigrationError(f"{kind} devices[{index}].config_version is invalid")
        if kind == "run":
            blocks = record.get("blocks")
            if not isinstance(blocks, list):
                raise MigrationError(f"run devices[{index}].blocks must be a list")
            for block_index, block in enumerate(blocks):
                if not isinstance(block, dict):
                    raise MigrationError(
                        f"run devices[{index}].blocks[{block_index}] must be an object"
                    )
                if not isinstance(block.get("operation_id"), str) or not block.get("operation_id"):
                    raise MigrationError(
                        f"run devices[{index}].blocks[{block_index}].operation_id is required"
                    )
    sequence = data.get("sequence")
    if not isinstance(sequence, list):
        raise MigrationError(f"{kind} sequence must be a list")
    for index, step in enumerate(sequence):
        if not isinstance(step, dict):
            raise MigrationError(f"{kind} sequence[{index}] must be an object")
        device_id = step.get("device_id")
        if device_id == SYSTEM_DEVICE_ID:
            pass
        elif isinstance(device_id, str) and device_id.casefold() in known_ids:
            step["device_id"] = canonical_ids[device_id.casefold()]
        else:
            raise MigrationError(
                f"{kind} sequence[{index}] references unknown device_id {device_id!r}"
            )
        if not isinstance(step.get("device_name"), str):
            raise MigrationError(f"{kind} sequence[{index}].device_name must be text")
        if not isinstance(step.get("action"), str) or not step.get("action"):
            raise MigrationError(f"{kind} sequence[{index}].action is required")
        if not isinstance(step.get("operation_id"), str) or not step.get("operation_id"):
            raise MigrationError(f"{kind} sequence[{index}].operation_id is required")
        if not isinstance(step.get("parameters"), dict):
            raise MigrationError(f"{kind} sequence[{index}].parameters must be an object")
    if kind == "config":
        execution_control = data.get("execution_control")
        if not isinstance(execution_control, dict):
            raise MigrationError("config execution_control must be an object")
        required = execution_control.get("required_devices")
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise MigrationError("config execution_control.required_devices must be a list of text")


def _config_0_to_1(data):
    result, warnings = _normalize_system_names(data)
    result["schema_version"] = 1
    return result, warnings


def _config_1_to_2(data):
    result = deepcopy(data)
    warnings = []
    records = result.get("devices", [])
    if records is None:
        records = []
    if not isinstance(records, list):
        raise MigrationError("devices must be a list")

    execution_control = result.get("execution_control", {})
    if not isinstance(execution_control, dict):
        execution_control = {}
        warnings.append("execution_control reset to an object")
    required_devices = execution_control.get("required_devices", [])
    if not isinstance(required_devices, list):
        warnings.append("execution_control.required_devices was not a list")
        required_devices = []

    result["execution_control"] = execution_control
    for device_type in required_devices:
        if not isinstance(device_type, str) or not device_type:
            warnings.append("Ignored invalid required device entry")
            continue
        if not any(
            isinstance(device, dict) and device.get("device_type") == device_type
            for device in records
        ):
            records.append({
                "name": device_type,
                "device_type": device_type,
                "config": {},
                "config_version": 1,
            })

    sequence = result.get("sequence", [])
    if sequence is None:
        sequence = []
    if not isinstance(sequence, list):
        raise MigrationError("sequence must be a list")

    name_map = {}
    id_to_name = {}
    seen_ids = set()
    seen_names = set()
    migrated_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise MigrationError(f"devices[{index}] must be an object")
        migrated = deepcopy(record)
        name = migrated.get("name", "")
        if not isinstance(name, str):
            name = str(name)
            warnings.append(f"devices[{index}].name converted to text")
        if not name:
            name = f"Device {index + 1}"
            warnings.append(f"devices[{index}].name set to {name!r}")
        if name in seen_names:
            raise MigrationError(f"Duplicate device name cannot be migrated safely: {name!r}")
        seen_names.add(name)
        device_type = migrated.get("device_type", "unknown")
        if not isinstance(device_type, str) or not device_type:
            device_type = "unknown"
            warnings.append(f"devices[{index}].device_type set to unknown")
        if device_type == SYSTEM_DEVICE_TYPE:
            continue
        migrated["name"] = name
        migrated["device_type"] = device_type

        device_id = migrated.get("device_id") or migrated.get("id")
        if device_id is None:
            device_id = _migrated_device_id("config", index, name, device_type)
        normalized_id = _validate_device_id(device_id, "config", index)
        if normalized_id in seen_ids:
            raise MigrationError(f"Duplicate device_id cannot be migrated safely: {device_id!r}")
        seen_ids.add(normalized_id)
        migrated["device_id"] = device_id

        config = migrated.get("config")
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise MigrationError(f"devices[{index}].config must be an object")
        migrated["config"] = deepcopy(config)

        config_version = migrated.get("config_version", 1)
        if isinstance(config_version, bool) or not isinstance(config_version, int) or config_version < 0:
            config_version = 1
            warnings.append(f"devices[{index}].config_version reset to 1")
        migrated["config_version"] = config_version
        migrated_records.append(migrated)
        name_map[name] = device_id
        id_to_name[device_id.casefold()] = name

    known_ids = set(id_to_name)
    for index, step in enumerate(sequence):
        if not isinstance(step, dict):
            raise MigrationError(f"sequence[{index}] must be an object")
        name = step.get("device_name", "")
        if not isinstance(name, str):
            name = str(name)
            step["device_name"] = name
            warnings.append(f"sequence[{index}].device_name converted to text")
        existing_id = step.get("device_id")
        if existing_id == SYSTEM_DEVICE_ID or _is_legacy_system_step(step):
            step["device_id"] = SYSTEM_DEVICE_ID
            step["device_name"] = ""
        elif existing_id:
            try:
                normalized_id = _validate_device_id(existing_id, "config", index)
            except MigrationError:
                raise
            if normalized_id not in known_ids:
                raise MigrationError(
                    f"sequence[{index}] references unknown device_id {existing_id!r}"
                )
            step["device_id"] = existing_id
            step["device_name"] = id_to_name[normalized_id]
        elif name in name_map:
            step["device_id"] = name_map[name]
        else:
            raise MigrationError(
                f"sequence[{index}] references unknown device name {name!r}"
            )
        if not isinstance(step.get("parameters"), dict):
            raise MigrationError(f"sequence[{index}].parameters must be an object")
        if not isinstance(step.get("action"), str) or not step.get("action"):
            raise MigrationError(f"sequence[{index}].action is required")
        if not step.get("operation_id"):
            step["operation_id"] = step["action"]

    result["devices"] = migrated_records
    result["sequence"] = sequence
    result["execution_control"].setdefault("required_devices", [])
    result["schema_version"] = 2
    return result, warnings


def migrate_config_data(data):
    return _run_chain(
        data,
        CONFIG_SCHEMA_VERSION,
        {0: _config_0_to_1, 1: _config_1_to_2},
        "config",
    )


def _run_0_to_1(data):
    result, warnings = _normalize_system_names(data)
    result["schema_version"] = 1
    return result, warnings


def _run_1_to_2(data):
    result = deepcopy(data)
    warnings = []
    if "devices" not in result and "sequence" not in result:
        result["schema_version"] = 2
        return result, warnings

    records = result.get("devices", [])
    if records is None:
        records = []
    if not isinstance(records, list):
        raise MigrationError("run devices must be a list")
    name_map = {}
    id_to_name = {}
    seen_ids = set()
    seen_names = set()
    migrated_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise MigrationError(f"run devices[{index}] must be an object")
        migrated = deepcopy(record)
        name = migrated.get("name", "")
        if not isinstance(name, str):
            name = str(name)
        if not name:
            name = f"Device {index + 1}"
            warnings.append(f"run devices[{index}].name set to {name!r}")
        if name in seen_names:
            raise MigrationError(f"Duplicate run device name cannot be migrated safely: {name!r}")
        seen_names.add(name)
        device_type = migrated.get("device_type", "unknown")
        if not isinstance(device_type, str) or not device_type:
            device_type = "unknown"
        if device_type == SYSTEM_DEVICE_TYPE:
            continue
        device_id = migrated.get("device_id") or migrated.get("id")
        if device_id is None:
            device_id = _migrated_device_id("run", index, name, device_type)
        normalized_id = _validate_device_id(device_id, "run", index)
        if normalized_id in seen_ids:
            raise MigrationError(f"Duplicate run device_id cannot be migrated safely: {device_id!r}")
        seen_ids.add(normalized_id)
        migrated["name"] = name
        migrated["device_type"] = device_type
        migrated["device_id"] = device_id
        config = migrated.get("config", {})
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise MigrationError(f"run devices[{index}].config must be an object")
        migrated["config"] = deepcopy(config)
        version = migrated.get("config_version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            version = 1
        migrated["config_version"] = version
        blocks = migrated.get("blocks", [])
        if blocks is None:
            blocks = []
        if not isinstance(blocks, list):
            raise MigrationError(f"run devices[{index}].blocks must be a list")
        migrated["blocks"] = blocks
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                raise MigrationError(
                    f"run devices[{index}].blocks[{block_index}] must be an object"
                )
            if not block.get("operation_id"):
                action = block.get("action", "")
                if not isinstance(action, str) or not action:
                    raise MigrationError(
                        f"run devices[{index}].blocks[{block_index}].action is required"
                    )
                block["operation_id"] = action
        migrated_records.append(migrated)
        name_map[name] = device_id
        id_to_name[device_id.casefold()] = name

    known_ids = set(id_to_name)
    sequence = result.get("sequence", [])
    if sequence is None:
        sequence = []
    if not isinstance(sequence, list):
        raise MigrationError("run sequence must be a list")
    for index, step in enumerate(sequence):
        if not isinstance(step, dict):
            raise MigrationError(f"run sequence[{index}] must be an object")
        name = step.get("device_name", "")
        if not isinstance(name, str):
            name = str(name)
            step["device_name"] = name
        existing_id = step.get("device_id")
        if existing_id == SYSTEM_DEVICE_ID or _is_legacy_system_step(step):
            step["device_id"] = SYSTEM_DEVICE_ID
            step["device_name"] = ""
        elif existing_id:
            normalized_id = _validate_device_id(existing_id, "run", index)
            if normalized_id not in known_ids:
                raise MigrationError(
                    f"run sequence[{index}] references unknown device_id {existing_id!r}"
                )
            step["device_id"] = existing_id
            step["device_name"] = id_to_name[normalized_id]
        elif name in name_map:
            step["device_id"] = name_map[name]
        else:
            raise MigrationError(
                f"run sequence[{index}] references unknown device name {name!r}"
            )
        if not isinstance(step.get("parameters"), dict):
            raise MigrationError(f"run sequence[{index}].parameters must be an object")
        if not isinstance(step.get("action"), str) or not step.get("action"):
            raise MigrationError(f"run sequence[{index}].action is required")
        if not step.get("operation_id"):
            step["operation_id"] = step["action"]

    result["devices"] = migrated_records
    result["sequence"] = sequence
    result["schema_version"] = 2
    return result, warnings


def migrate_run_data(data):
    return _run_chain(
        data,
        RUN_SCHEMA_VERSION,
        {0: _run_0_to_1, 1: _run_1_to_2},
        "run",
    )
