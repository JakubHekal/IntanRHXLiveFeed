from rhx_realtime_feed.device import _PLUGIN_REGISTRY, DeviceOperation, ParamDef

_DEVICE_CLASSES = dict(_PLUGIN_REGISTRY)

_SYSTEM_OPERATIONS = [
    DeviceOperation("wait_input", "Wait for User Input", instantaneous=True, default_duration=0, color="#FFD700", params=[
        ParamDef("message", "Message", "str", default="Click OK to continue"),
    ]),
    DeviceOperation("log_event", "Log Event", instantaneous=True, default_duration=0, color="#FFA500"),
    DeviceOperation("start_recording", "Start Recording", instantaneous=True, default_duration=0, color="#00CC66"),
    DeviceOperation("stop_recording", "Stop Recording", instantaneous=True, default_duration=0, color="#CC3333"),
    DeviceOperation("pause", "Pause", default_duration=1.0, color="#FF8C00"),
]
