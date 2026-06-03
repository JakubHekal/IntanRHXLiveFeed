"""
Finite State Machine for managing app and device states.

States:
  - IDLE: App initialized, no device connection
  - CONNECTING: Connection attempt in progress
  - CONNECTED: Device connected, not streaming
  - STREAMING: Device connected and actively receiving data
  - WAITING_FOR_DATA: Streaming but no data arriving yet
  - PAUSED: User paused data acquisition
  - CONNECTION_LOST: Device disconnected unexpectedly
  - DISCONNECTING: Cleanup in progress after disconnect
"""

from enum import Enum
from typing import Optional, Dict, Callable, Any
from PyQt5 import QtCore


class AppState(Enum):
    """All possible application states."""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    WAITING_FOR_DATA = "waiting_for_data"
    PAUSED = "paused"
    CONNECTION_LOST = "connection_lost"
    DISCONNECTING = "disconnecting"


# Transition table: {from_state: {trigger: to_state}}
_TRANSITIONS: Dict[AppState, Dict[str, AppState]] = {
    AppState.IDLE: {
        "request_connect": AppState.CONNECTING,
    },
    AppState.CONNECTING: {
        "connection_succeeded": AppState.CONNECTED,
        "connection_failed": AppState.IDLE,
    },
    AppState.CONNECTED: {
        "request_stream": AppState.STREAMING,
        "request_disconnect": AppState.DISCONNECTING,
    },
    AppState.STREAMING: {
        "no_data_available": AppState.WAITING_FOR_DATA,
        "user_pause": AppState.PAUSED,
        "request_disconnect": AppState.DISCONNECTING,
        "device_disconnected": AppState.CONNECTION_LOST,
    },
    AppState.WAITING_FOR_DATA: {
        "data_arrived": AppState.STREAMING,
        "user_pause": AppState.PAUSED,
        "request_disconnect": AppState.DISCONNECTING,
        "device_disconnected": AppState.CONNECTION_LOST,
    },
    AppState.PAUSED: {
        "user_resume": AppState.STREAMING,
        "request_disconnect": AppState.DISCONNECTING,
        "device_disconnected": AppState.CONNECTION_LOST,
    },
    AppState.CONNECTION_LOST: {
        "request_disconnect": AppState.DISCONNECTING,
    },
    AppState.DISCONNECTING: {
        "disconnect_complete": AppState.IDLE,
    },
}


class AppStateMachine(QtCore.QObject):
    """
    Finite State Machine for the application lifecycle and device communication.

    Emits PyQt signals on state transitions to notify UI components.
    Uses a dict-based transition table for formal FSM with guarded transitions.
    """

    # PyQt Signals
    state_changed = QtCore.pyqtSignal(AppState)
    state_changed_str = QtCore.pyqtSignal(str)  # For backwards compatibility
    state_entering = QtCore.pyqtSignal(AppState)
    state_exiting = QtCore.pyqtSignal(AppState)

    def __init__(self):
        super().__init__()
        self._state: AppState = AppState.IDLE
        self._transition_history = []

    @property
    def state(self) -> str:
        """Get the current state as a string (compatibility with old .state usage)."""
        return self._state.value

    # ── Query Methods ───────────────────────────────────────────────────────────

    def get_current_state(self) -> AppState:
        """Get the current state as an AppState enum."""
        return self._state

    def get_current_state_str(self) -> str:
        """Get the current state as a string."""
        return self._state.value

    def can_trigger(self, trigger: str) -> bool:
        """
        Check whether the given trigger is valid from the current state.
        Returns True if a transition exists for this trigger.
        """
        transitions = _TRANSITIONS.get(self._state, {})
        return trigger in transitions

    def can_transition_to(self, target_state: AppState) -> bool:
        """
        Check if a transition to the target state is valid from current state.
        Returns True if valid, False otherwise.
        """
        transitions = _TRANSITIONS.get(self._state, {})
        return target_state in transitions.values()

    def process_trigger(self, trigger: str) -> bool:
        """
        Execute a state transition if the trigger is valid from the current state.
        Returns True if transition was executed, False otherwise.
        """
        transitions = _TRANSITIONS.get(self._state, {})
        target_state = transitions.get(trigger)
        if target_state is None:
            return False

        old_state = self._state
        self.state_exiting.emit(old_state)

        self._state = target_state

        self.state_changed.emit(self._state)
        self.state_changed_str.emit(self._state.value)
        self._log_transition(self._state)

        return True

    def is_connected(self) -> bool:
        """Check if device is connected (any state except IDLE and CONNECTING)."""
        return self._state in [
            AppState.CONNECTED,
            AppState.STREAMING,
            AppState.WAITING_FOR_DATA,
            AppState.PAUSED,
            AppState.CONNECTION_LOST,
        ]

    def is_streaming(self) -> bool:
        """Check if device is actively streaming or waiting for data."""
        return self._state in [
            AppState.STREAMING,
            AppState.WAITING_FOR_DATA,
        ]

    def is_streaming_active(self) -> bool:
        """Check if device is actively streaming (not paused, not waiting)."""
        return self._state == AppState.STREAMING

    # ── History ─────────────────────────────────────────────────────────────────

    def _log_transition(self, new_state: AppState):
        """Log state transitions for debugging."""
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        self._transition_history.append({
            'timestamp': timestamp,
            'new_state': new_state.value
        })
        if len(self._transition_history) > 100:
            self._transition_history.pop(0)

    def get_transition_history(self):
        """Get the list of recent state transitions for debugging."""
        return list(self._transition_history)

    def reset_history(self):
        """Clear the transition history."""
        self._transition_history.clear()
