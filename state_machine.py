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
from transitions import Machine


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


class AppStateMachine(QtCore.QObject):
    """
    Finite State Machine for the application lifecycle and device communication.
    
    Emits PyQt signals on state transitions to notify UI components.
    Uses transitions library for formal FSM with guarded transitions.
    """
    
    # PyQt Signals
    state_changed = QtCore.pyqtSignal(AppState)
    state_changed_str = QtCore.pyqtSignal(str)  # For backwards compatibility
    state_entering = QtCore.pyqtSignal(AppState)
    state_exiting = QtCore.pyqtSignal(AppState)
    
    def __init__(self):
        super().__init__()
        self.current_state_enum: AppState = AppState.IDLE
        self._callbacks: Dict[str, list] = {}
        self._transition_history = []
        
        # Define all valid states (using enum values as state names for transitions lib)
        states = [s.value for s in AppState]
        
        # Initialize the state machine
        self.machine = Machine(
            model=self,
            states=states,
            initial=AppState.IDLE.value,
            auto_transitions=False,  # Only allow defined transitions
            ignore_invalid_triggers=True,  # Don't raise on invalid transitions
        )
        
        # Define all valid transitions
        self._define_transitions()
        
    def _define_transitions(self):
        """Define all valid state transitions with guards and callbacks."""
        
        # IDLE transitions
        self.machine.add_transition(
            'request_connect', AppState.IDLE.value, AppState.CONNECTING.value,
            before='_before_transition', after='_after_transition'
        )
        
        # CONNECTING transitions
        self.machine.add_transition(
            'connection_succeeded', AppState.CONNECTING.value, AppState.CONNECTED.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'connection_failed', AppState.CONNECTING.value, AppState.IDLE.value,
            before='_before_transition', after='_after_transition'
        )
        
        # CONNECTED transitions
        self.machine.add_transition(
            'request_stream', AppState.CONNECTED.value, AppState.STREAMING.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'request_disconnect', AppState.CONNECTED.value, AppState.DISCONNECTING.value,
            before='_before_transition', after='_after_transition'
        )
        
        # STREAMING transitions
        self.machine.add_transition(
            'no_data_available', AppState.STREAMING.value, AppState.WAITING_FOR_DATA.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'user_pause', AppState.STREAMING.value, AppState.PAUSED.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'request_disconnect', AppState.STREAMING.value, AppState.DISCONNECTING.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'device_disconnected', AppState.STREAMING.value, AppState.CONNECTION_LOST.value,
            before='_before_transition', after='_after_transition'
        )
        
        # WAITING_FOR_DATA transitions
        self.machine.add_transition(
            'data_arrived', AppState.WAITING_FOR_DATA.value, AppState.STREAMING.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'user_pause', AppState.WAITING_FOR_DATA.value, AppState.PAUSED.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'request_disconnect', AppState.WAITING_FOR_DATA.value, AppState.DISCONNECTING.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'device_disconnected', AppState.WAITING_FOR_DATA.value, AppState.CONNECTION_LOST.value,
            before='_before_transition', after='_after_transition'
        )
        
        # PAUSED transitions
        self.machine.add_transition(
            'user_resume', AppState.PAUSED.value, AppState.STREAMING.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'request_disconnect', AppState.PAUSED.value, AppState.DISCONNECTING.value,
            before='_before_transition', after='_after_transition'
        )
        self.machine.add_transition(
            'device_disconnected', AppState.PAUSED.value, AppState.CONNECTION_LOST.value,
            before='_before_transition', after='_after_transition'
        )
        
        # CONNECTION_LOST transitions
        self.machine.add_transition(
            'request_disconnect', AppState.CONNECTION_LOST.value, AppState.DISCONNECTING.value,
            before='_before_transition', after='_after_transition'
        )
        
        # DISCONNECTING transitions
        self.machine.add_transition(
            'disconnect_complete', AppState.DISCONNECTING.value, AppState.IDLE.value,
            before='_before_transition', after='_after_transition'
        )
    
    def _before_transition(self):
        """Called before any state transition."""
        old_state = AppState(self.state)
        self.state_exiting.emit(old_state)
    
    def _after_transition(self):
        """Called after any state transition."""
        new_state = AppState(self.state)
        self.current_state_enum = new_state
        self.state_changed.emit(new_state)
        self.state_changed_str.emit(new_state.value)
        self._log_transition(new_state)
    
    def _log_transition(self, new_state: AppState):
        """Log state transitions for debugging."""
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        self._transition_history.append({
            'timestamp': timestamp,
            'new_state': new_state.value
        })
        # Keep only last 100 transitions to avoid memory bloat
        if len(self._transition_history) > 100:
            self._transition_history.pop(0)
    
    def get_current_state(self) -> AppState:
        """Get the current state as an AppState enum."""
        return self.current_state_enum
    
    def get_current_state_str(self) -> str:
        """Get the current state as a string."""
        return self.current_state_enum.value
    
    def can_transition_to(self, target_state: AppState) -> bool:
        """
        Check if a transition to the target state is valid from current state.
        Returns True if valid, False otherwise.
        """
        current = AppState(self.state)
        
        # Define valid transitions as a state transition table
        valid_transitions = {
            AppState.IDLE: [AppState.CONNECTING],
            AppState.CONNECTING: [AppState.CONNECTED, AppState.IDLE],
            AppState.CONNECTED: [AppState.STREAMING, AppState.DISCONNECTING],
            AppState.STREAMING: [AppState.WAITING_FOR_DATA, AppState.PAUSED, AppState.DISCONNECTING, AppState.CONNECTION_LOST],
            AppState.WAITING_FOR_DATA: [AppState.STREAMING, AppState.PAUSED, AppState.DISCONNECTING, AppState.CONNECTION_LOST],
            AppState.PAUSED: [AppState.STREAMING, AppState.DISCONNECTING, AppState.CONNECTION_LOST],
            AppState.CONNECTION_LOST: [AppState.DISCONNECTING],
            AppState.DISCONNECTING: [AppState.IDLE],
        }
        
        return target_state in valid_transitions.get(current, [])
    
    def is_connected(self) -> bool:
        """Check if device is connected (any state except IDLE and CONNECTING)."""
        return self.current_state_enum in [
            AppState.CONNECTED,
            AppState.STREAMING,
            AppState.WAITING_FOR_DATA,
            AppState.PAUSED,
            AppState.CONNECTION_LOST,
        ]
    
    def is_streaming(self) -> bool:
        """Check if device is actively streaming or waiting for data."""
        return self.current_state_enum in [
            AppState.STREAMING,
            AppState.WAITING_FOR_DATA,
        ]
    
    def is_streaming_active(self) -> bool:
        """Check if device is actively streaming (not paused, not waiting)."""
        return self.current_state_enum == AppState.STREAMING
    
    def get_transition_history(self):
        """Get the list of recent state transitions for debugging."""
        return self._transition_history.copy()
    
    def reset_history(self):
        """Clear the transition history."""
        self._transition_history.clear()
