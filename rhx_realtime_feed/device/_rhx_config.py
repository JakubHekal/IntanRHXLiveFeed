"""
rhx_realtime_feed.device._rhx_config

Low-level TCP configuration interface for the Intan RHX system.

This class wraps `set` and `get` TCP commands to configure:
- Analog channel streaming (wide, highpass, lowpass, spike)
- Sample rate and block size
- Digital inputs and trigger behavior
- Disk recording parameters and filename handling
- Filter settings (DSP, notch, bandwidth)

This module is inherited by `IntanRHXDevice`, which handles data streaming.
"""

import time
from collections.abc import Iterable


class GetSampleRateFailure(Exception):
    """Raised when unable to get sample rate from RHX server."""
    pass


class RHXConfig:
    """
    Stream and record EMG data from the Intan RHX system.

    Inherits:
        RHXConfig: Command/control interface for RHX configuration

    Responsibilities:
        - Establish TCP connections to RHX server
        - Parse incoming EMG waveform blocks
        - Buffer, store, or visualize real-time EMG data
        - Record sessions of configurable duration

    Attributes:
        host (str): IP address of the RHX system.
        command_port (int): Port for command/control TCP socket.
        data_port (int): Port for binary waveform data.
        num_channels (int): Number of EMG channels enabled.
        sample_rate (float): Sampling rate in Hz.
        verbose (bool): Debug logging toggle.
    """
    def __init__(self, command_socket, send_delay=0.05, verbose=False):
        self.command_socket = command_socket
        self.send_delay = send_delay
        self.verbose = verbose

    def set_parameter(self, param, value):
        """
        Set a parameter on the RHX system.

        Parameters:
            param (str): Parameter to set.
            value (str): Value to set the parameter to.
            send_delay (float): Delay after sending the command.

        Raises:
            ValueError: If the parameter is not recognized.
        """
        self.command_socket.sendall(f"set {param} {value}\n".encode())
        time.sleep(self.send_delay)

    def get_parameter(self, param):
        """
        Get a parameter from the RHX system.

        Parameters:
            param (str): Parameter to get.

        Returns:
            str: Value of the parameter.
        """
        self.command_socket.sendall(f"get {param}\n".encode())
        time.sleep(self.send_delay)
        return self.command_socket.recv(1024).decode()

    def enable_wide_channel(self, channels, port: str = 'a', status: bool = True):
        """
        Enable or disable wide channel data output.

        Parameters:
            channels (int, range, or iterable): Channel numbers to enable/disable.
            port (str): Port letter ('a', 'b', 'c', or 'd').
            status (bool): True to enable, False to disable.

        Raises:
            TypeError: If channels is not an int, range, or iterable.
        """
        if isinstance(channels, int):
            channels = [channels]
        elif not isinstance(channels, Iterable):
            raise TypeError("Channels must be an int, range, or iterable list.")

        for ch in channels:
            name = f"{port}-{ch:03d}"
            self.set_parameter(f"{name}.tcpdataoutputenabled", 'true' if status else 'false')

    def clear_all_data_outputs(self):
        """Clear all data output settings."""
        self.command_socket.sendall(b"execute clearalldataoutputs\n")
        time.sleep(self.send_delay)

    def get_run_mode(self):
        """
        Get the current run mode of the RHX system.

        Returns:
            str: Current run mode, either "run" or "stop".
        """
        response = self.get_parameter("runmode")
        return response.strip().split()[-1]

    def set_run_mode(self, mode):
        """
        Set the run mode of the RHX system.

        Parameters:
            mode (str): Run mode to set, either "run" or "stop".

        Raises:
            AssertionError: If the mode is not "run" or "stop".
        """
        assert mode in ["run", "stop"], "Mode must be 'run' or 'stop'"
        self.set_parameter("runmode", mode)

    def get_sample_rate(self):
        """
        Get the sample rate of the RHX system.

        Returns:
            float: Sample rate in Hz.
        Raises:
            GetSampleRateFailure: If unable to get sample rate.
        """
        resp = self.get_parameter("sampleratehertz")
        expected_return_string = "Return: SampleRateHertz "
        if resp.find(expected_return_string) == -1:
            raise GetSampleRateFailure('Unable to get sample rate from server.')
        if expected_return_string in resp:
            sample_rate = float(resp[len(expected_return_string):])
        else:
            raise ValueError(f"Unable to get sample rate from server: {resp}")
        return sample_rate

    def set_blocks_per_write(self, num_blocks):
        self.set_parameter("TCPNumberDataBlocksPerWrite", num_blocks)