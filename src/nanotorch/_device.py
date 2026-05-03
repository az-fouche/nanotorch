"""CUDA device handling."""

from typing import Literal

from nanotorch import _C

Device = _C.Device
DeviceLiteral = Literal["cpu", "cuda"]


def is_cuda_available() -> bool:
    """Detect if a CUDA device is ready to be used."""
    return _C.is_cuda_available()


def get_std_device(device: Device | DeviceLiteral) -> Device:
    """Converts any device repr to standard Device."""
    if isinstance(device, Device):
        return device
    match device:
        case "cpu":
            return Device.Cpu
        case "cuda":
            return Device.Cuda
        case _:
            raise ValueError(f"Unknown device {device}.")
