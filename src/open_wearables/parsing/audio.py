from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np


class MicPacket(TypedDict):
    timestamp: float
    samples: tuple[int, ...]


def interleaved_mic_to_stereo(
    samples: Union[np.ndarray, List[int], tuple[int, ...]],
) -> np.ndarray:
    """Convert interleaved [outer, inner, ...] int16 samples to [inner, outer] frames."""
    interleaved = np.asarray(samples, dtype=np.int16)
    if interleaved.size < 2:
        return np.empty((0, 2), dtype=np.int16)

    frame_count = interleaved.size // 2
    interleaved = interleaved[: frame_count * 2]
    return np.column_stack((interleaved[1::2], interleaved[0::2]))


def mic_packet_to_stereo_frames(
    packet: MicPacket,
    sampling_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return timestamps and stereo frames for a parsed microphone packet."""
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be > 0, got {sampling_rate}")

    stereo = interleaved_mic_to_stereo(packet["samples"])
    if stereo.size == 0:
        return np.empty((0,), dtype=np.float64), stereo

    timestamps = float(packet["timestamp"]) + (
        np.arange(stereo.shape[0], dtype=np.float64) / sampling_rate
    )
    return timestamps, stereo


def mic_samples_to_stereo(mic_samples: List[int]) -> Optional[np.ndarray]:
    if not mic_samples:
        return None
    stereo = interleaved_mic_to_stereo(mic_samples)
    if stereo.size == 0:
        return None
    return stereo
