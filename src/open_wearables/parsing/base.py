from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .audio import MicPacket, mic_samples_to_stereo


class PayloadParser:
    """Abstract base class for payload parsers."""

    expected_size: int

    def parse(self, data: bytes, **kwargs) -> List[dict]:
        raise NotImplementedError

    def should_build_df(self) -> bool:
        return True


@dataclass
class ParseResult:
    """Result of parsing a stream."""

    sensor_dfs: Dict[int, pd.DataFrame]
    mic_samples: List[int]
    mic_packets: List[MicPacket] = field(default_factory=list)
    audio_stereo: Optional[np.ndarray] = None

    @staticmethod
    def mic_samples_to_stereo(mic_samples: List[int]) -> Optional[np.ndarray]:
        return mic_samples_to_stereo(mic_samples)
