import os
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from IPython.display import Audio, display
from scipy.io.wavfile import write

from open_wearables.parsing import MicPayloadParser, ParseResult, Parser, mic_packet_to_stereo_frames
from open_wearables.schema import build_default_sensor_schemes

from .accessors import SensorAccessor
from .constants import COLORS, LABELS, SENSOR_FORMATS, SENSOR_SID, SID_NAMES


class SensorDataset:
    """High-level representation of an OpenEarable sensor recording file."""

    SENSOR_SID: Dict[str, int] = SENSOR_SID
    SID_NAMES: Dict[int, str] = SID_NAMES
    sensor_formats: Dict[int, str] = SENSOR_FORMATS

    def __init__(self, filename: str, verbose: bool = False):
        self.filename = filename
        self.verbose = verbose
        self.parse_result: ParseResult = ParseResult(sensor_dfs={}, mic_samples=[])
        self.sensor_dfs: Dict[int, pd.DataFrame] = {}
        self.audio_stereo: Optional[np.ndarray] = None
        self.audio_df: pd.DataFrame = pd.DataFrame()
        self._audio_df_sampling_rate: Optional[int] = None
        self.bone_sound: Optional[np.ndarray] = None
        self.df: pd.DataFrame = pd.DataFrame()

        for sensor_name, labels in LABELS.items():
            setattr(
                self,
                sensor_name,
                SensorAccessor(pd.DataFrame(columns=labels), labels),
            )

        self.parser: Parser = self._build_parser(verbose=verbose)

        self.parse()
        self._build_accessors()

    @classmethod
    def _build_parser(cls, verbose: bool = False) -> Parser:
        sensor_schemes = build_default_sensor_schemes(cls.SENSOR_SID)
        dataset_parser = Parser.from_sensor_schemes(
            sensor_schemes=sensor_schemes,
            verbose=verbose,
        )
        dataset_parser.parsers[cls.SENSOR_SID["microphone"]] = MicPayloadParser(
            sample_count=48000,
            verbose=verbose,
        )
        return dataset_parser

    def parse(self) -> None:
        with open(self.filename, "rb") as stream:
            self.parse_result = self.parser.parse(stream)

    def _build_accessors(self) -> None:
        self.audio_stereo = self.parse_result.audio_stereo
        self.audio_df = pd.DataFrame()
        self._audio_df_sampling_rate = None
        self.sensor_dfs = {}

        data_dict = self.parse_result.sensor_dfs
        for name, sid in self.SENSOR_SID.items():
            labels = LABELS.get(name, [])
            if name == "microphone":
                df = self.get_audio_dataframe()
            elif sid in data_dict and isinstance(data_dict[sid], pd.DataFrame):
                df = data_dict[sid]
                df = df[~df.index.duplicated(keep="first")]
            else:
                df = pd.DataFrame(columns=labels)

            self.sensor_dfs[sid] = df
            setattr(self, name, SensorAccessor(df, labels))

        self.df = pd.DataFrame()

    def list_sensors(self) -> List[str]:
        available_sensors = []
        for name in self.SENSOR_SID:
            accessor = getattr(self, name, None)
            if isinstance(accessor, SensorAccessor) and not accessor.df.empty:
                available_sensors.append(name)
        return available_sensors

    def get_sensor_dataframe(self, name: str) -> pd.DataFrame:
        if name not in self.SENSOR_SID:
            raise KeyError(
                f"Unknown sensor name: {name!r}. "
                f"Known sensors: {sorted(self.SENSOR_SID.keys())}"
            )

        accessor = getattr(self, name, None)
        if isinstance(accessor, SensorAccessor):
            return accessor.to_dataframe()

        return pd.DataFrame()

    def get_dataframe(self) -> pd.DataFrame:
        if not self.df.empty:
            return self.df

        if not getattr(self, "sensor_dfs", None):
            return self.df

        dfs = [df for df in self.sensor_dfs.values() if not df.empty]
        if not dfs:
            return self.df

        common_index = pd.Index([])
        for df in dfs:
            common_index = common_index.union(df.index)
        common_index = common_index.sort_values()

        reindexed_dfs = [df.reindex(common_index) for df in dfs]
        self.df = pd.concat(reindexed_dfs, axis=1)

        return self.df

    def get_audio_dataframe(self, sampling_rate: int = 48000) -> pd.DataFrame:
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be > 0, got {sampling_rate}")

        if self._audio_df_sampling_rate == sampling_rate:
            return self.audio_df

        mic_packets = getattr(self.parse_result, "mic_packets", [])
        if not mic_packets:
            self.audio_df = pd.DataFrame(columns=["mic.inner", "mic.outer"])
            self.audio_df.index.name = "timestamp"
            self._audio_df_sampling_rate = sampling_rate
            return self.audio_df

        timestamps: List[np.ndarray] = []
        stereo_frames: List[np.ndarray] = []

        for packet in mic_packets:
            ts, stereo = mic_packet_to_stereo_frames(
                packet=packet,
                sampling_rate=sampling_rate,
            )
            if stereo.size == 0:
                continue
            timestamps.append(ts)
            stereo_frames.append(stereo)

        if not timestamps:
            self.audio_df = pd.DataFrame(columns=["mic.inner", "mic.outer"])
            self.audio_df.index.name = "timestamp"
            self._audio_df_sampling_rate = sampling_rate
            return self.audio_df

        all_ts = np.concatenate(timestamps)
        all_stereo = np.vstack(stereo_frames)

        self.audio_df = pd.DataFrame(
            {
                "mic.inner": all_stereo[:, 0],
                "mic.outer": all_stereo[:, 1],
            },
            index=all_ts,
        )
        self.audio_df.index.name = "timestamp"
        self.audio_df = self.audio_df[~self.audio_df.index.duplicated(keep="first")]
        self._audio_df_sampling_rate = sampling_rate

        if sampling_rate == 48000:
            self.sensor_dfs[self.SENSOR_SID["microphone"]] = self.audio_df

        return self.audio_df

    def export_csv(self) -> None:
        base_filename, _ = os.path.splitext(self.filename)
        self.save_csv(base_filename + ".csv")

    def save_csv(self, path: str) -> None:
        if not self.df.empty:
            self.df.to_csv(path)

    def play_audio(self, sampling_rate: int = 48000) -> None:
        if self.audio_stereo is None:
            print("❌ No microphone data available.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            write(tmp.name, sampling_rate, self.audio_stereo)
            display(Audio(tmp.name))

    def save_audio(self, path: str, sampling_rate: int = 48000) -> None:
        if self.audio_stereo is None:
            print("❌ No microphone data available to save.")
            return
        try:
            write(path, sampling_rate, self.audio_stereo)
            print(f"✅ Audio saved successfully to {path}")
        except Exception as exc:
            print(f"❌ Error saving audio to {path}: {exc}")


def load_recordings(file_paths: Sequence[str]) -> List[SensorDataset]:
    return [SensorDataset(path) for path in file_paths if os.path.isfile(path)]


__all__ = [
    "COLORS",
    "LABELS",
    "SensorAccessor",
    "SensorDataset",
    "load_recordings",
]
