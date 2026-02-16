import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from open_earable_python import parser
import open_earable_python.scheme as scheme
from IPython.display import Audio, display
from scipy.io.wavfile import write

LABELS: Dict[str, List[str]] = {
    "imu": [
        "acc.x", "acc.y", "acc.z",
        "gyro.x", "gyro.y", "gyro.z",
        "mag.x", "mag.y", "mag.z",
    ],
    "barometer": ["barometer.temperature", "barometer.pressure"],
    "ppg": ["ppg.red", "ppg.ir", "ppg.green", "ppg.ambient"],
    "bone_acc": ["bone_acc.x", "bone_acc.y", "bone_acc.z"],
    "optical_temp": ["optical_temp"],
    "microphone": ["mic.inner", "mic.outer"],
}

COLORS: Dict[str, List[str]] = {
    "ppg": ["red", "darkred", "green", "gray"],
}


class _SensorAccessor:
    """Convenience wrapper around a pandas DataFrame to provide grouped access
    to sensor channels.

    For IMU data with columns:
    - acc.x, acc.y, acc.z
    - gyro.x, gyro.y, gyro.z
    - mag.x, mag.y, mag.z

    Access patterns:

    - accessor["imu"] or accessor.imu -> sub-DataFrame
    - accessor.acc["x"] or accessor.acc.x -> Series
    """

    def __init__(self, df: pd.DataFrame, labels: Sequence[str]):
        self._df = df
        self._data: Dict[str, pd.DataFrame] = {}

        groups: Dict[str, List[str]] = defaultdict(list)

        for label in labels:
            parts = label.split(".")
            if len(parts) == 2:
                group, _field = parts
                if label in df:
                    groups[group].append(label)
            elif label in df:
                # Single-level column names are exposed directly.
                self._data[label] = df[label]

        for group, columns in groups.items():
            short_names = [label.split(".")[1] for label in columns]
            subdf = df[columns].copy()
            subdf.columns = short_names
            self._data[group] = subdf

        # Preserve the original column names to avoid collisions between groups
        # with identical short names (e.g., acc.x vs gyro.x).
        self._full_df = df.copy()

    @property
    def df(self) -> pd.DataFrame:
        """Return the underlying full DataFrame view."""
        return self._full_df

    def to_dataframe(self) -> pd.DataFrame:
        """Alias for :attr:`df` for convenience."""
        return self._full_df

    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]

        if key in self._full_df.columns:
            return self._full_df[key]

        raise KeyError(f"{key!r} not found in available sensor groups or channels")

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]

        if hasattr(self._full_df, name):
            return getattr(self._full_df, name)

        raise AttributeError(f"'SensorAccessor' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return repr(self._full_df)


class SensorDataset:
    """High-level representation of an OpenEarable sensor recording file."""

    SENSOR_SID: Dict[str, int] = {
        "imu": 0,
        "barometer": 1,
        "microphone": 2,
        "ppg": 4,
        "optical_temp": 6,
        "bone_acc": 7,
    }

    SID_NAMES: Dict[int, str] = {
        0: "imu",
        1: "barometer",
        2: "microphone",
        4: "ppg",
        6: "optical_temp",
        7: "bone_acc",
    }

    sensor_formats: Dict[int, str] = {
        SENSOR_SID["imu"]: "<9f",
        SENSOR_SID["barometer"]: "<2f",
        SENSOR_SID["ppg"]: "<4I",
        SENSOR_SID["optical_temp"]: "<f",
        SENSOR_SID["bone_acc"]: "<3h",
    }

    def __init__(self, filename: str, verbose: bool = False):
        self.filename = filename
        self.verbose = verbose
        self.parse_result: parser.ParseResult = parser.ParseResult(
            sensor_dfs={},
            mic_samples=[],
        )
        # Per-SID dataframes built in _build_accessors
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
                _SensorAccessor(pd.DataFrame(columns=labels), labels),
            )

        self.parser: parser.Parser = self._build_parser(verbose=verbose)

        self.parse()
        self._build_accessors()

    @classmethod
    def _build_parser(cls, verbose: bool = False) -> parser.Parser:
        sensor_schemes = scheme.build_default_sensor_schemes(cls.SENSOR_SID)
        dataset_parser = parser.Parser.from_sensor_schemes(
            sensor_schemes=sensor_schemes,
            verbose=verbose,
        )
        dataset_parser.parsers[cls.SENSOR_SID["microphone"]] = parser.MicPayloadParser(
            sample_count=48000,
            verbose=verbose,
        )
        return dataset_parser

    def parse(self) -> None:
        """Parse the binary recording file into structured sensor data."""
        with open(self.filename, "rb") as f:
            parse_result = self.parser.parse(f)
        self.parse_result = parse_result
    
    def _build_accessors(self) -> None:
        """Construct per-sensor accessors and per-SID DataFrames.

        Each sensor's data is stored in its own DataFrame in ``self.sensor_dfs``.
        The combined DataFrame over all sensors is built lazily in
        :meth:`get_dataframe`.
        """
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

            # Store per-SID dataframe
            self.sensor_dfs[sid] = df

            # Create/update SensorAccessor for this sensor name
            setattr(self, name, _SensorAccessor(df, labels))

        # Clear combined dataframe; it will be built lazily on demand
        self.df = pd.DataFrame()

    def list_sensors(self) -> List[str]:
        """Return a list of available sensor names in the dataset."""
        available_sensors = []
        for name, sid in self.SENSOR_SID.items():
            accessor = getattr(self, name, None)
            if isinstance(accessor, _SensorAccessor) and not accessor.df.empty:
                available_sensors.append(name)
        return available_sensors

    def get_sensor_dataframe(self, name: str) -> pd.DataFrame:
        """Return the DataFrame for a single sensor.

        Parameters
        ----------
        name:
            Sensor name, e.g. "imu", "barometer", "ppg", "bone_acc", "optical_temp".

        Returns
        -------
        pandas.DataFrame
            The time-indexed DataFrame for the requested sensor.
        """
        if name not in self.SENSOR_SID:
            raise KeyError(f"Unknown sensor name: {name!r}. "
                           f"Known sensors: {sorted(self.SENSOR_SID.keys())}")

        accessor = getattr(self, name, None)
        if isinstance(accessor, _SensorAccessor):
            return accessor.to_dataframe()

        # Fallback: should not normally happen, but return an empty DataFrame
        # instead of crashing.
        return pd.DataFrame()

    def get_dataframe(self) -> pd.DataFrame:
        """Return the combined, time-indexed DataFrame of all sensors.

        The merged DataFrame is built lazily from the per-SID DataFrames in
        :attr:`sensor_dfs` and cached in :attr:`df`.
        """
        # If we've already built a non-empty combined DataFrame, reuse it
        if not self.df.empty:
            return self.df

        # If per-SID dataframes are not available, nothing to merge
        if not getattr(self, "sensor_dfs", None):
            return self.df

        # Collect all non-empty per-SID dataframes
        dfs = [df for df in self.sensor_dfs.values() if not df.empty]
        if not dfs:
            return self.df

        # Build a common time index over all sensors
        common_index = pd.Index([])
        for df in dfs:
            common_index = common_index.union(df.index)
        common_index = common_index.sort_values()

        # Reindex each DataFrame to the common index and concatenate
        reindexed_dfs = [df.reindex(common_index) for df in dfs]
        self.df = pd.concat(reindexed_dfs, axis=1)

        return self.df

    def get_audio_dataframe(self, sampling_rate: int = 48000) -> pd.DataFrame:
        """Return microphone audio as a timestamp-indexed stereo DataFrame.

        The returned DataFrame has:
        - index: ``timestamp`` in seconds
        - columns: ``mic.inner`` and ``mic.outer`` (int16 PCM)
        """
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
            ts, stereo = parser.mic_packet_to_stereo_frames(
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
        except Exception as e:
            print(f"❌ Error saving audio to {path}: {e}")


def load_recordings(file_paths: Sequence[str]) -> List[SensorDataset]:
    return [SensorDataset(path) for path in file_paths if os.path.isfile(path)]
