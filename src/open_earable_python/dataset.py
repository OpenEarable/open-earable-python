import os
import struct
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from open_earable_python import parser
import open_earable_python.scheme as scheme
from IPython.display import Audio, display
from scipy.io.wavfile import write
from scipy.signal import butter, filtfilt, istft, resample, stft
from sklearn.decomposition import PCA

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
}

COLORS: Dict[str, List[str]] = {
    "ppg": ["red", "darkred", "green", "gray"],
}


class SensorAccessor:
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
        self.parse_result: Dict[int, List] = defaultdict(list)
        # Per-SID dataframes built in _build_accessors
        self.sensor_dfs: Dict[int, pd.DataFrame] = {}
        self.audio_stereo: Optional[np.ndarray] = None
        self.bone_sound: Optional[np.ndarray] = None
        self.df: pd.DataFrame = pd.DataFrame()

        self.imu = SensorAccessor(pd.DataFrame(columns=LABELS["imu"]), LABELS["imu"])
        self.barometer = SensorAccessor(pd.DataFrame(columns=LABELS["barometer"]), LABELS["barometer"])
        self.ppg = SensorAccessor(pd.DataFrame(columns=LABELS["ppg"]), LABELS["ppg"])
        self.bone_acc = SensorAccessor(pd.DataFrame(columns=LABELS["bone_acc"]), LABELS["bone_acc"])
        self.optical_temp = SensorAccessor(pd.DataFrame(columns=LABELS["optical_temp"]), LABELS["optical_temp"])

        self.parser: parser.Parser = parser.Parser({
            self.SENSOR_SID["imu"]: parser.SchemePayloadParser(scheme.SensorScheme(
                name='imu',
                sid=self.SENSOR_SID["imu"],
                groups=[
                    scheme.SensorComponentGroupScheme(
                        name='acc',
                        components=[
                            scheme.SensorComponentScheme('x', scheme.ParseType.FLOAT),
                            scheme.SensorComponentScheme('y', scheme.ParseType.FLOAT),
                            scheme.SensorComponentScheme('z', scheme.ParseType.FLOAT),
                        ]
                    ),
                    scheme.SensorComponentGroupScheme(
                        name='gyro',
                        components=[
                            scheme.SensorComponentScheme('x', scheme.ParseType.FLOAT),
                            scheme.SensorComponentScheme('y', scheme.ParseType.FLOAT),
                            scheme.SensorComponentScheme('z', scheme.ParseType.FLOAT),
                        ]
                    ),
                    scheme.SensorComponentGroupScheme(
                        name='mag',
                        components=[
                            scheme.SensorComponentScheme('x', scheme.ParseType.FLOAT),
                            scheme.SensorComponentScheme('y', scheme.ParseType.FLOAT),
                            scheme.SensorComponentScheme('z', scheme.ParseType.FLOAT),
                        ]
                    ),
                ])),
            self.SENSOR_SID["barometer"]: parser.SchemePayloadParser(scheme.SensorScheme(
                name='barometer',
                sid=self.SENSOR_SID["barometer"],
                groups=[
                    scheme.SensorComponentGroupScheme(
                        name='barometer',
                        components=[
                            scheme.SensorComponentScheme('temperature', scheme.ParseType.FLOAT),
                            scheme.SensorComponentScheme('pressure', scheme.ParseType.FLOAT),
                        ]
                    ),
                ])),
            self.SENSOR_SID["ppg"]: parser.SchemePayloadParser(scheme.SensorScheme(
                name='ppg',
                sid=self.SENSOR_SID["ppg"],
                groups=[
                    scheme.SensorComponentGroupScheme(
                        name='ppg',
                        components=[
                            scheme.SensorComponentScheme('red', scheme.ParseType.UINT32),
                            scheme.SensorComponentScheme('ir', scheme.ParseType.UINT32),
                            scheme.SensorComponentScheme('green', scheme.ParseType.UINT32),
                            scheme.SensorComponentScheme('ambient', scheme.ParseType.UINT32),
                        ]
                    ),
                ])),
            self.SENSOR_SID["optical_temp"]: parser.SchemePayloadParser(scheme.SensorScheme(
                name='optical_temp',
                sid=self.SENSOR_SID["optical_temp"],
                groups=[
                    scheme.SensorComponentGroupScheme(
                        name='optical_temp',
                        components=[
                            scheme.SensorComponentScheme('optical_temp', scheme.ParseType.FLOAT),
                        ]
                    ),
                ])),
            self.SENSOR_SID["bone_acc"]: parser.SchemePayloadParser(scheme.SensorScheme(
                name='bone_acc',
                sid=self.SENSOR_SID["bone_acc"],
                groups=[
                    scheme.SensorComponentGroupScheme(
                        name='bone_acc',
                        components=[
                            scheme.SensorComponentScheme('x', scheme.ParseType.INT16),
                            scheme.SensorComponentScheme('y', scheme.ParseType.INT16),
                            scheme.SensorComponentScheme('z', scheme.ParseType.INT16),
                        ]
                    ),
                ])),
            self.SENSOR_SID["microphone"]: parser.MicPayloadParser(
                sample_count=48000,
            ),
        }, verbose=verbose)

        self.parse()
        self._build_accessors()

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
        data_dict = self.parse_result.sensor_dfs
        for name, sid in self.SENSOR_SID.items():
            labels = LABELS.get(name, [f"val{i}" for i in range(0)])
            if sid in data_dict and isinstance(data_dict[sid], pd.DataFrame):
                df = data_dict[sid]
                df = df[~df.index.duplicated(keep="first")]
            else:
                df = pd.DataFrame(columns=labels)

            # Store per-SID dataframe
            self.sensor_dfs[sid] = df

            # Create/update SensorAccessor for this sensor name
            setattr(self, name, SensorAccessor(df, labels))

        # Clear combined dataframe; it will be built lazily on demand
        self.df = pd.DataFrame()

        self.audio_stereo = self.parse_result.audio_stereo

    def list_sensors(self) -> List[str]:
        """Return a list of available sensor names in the dataset."""
        available_sensors = []
        for name, sid in self.SENSOR_SID.items():
            accessor = getattr(self, name, None)
            if isinstance(accessor, SensorAccessor) and not accessor.df.empty:
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
        if isinstance(accessor, SensorAccessor):
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

    def process_bone(
        self,
        target_sampling_rate: int = 16000,
        enable_noise_reduction: bool = True,
        enable_equalization: bool = True,
    ) -> None:
        sid_bone = self.SENSOR_SID["bone_acc"]
        if not self.parse_result[sid_bone]:
            print("❌ No bone sound data available.")
            return

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype="band")
            return b, a

        def band_pass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = filtfilt(b, a, data, axis=0)
            return y

        time_stamps = np.array([item[0] for item in self.parse_result[sid_bone]])
        bone_sound = np.array([item[1] for item in self.parse_result[sid_bone]])

        original_samplerate = 1.0 / np.median(np.diff(time_stamps))

        lowcut_frequency = 150
        highcut_frequency = 400
        filtered_signal = band_pass_filter(
            bone_sound, lowcut_frequency, highcut_frequency, original_samplerate
        )

        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(filtered_signal)
        processed_signal = pca_result[:, 0]

        num_samples_target = int(
            len(processed_signal) * (target_sampling_rate / original_samplerate)
        )
        resampled_signal = resample(processed_signal, num_samples_target)
        current_samplerate = target_sampling_rate

        if enable_noise_reduction:
            n_fft = 2048
            hop_length = n_fft // 8

            frequencies, times_stft, Zxx = stft(
                resampled_signal,
                fs=current_samplerate,
                nperseg=n_fft,
                noverlap=n_fft - hop_length,
            )

            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)

            frame_energy = np.sum(magnitude**2, axis=0)
            noise_segment_length = int(
                current_samplerate * 0.5 / (n_fft / hop_length)
            )
            min_energy_index = np.argmin(frame_energy)
            start_index = max(0, min_energy_index - noise_segment_length // 2)
            end_index = min(magnitude.shape[1], start_index + noise_segment_length)
            if end_index - start_index < noise_segment_length:
                start_index = max(0, end_index - noise_segment_length)

            noise_estimation_segment = magnitude[:, start_index:end_index]
            noise_estimation = np.mean(noise_estimation_segment, axis=1, keepdims=True)

            magnitude_denoised = np.maximum(magnitude - noise_estimation, 0)
            Zxx_denoised = magnitude_denoised * np.exp(1j * phase)
            _, denoised_signal = istft(
                Zxx_denoised,
                fs=current_samplerate,
                nperseg=n_fft,
                noverlap=n_fft - hop_length,
            )

            processed_signal = denoised_signal[: len(resampled_signal)]

        if enable_equalization:
            Q = 2
            gain = 10 ** (-4 / 20)
            w0_110 = 2 * np.pi * 110 / current_samplerate
            w0_220 = 2 * np.pi * 220 / current_samplerate
            alpha_110 = np.sin(w0_110) / (2 * Q)
            alpha_220 = np.sin(w0_220) / (2 * Q)

            a0_110 = 1 + alpha_110 / gain
            a1_110 = -2 * np.cos(w0_110)
            a2_110 = 1 - alpha_110 / gain
            b0_110 = 1 + alpha_110 * gain
            b1_110 = -2 * np.cos(w0_110)
            b2_110 = 1 - alpha_110 * gain

            a0_220 = 1 + alpha_220 / gain
            a1_220 = -2 * np.cos(w0_220)
            a2_220 = 1 - alpha_220 / gain
            b0_220 = 1 + alpha_220 * gain
            b1_220 = -2 * np.cos(w0_220)
            b2_220 = 1 - alpha_220 * gain

            equalized_signal = filtfilt(
                [b0_110, b1_110, b2_110],
                [a0_110, a1_110, a2_110],
                processed_signal,
            )
            processed_signal = filtfilt(
                [b0_220, b1_220, b2_220],
                [a0_220, a1_220, a2_220],
                equalized_signal,
            )

        self.bone_sound = np.int16(
            processed_signal / np.max(np.abs(processed_signal)) * 32767
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            write(tmp.name, target_sampling_rate, self.bone_sound)
            display(Audio(tmp.name))

    def plot(self) -> None:
        fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=False)
        axes = axes.flatten()

        col_titles = [
            "Acc",
            "Gyro",
            "Mag",
            "PPG",
            "Barometer Temp",
            "Barometer Pressure",
            "Bone_Acc",
            "Mic",
            "Optical Temp",
        ]
        for ax, title in zip(axes, col_titles):
            ax.set_title(title)

        if not self.df.empty:
            for axis in ["x", "y", "z"]:
                series = self.imu.acc.get(axis) if "acc" in self.imu._data else None
                if series is not None:
                    axes[0].plot(series.index, series, label=f"acc.{axis}")

            for axis in ["x", "y", "z"]:
                series = self.imu.gyro.get(axis) if "gyro" in self.imu._data else None
                if series is not None:
                    axes[1].plot(series.index, series, label=f"gyro.{axis}")

            for axis in ["x", "y", "z"]:
                series = self.imu.mag.get(axis) if "mag" in self.imu._data else None
                if series is not None:
                    axes[2].plot(series.index, series, label=f"mag.{axis}")

            for label, color in zip(LABELS["ppg"], COLORS["ppg"]):
                name = label.split(".")[1]
                series = getattr(self.ppg, name, None)
                if series is not None:
                    axes[3].plot(series.index, series, label=label, color=color)

            temp = getattr(self.barometer, "temperature", None)
            if temp is not None:
                axes[4].plot(temp.index, temp, label="Temperature")

            pressure = getattr(self.barometer, "pressure", None)
            if pressure is not None:
                axes[5].plot(pressure.index, pressure, label="Pressure")

            for axis in ["x", "y", "z"]:
                series = self.bone_acc.get(axis)
                if series is not None:
                    axes[6].plot(series.index, series, label=f"bone_acc.{axis}")

            optical_temp = getattr(self.optical_temp, "optical_temp", None)
            if optical_temp is not None:
                axes[8].plot(
                    optical_temp.index, optical_temp, label="Optical Temperature"
                )

        if self.audio_stereo is not None:
            inner, outer = self.audio_stereo[:, 0], self.audio_stereo[:, 1]
            sample_rate = 48000
            duration = len(inner) / sample_rate
            times = np.linspace(0, duration, num=len(inner))
            axes[7].plot(times, inner, label="Mic Inner", alpha=0.7)
            axes[7].plot(times, outer, label="Mic Outer", alpha=0.7)

        for ax in axes:
            ax.grid(True)
            if ax.get_legend_handles_labels()[1]:
                ax.legend()

        fig.suptitle(f"Recording: {os.path.basename(self.filename)}", fontsize=14)
        plt.tight_layout()
        plt.show()


def load_recordings(file_paths: Sequence[str]) -> List[SensorDataset]:
    return [SensorDataset(path) for path in file_paths if os.path.isfile(path)]


def display_recordings(recordings: Sequence[SensorDataset]) -> None:
    for ds in recordings:
        ds.plot()
        ds.play_audio()
        print()
