import os
import struct
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        self.data: Dict[int, List] = defaultdict(list)
        self.audio_stereo: Optional[np.ndarray] = None
        self.bone_sound: Optional[np.ndarray] = None
        self.df: pd.DataFrame = pd.DataFrame()

        self.imu = SensorAccessor(pd.DataFrame(columns=LABELS["imu"]), LABELS["imu"])
        self.barometer = SensorAccessor(pd.DataFrame(columns=LABELS["barometer"]), LABELS["barometer"])
        self.ppg = SensorAccessor(pd.DataFrame(columns=LABELS["ppg"]), LABELS["ppg"])
        self.bone_acc = SensorAccessor(pd.DataFrame(columns=LABELS["bone_acc"]), LABELS["bone_acc"])
        self.optical_temp = SensorAccessor(pd.DataFrame(columns=LABELS["optical_temp"]), LABELS["optical_temp"])

        self.parse()
        self._build_accessors()

    def parse(self) -> None:
        """Parse the binary recording file into structured sensor data."""
        FILE_HEADER_FORMAT = "<HQ"
        FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_FORMAT)
        mic_samples: List[int] = []
        _sid: Optional[int] = None
        packet_idx = 0

        with open(self.filename, "rb") as f:
            header_bytes = f.read(FILE_HEADER_SIZE)
            if len(header_bytes) < FILE_HEADER_SIZE:
                if self.verbose:
                    print(f"File {self.filename} is too short to contain a valid header "
                          f"({len(header_bytes)} bytes read, expected {FILE_HEADER_SIZE}).")
                return

            version, timestamp = struct.unpack(FILE_HEADER_FORMAT, header_bytes)
            if self.verbose:
                print(
                    f"File: {os.path.basename(self.filename)}, "
                    f"Version: {version}, Timestamp (since boot): {timestamp}µs"
                )

            while True:
                packet_start = f.tell()
                header = f.read(10)
                if len(header) == 0:
                    if self.verbose:
                        print(f"Reached EOF after {packet_idx} packets.")
                    break

                if len(header) < 10:
                    if self.verbose:
                        print(
                            f"Incomplete packet header at packet #{packet_idx}: "
                            f"got {len(header)} bytes, expected 10. Stopping parse."
                        )
                    break

                sid, size, time = struct.unpack("<BBQ", header)
                timestamp_s = time / 1e6

                if self.verbose:
                    sensor_name =self.SID_NAMES.get(sid, f"sid{sid}")
                    print(
                        f"Packet #{packet_idx}: SID={sid} ({sensor_name}), "
                        f"size={size} bytes, time={timestamp_s:.6f}s"
                    )

                # Sanity check on size and sid
                if size > 192 or sid > 7:
                    if self.verbose:
                        print(
                            f"Invalid header at packet #{packet_idx}: "
                            f"SID={sid}, size={size} (>192 or SID>7)."
                        )

                    # Try to resync instead of immediately giving up
                    if self._attempt_resync(f, packet_start, packet_idx, version):
                        # Do NOT increment packet_idx here; the next loop iteration
                        # will re-parse from the recovered header.
                        continue

                    # Resync failed: optional rollback & stop
                    if _sid is not None and _sid in self.data and self.data[_sid]:
                        if self.verbose:
                            print(
                                f"Rolling back last valid sample for SID={_sid} "
                                f"and stopping parse."
                            )
                        self.data[_sid].pop()
                    break

                _sid = sid
                data = f.read(size)
                if len(data) < size:
                    if self.verbose:
                        print(
                            f"Truncated payload at packet #{packet_idx}: "
                            f"expected {size} bytes, got {len(data)}. Stopping parse."
                        )
                    break

                try:
                    if sid == self.SENSOR_SID["microphone"]:
                        # Expect 16-bit samples; original code assumed exactly 96 samples.
                        # We keep the same behavior but log if size is unexpected.
                        if size != 96 * 2 and self.verbose:
                            print(
                                f"Microphone packet #{packet_idx} has size={size} bytes, "
                                f"expected 192 (96 samples). Trying to unpack anyway."
                            )
                        n_samples = size // 2
                        samples = struct.unpack(f"<{n_samples}h", data[: n_samples * 2])
                        mic_samples.extend(samples)

                    elif version < 2 and sid == self.SENSOR_SID["bone_acc"]:
                        # Legacy bone_acc format: raw payload stored for later processing
                        self.data[sid].append((timestamp_s, data))

                    elif sid in self.sensor_formats:
                        fmt = self.sensor_formats[sid]
                        expected_size = struct.calcsize(fmt)

                        if size == expected_size:
                            values = struct.unpack(fmt, data)
                            self.data[sid].append((timestamp_s, values))

                        elif (size - 2) % expected_size == 0:
                            # Packed samples with delta time at the end
                            delta = struct.unpack("<H", data[-2:])[0] / 1e6
                            data_payload = data[:-2]
                            n_packets = len(data_payload) // expected_size

                            if self.verbose:
                                sensor_name = self.SID_NAMES.get(sid, f"sid{sid}")
                                print(
                                    f"Packed samples for SID={sid} ({sensor_name}) at "
                                    f"{timestamp_s:.6f}s: {n_packets} samples, Δt={delta:.6e}s"
                                )

                            for n in range(n_packets):
                                start = n * expected_size
                                end = (n + 1) * expected_size
                                values = struct.unpack(fmt, data_payload[start:end])
                                self.data[sid].append((timestamp_s + n * delta, values))
                        else:
                            if self.verbose:
                                sensor_name = self.SID_NAMES.get(sid, f"sid{sid}")
                                print(
                                    f"Could not parse SID={sid} ({sensor_name}) at "
                                    f"{timestamp_s:.6f}s: size={size}, expected {expected_size} "
                                    f"or k*{expected_size}+2 for packed samples."
                                )
                            continue

                    else:
                        # Unknown SID, skip but log
                        if self.verbose:
                            print(
                                f"Skipping unknown SID={sid} at packet #{packet_idx}, "
                                f"time={timestamp_s:.6f}s."
                            )
                        continue

                except struct.error as e:
                    if self.verbose:
                        print(
                            f"struct.error while parsing packet #{packet_idx} "
                            f"(SID={sid}, size={size}, time={timestamp_s:.6f}s): {e}"
                        )

                packet_idx += 1

        # Post-processing microphone samples
        if mic_samples:
            if self.verbose:
                print(f"Collected {len(mic_samples)} microphone samples.")
            mic_array = np.array(mic_samples, dtype=np.int16)
            self.audio_stereo = np.column_stack((mic_array[1::2], mic_array[0::2]))

        if version < 2 and len(self.data[self.SENSOR_SID["bone_acc"]]) > 0:
            if self.verbose:
                print("Processing legacy bone_acc format for version < 2.")
            all_samples: List[Sequence[int]] = []
            sample_counts: List[int] = []

            for _, d in self.data[self.SENSOR_SID["bone_acc"]]:
                samples_per_packet = len(d) // struct.calcsize("<3h")
                sample_counts.append(samples_per_packet)

                for i in range(samples_per_packet):
                    offset = i * struct.calcsize("<3h")
                    sample = struct.unpack("<3h", d[offset : offset + struct.calcsize("<3h")])
                    all_samples.append(sample)

            detailed_times: List[float] = []

            for i in range(len(self.data[self.SENSOR_SID["bone_acc"]]) - 1):
                current_time = self.data[self.SENSOR_SID["bone_acc"]][i][0]
                next_time = self.data[self.SENSOR_SID["bone_acc"]][i + 1][0]
                samples_in_packet = sample_counts[i]
                if samples_in_packet > 0:
                    time_diff = (next_time - current_time) / (samples_in_packet + 1)
                    detailed_times.extend(
                        [current_time + j * time_diff for j in range(samples_in_packet)]
                    )

            self.data[self.SENSOR_SID["bone_acc"]] = list(zip(detailed_times, all_samples))

        if self.verbose:
            # Small summary
            print("Parsing summary:")
            for sid, samples in self.data.items():
                sensor_name =self.SID_NAMES.get(sid, f"sid{sid}")
                print(f"  - SID={sid} ({sensor_name}): {len(samples)} samples")

    def _is_plausible_header(self, sid: int, size: int, version: int) -> bool:
        """Quick sanity check to decide if a (sid, size) pair could be a real packet."""
        if not (0 <= sid <= 7):
            return False
        if size <= 0 or size > 192:
            return False

        # Microphone: any even size up to 192 is okay (16-bit samples)
        if sid == self.SENSOR_SID["microphone"]:
            return size % 2 == 0

        # Known sensor formats (IMU, baro, ppg, optical_temp, bone_acc)
        if sid in self.sensor_formats:
            fmt = self.sensor_formats[sid]
            expected_size = struct.calcsize(fmt)

            # 1: Single-sample packet
            if size == expected_size:
                return True

            # 2: Packed samples + 2-byte delta at the end
            if size > 2 and (size - 2) % expected_size == 0:
                return True

            return False

        # Legacy bone_acc (version < 2) can have raw multiples of 3h
        if version < 2 and sid == self.SENSOR_SID["bone_acc"]:
            return size % struct.calcsize("<3h") == 0

        # Unknown SIDs: treat as implausible for resync
        return False

    def _attempt_resync(
        self,
        f,
        packet_start_pos: int,
        packet_idx: int,
        version: int,
        max_scan_bytes: int = 256,
    ) -> bool:
        """Try to recover from a corrupted header by scanning forward for a plausible one.

        Returns True if a new plausible header was found and the file pointer
        was moved to that position; False otherwise (caller should then stop)."""
        original_pos = f.tell()

        if self.verbose:
            print(
                f"Attempting resync after packet #{packet_idx} "
                f"from byte {packet_start_pos} (scan up to {max_scan_bytes} bytes ahead)..."
            )

        for offset in range(1, max_scan_bytes + 1):
            candidate_pos = packet_start_pos + offset
            f.seek(candidate_pos)
            header = f.read(10)
            if len(header) < 10:
                # Not enough bytes left for a full header
                break

            try:
                sid, size, time = struct.unpack("<BBQ", header)
            except struct.error:
                continue

            if not self._is_plausible_header(sid, size, version):
                continue

            # We found something that *looks* like a valid header.
            if self.verbose:
                sensor_name = self.SID_NAMES.get(sid, f"sid{sid}")
                print(
                    f"Resynced at byte offset {candidate_pos} "
                    f"(skipped {candidate_pos - packet_start_pos} bytes): "
                    f"SID={sid} ({sensor_name}), size={size}"
                )

            # Position the stream so that the main loop will re-read this header
            f.seek(candidate_pos)
            return True

        # Failed to find a plausible header; restore file position
        f.seek(original_pos)
        if self.verbose:
            print(
                f"Resync failed within {max_scan_bytes} bytes after packet #{packet_idx}; "
                f"stopping parse."
            )
        return False

    def _build_accessors(self) -> None:
        """Construct per-sensor accessors and a combined DataFrame."""
        dfs: List[pd.DataFrame] = []

        for name, sid in self.SENSOR_SID.items():
            labels = LABELS.get(name, [f"val{i}" for i in range(0)])
            if sid in self.data and self.data[sid]:
                times, values = zip(*self.data[sid])
                df = pd.DataFrame(values, columns=labels)
                df["timestamp"] = times
                df.set_index("timestamp", inplace=True)
                df = df[~df.index.duplicated(keep="first")]
                dfs.append(df)
            else:
                df = pd.DataFrame(columns=labels)

            setattr(self, name, SensorAccessor(df, labels))

        if dfs:
            common_index = pd.Index([])
            for df in dfs:
                common_index = common_index.union(df.index)
            common_index = common_index.sort_values()

            reindexed_dfs = [df.reindex(common_index) for df in dfs]
            self.df = pd.concat(reindexed_dfs, axis=1)
        else:
            self.df = pd.DataFrame()

    def get_dataframe(self) -> pd.DataFrame:
        """Return the combined, time-indexed DataFrame of all sensors."""
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
        if not self.data[sid_bone]:
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

        time_stamps = np.array([item[0] for item in self.data[sid_bone]])
        bone_sound = np.array([item[1] for item in self.data[sid_bone]])

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
