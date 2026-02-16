# Getting Started

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `ipython` (installed automatically with this package)

## Installation

```bash
pip install open-earable-python
```

From source:

```bash
git clone https://github.com/OpenEarable/open-earable-python.git
cd open-earable-python
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Load a Recording

```python
from open_earable_python import SensorDataset

dataset = SensorDataset("my_recording.oe")
```

`SensorDataset` parses the file immediately during initialization.

## Work with Sensor Data

```python
# Combined DataFrame (all available non-empty sensor streams)
df = dataset.get_dataframe()
print(df.head())

# List non-empty sensor streams
print(dataset.list_sensors())

# Access one sensor DataFrame directly
imu_df = dataset.get_sensor_dataframe("imu")
print(imu_df.columns)
```

## Access Channels via Accessors

```python
# Full IMU DataFrame (columns: acc.x, acc.y, ...)
imu = dataset.imu.df

# Group-level access (columns renamed to x, y, z)
acc = dataset.imu.acc
gyro = dataset.imu.gyro

# Channel-level access
acc_x = dataset.imu.acc["x"]
mag_z = dataset.imu.mag.z
```

## Work with Audio

```python
# Timestamp-indexed stereo audio DataFrame
audio_df = dataset.get_audio_dataframe()  # default 48_000 Hz
print(audio_df.columns)  # mic.inner, mic.outer

# Save WAV
dataset.save_audio("recording.wav")

# Play in Jupyter/IPython environments
dataset.play_audio()
```

## Export CSV

```python
# Build combined DataFrame, then export it
dataset.get_dataframe()
dataset.save_csv("recording.csv")
```

`save_csv()` writes only if the combined DataFrame is already populated (for example after calling `get_dataframe()`).

## Load Multiple Files

```python
from open_earable_python import load_recordings

recordings = load_recordings(["session1.oe", "session2.oe"])
for rec in recordings:
    print(rec.filename, rec.list_sensors())
```
