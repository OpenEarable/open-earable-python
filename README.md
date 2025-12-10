# open-earable-python

Python library to parse, analyze, and visualize multi-sensor recordings
from an OpenEarable device (IMU, barometer, PPG, bone accelerometer, optical temperature, and microphone).

## Installation

Once published on PyPI:

```bash
pip install open-earable-python
```

## Usage

```python
from open_earable_python import SensorDataset, load_recordings

ds = SensorDataset("my_recording.oe")

# Pandas DataFrame of all sensors
df = ds.get_dataframe()
print(df.head())

# Plot all sensors
ds.plot()

# Play microphone audio (in Jupyter)
ds.play_audio()

# Process bone accelerometer into audio
ds.process_bone()
```
