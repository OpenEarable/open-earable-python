# Open Earable Python

Python toolkit for parsing and analyzing multi-sensor OpenEarable recordings.

## Installation

```bash
pip install open-earable-python
```

For local development:

```bash
git clone https://github.com/OpenEarable/open-earable-python.git
cd open-earable-python
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Example

```python
from open_earable_python import SensorDataset

dataset = SensorDataset("recording.oe")

# Combined time-indexed DataFrame of all parsed sensors
df = dataset.get_dataframe()

# Per-sensor views
imu_df = dataset.imu.df
ppg_red = dataset.ppg["ppg.red"]
audio_df = dataset.get_audio_dataframe()
```

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Data model and sensor channels](docs/data-model.md)
- [API reference](docs/api-reference.md)

## License

MIT. See `LICENSE`.
