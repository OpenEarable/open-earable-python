# Open Wearable

Python toolkit for parsing and analyzing multi-sensor OpenEarable recordings.

## Installation

```bash
pip install open-wearables
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
from open_wearables import SensorDataset

dataset = SensorDataset("recording.oe")

# Combined time-indexed DataFrame of all parsed sensors
df = dataset.get_dataframe()

# Per-sensor views
imu_df = dataset.imu.df
ppg_red = dataset.ppg["ppg.red"]
audio_df = dataset.get_audio_dataframe()
```

## IPC WebSocket Example

```python
import asyncio
from open_wearable import OpenWearableIPCClient


async def main() -> None:
    async with OpenWearableIPCClient() as client:
        await client.start_scan()
        devices = await client.get_discovered_devices()
        wearable = client.wearable(devices[0].id)

        await wearable.connect()
        await wearable.actions.synchronize_time()

        sensors = await wearable.actions.list_sensors()
        stream = await wearable.streams.sensor_values(sensor_id=sensors[0].sensor_id)
        async for event in stream:
            print(event.data)
            break
        await stream.close()


asyncio.run(main())
```

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Data model and sensor channels](docs/data-model.md)
- [API reference](docs/api-reference.md)

## Package Architecture

The library is organized into focused layers:

- `open_wearable.schema`: sensor schema types and default schema builders.
- `open_wearable.parsing`: stream parsing, payload parsers, and microphone helpers.
- `open_wearable.data`: high-level dataset API (`SensorDataset`) and sensor accessors.
- `open_wearable.ipc`: asynchronous WebSocket IPC client and protocol models.

Legacy flat modules (`open_wearable.scheme`, `open_wearable.parser`, `open_wearable.dataset`) remain available as compatibility facades.

## License

MIT. See `LICENSE`.
