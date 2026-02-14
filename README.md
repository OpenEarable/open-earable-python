# Open Earable Python

A Python toolkit for parsing and analyzing multi-sensor recordings from an OpenEarable device. The library provides pandas-friendly accessors for IMU, barometer, PPG, bone accelerometer, optical temperature, and microphone data, along with audio utilities.

## Features
- Load `.oe` recordings into a single time-aligned pandas DataFrame.
- Convenient attribute and key-based accessors for grouped sensors and individual channels.
- Play or export microphone audio directly from notebooks.
- Export combined sensor data to CSV for downstream analysis.

## Installation
The package targets Python 3.9+.

Once published to PyPI:

```bash
pip install open-earable-python
```

From source (for development):

```bash
git clone https://github.com/OpenEarable/open-earable-python.git
cd open-earable-python
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart
Load a recording and explore the combined DataFrame:

```python
from open_earable_python import SensorDataset

# Load a single .oe file
recording = SensorDataset("my_recording.oe")

# Time-indexed dataframe containing all available sensors
full_df = recording.get_dataframe()
print(full_df.head())

# Export to CSV
recording.save_csv("my_recording.csv")
```

### Sensor access patterns
Each sensor has an accessor exposing both grouped views and individual channels using attribute or key syntax. For IMU data:

```python
imu = recording.imu

# Full IMU dataframe (original column names retained)
imu.df          # or imu.to_dataframe()
imu["acc.x"]   # Column-style access

# Accelerometer
imu.acc         # Accelerometer dataframe
imu.acc["x"]   # Accelerometer X channel
imu.acc["y"]
imu.acc["z"]

# Gyroscope
imu.gyro        # Gyroscope dataframe
imu.gyro["x"]
imu.gyro["y"]
imu.gyro["z"]

# Magnetometer
imu.mag          # Magnetometer dataframe
imu.mag["x"]
imu.mag["y"]
imu.mag["z"]
```

PPG channels follow the same pattern:

```python
ppg = recording.ppg
ppg.df           # Full PPG dataframe
ppg["ppg.red"]  # Column-style access
ppg["red"]      # Channel shortcut
ppg.ir
ppg.green
ppg.ambient
```

### Working with multiple recordings
Load several files at once and iterate over them:

```python
from open_earable_python.dataset import load_recordings

paths = ["session1.oe", "session2.oe"]
recordings = load_recordings(paths)

# Access a specific recording
first = recordings[0]
print(first.list_sensors())
```

## IPC Control API (WebSocket)
Use `OpenEarableIPCClient` to connect to your local IPC daemon (`ws://127.0.0.1:8765/ws`) and control devices live.

```python
import asyncio
from open_earable_python import OpenEarableIPCClient


async def main():
    async with OpenEarableIPCClient() as client:
        await client.start_scan(check_and_request_permissions=True)

        # Wait for a discovered device.
        scan_event = await client.wait_for_event("scan", timeout=10)
        device_id = scan_event["device"]["id"]

        await client.connect_device(device_id)
        wearable = await client.get_wearable(device_id)
        print("Connected:", wearable["name"])

        # Configure sensor stream frequency via action API.
        await client.invoke_action(
            device_id=device_id,
            action="set_sensor_frequency_best_effort",
            args={
                "configuration_name": "acceleration",
                "target_hz": 50,
                "stream_data": True,
            },
        )

        # Subscribe and consume stream events.
        subscription = await client.subscribe(
            device_id=device_id,
            stream="sensor_values",
            args={"sensor_name": "acceleration"},
        )

        samples = 0
        async for event in subscription:
            print(event.data)
            samples += 1
            if samples >= 10:
                break

        await subscription.close()
        await client.disconnect(device_id)


asyncio.run(main())
```

### IPC notes
- Use `client.call(method, params)` for raw access to any server method.
- Use `client.on_event(name, callback)` to listen to events like `ready`, `scan`, `connected`, and `stream`.
- `subscribe(...)` returns an async iterator. `stream_error` raises `IPCStreamError` in the iterator.

### Audio utilities
- `play_audio(sampling_rate=48000)`: play stereo microphone data in a Jupyter environment.
- `save_audio(path, sampling_rate=48000)`: export microphone audio to WAV.
