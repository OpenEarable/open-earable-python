# API Reference

## Package Exports

```python
from open_wearables import SensorDataset, load_recordings
```

Primary internal package layers:

- `open_wearables.schema`
- `open_wearables.parsing`
- `open_wearables.data`
- `open_wearables.ipc`

## IPC Client (`open_wearables.ipc`)

`OpenWearableIPCClient` is an async JSON-RPC style client for
`ws://127.0.0.1:8765/ws` by default.

### Connection Lifecycle

```python
async with OpenWearableIPCClient() as client:
    await client.ping()
```

### Discovery and Connection

- `start_scan(check_and_request_permissions=True)`
- `start_scan_async(check_and_request_permissions=True) -> dict`
- `start_scan_stream(check_and_request_permissions=True) -> StreamSubscription`
- `get_discovered_devices() -> list[DiscoveredDevice]`
- `connect_device(device_id, connected_via_system=False) -> WearableSummary`
- `connect_system_devices(ignored_device_ids=None) -> list[WearableSummary]`
- `list_connected() -> list[WearableSummary]`
- `disconnect(device_id)`

### Action Sugar

- `client.synchronize_time(device_id)`
- `client.list_sensors(device_id) -> list[SensorInfo]`
- `client.list_sensor_configurations(device_id) -> list[SensorConfiguration]`
- `client.set_sensor_configuration(device_id, configuration_name=..., value_key=...)`

Per-device handle:

```python
wearable = client.wearable(device_id)
await wearable.connect()
await wearable.actions.synchronize_time()
```

### Stream Sugar

Use the typed stream helpers:

```python
stream = await wearable.streams.sensor_values(sensor_id="accelerometer_0")
async for event in stream:
    print(event.data)
```

Other helpers:

- `wearable.streams.sensor_configuration()`
- `wearable.streams.button_events()`
- `wearable.streams.battery_percentage()`
- `wearable.streams.battery_power_status()`
- `wearable.streams.battery_health_status()`
- `wearable.streams.battery_energy_status()`

## `SensorDataset`

High-level API for loading and analyzing a single `.oe` recording.

### Constructor

```python
SensorDataset(filename: str, verbose: bool = False)
```

- `filename`: path to `.oe` file.
- `verbose`: enables parser diagnostic output.

Parsing happens during initialization.

### Attributes

- `filename: str` source file path.
- `verbose: bool` parser verbosity flag.
- `parse_result: parser.ParseResult` raw parse output.
- `sensor_dfs: Dict[int, pandas.DataFrame]` per-SID DataFrames.
- `df: pandas.DataFrame` lazily built combined DataFrame.
- `audio_stereo: Optional[numpy.ndarray]` stereo audio frames (`int16`, shape `(N, 2)`).
- `audio_df: pandas.DataFrame` cached audio DataFrame.

Sensor accessor attributes:

- `dataset.imu`
- `dataset.barometer`
- `dataset.microphone`
- `dataset.ppg`
- `dataset.optical_temp`
- `dataset.bone_acc`

Each accessor supports grouped and channel-level access (see data model docs).

### Methods

#### `parse() -> None`

Re-parses the recording file and updates `parse_result`.

#### `list_sensors() -> List[str]`

Returns sensor names with non-empty DataFrames.

#### `get_sensor_dataframe(name: str) -> pandas.DataFrame`

Returns one sensor DataFrame by name.

- Valid names: `imu`, `barometer`, `microphone`, `ppg`, `optical_temp`, `bone_acc`
- Raises `KeyError` for unknown names.

#### `get_dataframe() -> pandas.DataFrame`

Builds and caches a merged DataFrame across all non-empty sensor streams.

#### `get_audio_dataframe(sampling_rate: int = 48000) -> pandas.DataFrame`

Returns timestamp-indexed audio DataFrame with columns:

- `mic.inner`
- `mic.outer`

Behavior:

- Raises `ValueError` if `sampling_rate <= 0`.
- Returns empty DataFrame with expected columns if no mic packets exist.
- Caches by sampling rate.

#### `export_csv() -> None`

Writes combined DataFrame to `<recording_basename>.csv` by delegating to `save_csv()`.

#### `save_csv(path: str) -> None`

Saves the combined DataFrame to CSV if `self.df` is non-empty.

Call `get_dataframe()` first to ensure `self.df` is populated.

#### `play_audio(sampling_rate: int = 48000) -> None`

Plays audio in IPython/Jupyter via `IPython.display.Audio`.

#### `save_audio(path: str, sampling_rate: int = 48000) -> None`

Writes WAV audio with `scipy.io.wavfile.write`.

## `load_recordings`

```python
load_recordings(file_paths: Sequence[str]) -> List[SensorDataset]
```

Creates `SensorDataset` objects for existing files only.

## Parser Module (`open_wearables.parser`)

Core classes and helpers for decoding binary packets:

- `Parser`: stream parser over packetized binary data.
- `PayloadParser`: base parser interface.
- `SchemePayloadParser`: parser built from `SensorScheme`.
- `MicPayloadParser`: parser for microphone payloads.
- `ParseResult`: parse container with per-SID DataFrames and microphone artifacts.
- `interleaved_mic_to_stereo(samples)`: converts interleaved samples to stereo.
- `mic_packet_to_stereo_frames(packet, sampling_rate)`: timestamp + stereo frame conversion.

Note: `open_wearables.parser` is a compatibility facade. New code should prefer
`open_wearables.parsing`.

## Scheme Module (`open_wearables.scheme`)

Defines sensor schema primitives:

- `ParseType` enum
- `SensorComponentScheme`
- `SensorComponentGroupScheme`
- `SensorScheme`
- `build_default_sensor_schemes(sensor_sid)`

Note: `open_wearables.scheme` is a compatibility facade. New code should prefer
`open_wearables.schema`.
