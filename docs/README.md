# Open Earable Python Documentation

> [!WARNING]
> `open-earable-python` is deprecated and renamed to **`open-wearable`**.
> Use `pip install open-wearable` and migrate imports to `open_wearable`.
> See [Migration to open-wearable](migration-to-open-wearable.md).

`open-earable-python` parses `.oe` recordings into pandas DataFrames and exposes convenient accessors for OpenEarable sensor streams.

## Contents

- [Getting started](getting-started.md)
- [Data model and sensor channels](data-model.md)
- [API reference](api-reference.md)
- [Migration to open-wearable](migration-to-open-wearable.md)

## Package Scope

- Parse binary OpenEarable streams into structured sensor samples.
- Build per-sensor and combined time-indexed DataFrames.
- Decode microphone PCM samples and export/play audio.
- Load one or multiple recordings with the same API.
