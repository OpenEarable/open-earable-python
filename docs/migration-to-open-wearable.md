# Migration to open-wearable

`open-earable-python` has been renamed to `open-wearable`.

This follows the common Python package-rename migration pattern:

- Keep the old package available temporarily as a compatibility package.
- Emit a runtime deprecation warning from the old import path.
- Point users to a single new package name and migration guide.
- Set and communicate a concrete deprecation timeline.

## Timeline

- Rename announced: February 24, 2026.
- `open-earable-python` maintenance mode until December 31, 2026 (critical fixes only).
- New features should target `open-wearable`.

## What to Change

1. Update install commands:

```bash
pip uninstall open-earable-python
pip install open-wearable
```

2. Update imports:

```python
# Old (deprecated)
from open_earable_python import SensorDataset, load_recordings

# New
from open_wearable import SensorDataset, load_recordings
```

3. Pin the new package in your environment files (`requirements.txt`, lockfiles, CI configs).

## Notes for Maintainers

- Keep the deprecation warning in the legacy package import path.
- Keep this migration guide linked from README, docs index, and package metadata.
- Keep deprecation and end-of-support dates explicit in release notes.
