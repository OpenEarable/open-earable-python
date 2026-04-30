"""Backward-compatible dataset module.

New code should import from :mod:`open_wearables.data`; this module keeps the
historic ``open_wearables.dataset`` import path working.
"""

from open_wearables.data import (
    COLORS,
    LABELS,
    SENSOR_FORMATS,
    SENSOR_SID,
    SID_NAMES,
    SensorAccessor,
    SensorDataset,
    load_recordings,
)

_SensorAccessor = SensorAccessor

__all__ = [
    "COLORS",
    "LABELS",
    "SENSOR_FORMATS",
    "SENSOR_SID",
    "SID_NAMES",
    "SensorAccessor",
    "SensorDataset",
    "_SensorAccessor",
    "load_recordings",
]
