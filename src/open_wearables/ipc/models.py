from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class StreamEvent:
    """A data event emitted by a subscription stream."""

    subscription_id: int
    stream: str
    device_id: str
    data: Any
    raw: dict[str, Any]


@dataclass(frozen=True)
class DiscoveredDevice:
    """Device metadata returned by discovery scans."""

    id: str
    name: str
    service_uuids: list[str]
    manufacturer_data: list[int]
    rssi: Optional[int]
    raw: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DiscoveredDevice":
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            service_uuids=[str(value) for value in payload.get("service_uuids", [])],
            manufacturer_data=[int(value) for value in payload.get("manufacturer_data", [])],
            rssi=int(payload["rssi"]) if payload.get("rssi") is not None else None,
            raw=payload,
        )


@dataclass(frozen=True)
class WearableSummary:
    """Connected wearable summary returned by connect/list methods."""

    device_id: str
    name: str
    type: str
    capabilities: list[str]
    raw: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WearableSummary":
        return cls(
            device_id=str(payload.get("device_id", "")),
            name=str(payload.get("name", "")),
            type=str(payload.get("type", "")),
            capabilities=[str(value) for value in payload.get("capabilities", [])],
            raw=payload,
        )


@dataclass(frozen=True)
class SensorInfo:
    """A sensor entry returned by `invoke_action(action='list_sensors')`."""

    sensor_id: str
    sensor_index: int
    name: str
    chart_title: str
    short_chart_title: str
    axis_names: list[str]
    axis_units: list[str]
    timestamp_exponent: int
    raw: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SensorInfo":
        return cls(
            sensor_id=str(payload.get("sensor_id", "")),
            sensor_index=int(payload.get("sensor_index", 0)),
            name=str(payload.get("name", "")),
            chart_title=str(payload.get("chart_title", "")),
            short_chart_title=str(payload.get("short_chart_title", "")),
            axis_names=[str(value) for value in payload.get("axis_names", [])],
            axis_units=[str(value) for value in payload.get("axis_units", [])],
            timestamp_exponent=int(payload.get("timestamp_exponent", 0)),
            raw=payload,
        )


@dataclass(frozen=True)
class SensorConfigurationValue:
    """A selectable configuration value returned by `list_sensor_configurations`."""

    key: str
    frequency_hz: Optional[float]
    options: list[str]
    raw: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SensorConfigurationValue":
        frequency_raw = payload.get("frequency_hz")
        frequency_hz: Optional[float]
        if frequency_raw is None:
            frequency_hz = None
        else:
            frequency_hz = float(frequency_raw)

        return cls(
            key=str(payload.get("key", "")),
            frequency_hz=frequency_hz,
            options=[str(value) for value in payload.get("options", [])],
            raw=payload,
        )


@dataclass(frozen=True)
class SensorConfiguration:
    """Sensor configuration definition for one sensor."""

    name: str
    unit: str
    values: list[SensorConfigurationValue]
    off_value: Optional[str]
    raw: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SensorConfiguration":
        return cls(
            name=str(payload.get("name", "")),
            unit=str(payload.get("unit", "")),
            values=[
                SensorConfigurationValue.from_payload(value)
                for value in payload.get("values", [])
            ],
            off_value=(
                str(payload["off_value"])
                if payload.get("off_value") is not None
                else None
            ),
            raw=payload,
        )
