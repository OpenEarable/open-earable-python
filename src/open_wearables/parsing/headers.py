"""Utilities for reading OpenEarable ``.oe`` file headers."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Iterable, List, Optional, Union

from open_wearables.schema import (
    ParseType,
    SensorComponentGroupScheme,
    SensorComponentScheme,
    SensorScheme,
)


OE_HEADER_VERSION_2 = 0x0002
OE_HEADER_VERSION_3 = 0x0003
OE_V2_HEADER_SIZE = 10
OE_V3_FIXED_HEADER_SIZE = 27
FREQUENCIES_DEFINED = 0x10


_FIRMWARE_PARSE_TYPES: Dict[int, ParseType] = {
    0: ParseType.INT8,
    1: ParseType.UINT8,
    2: ParseType.INT16,
    3: ParseType.UINT16,
    4: ParseType.INT32,
    5: ParseType.UINT32,
    6: ParseType.FLOAT,
    7: ParseType.DOUBLE,
}


@dataclass(frozen=True)
class FrequencyOptions:
    """Available sample rates encoded in a v3 ParseInfo sensor scheme."""

    frequency_count: int
    default_frequency_index: int
    max_ble_frequency_index: int
    frequencies: tuple[float, ...]


@dataclass(frozen=True)
class SensorConfigOptions:
    """Configuration metadata appended to a single sensor scheme."""

    available_options: int
    frequency_options: Optional[FrequencyOptions] = None


@dataclass(frozen=True)
class OeFileHeader:
    """Parsed OpenEarable file-level header metadata."""

    version: int
    timestamp: int
    header_size: int
    parse_info_size: int = 0
    device_id: Optional[int] = None
    side: Optional[int] = None
    sensor_ids: tuple[int, ...] = ()
    sensor_schemes: Dict[int, SensorScheme] = field(default_factory=dict)
    sensor_config_options: Dict[int, SensorConfigOptions] = field(default_factory=dict)


@dataclass(frozen=True)
class HeaderReadResult:
    """Result of attempting to consume an ``.oe`` header from a stream."""

    header: Optional[OeFileHeader]
    initial_packet_bytes: bytes = b""


class _ByteReader:
    """Bounds-checked reader for little-endian ParseInfo byte blobs."""

    def __init__(self, data: bytes):
        self._data = data
        self.offset = 0

    @property
    def remaining(self) -> int:
        """Number of unread bytes."""
        return len(self._data) - self.offset

    def read_uint8(self) -> int:
        """Read an unsigned 8-bit integer."""
        return int(self._unpack("<B", 1))

    def read_uint16(self) -> int:
        """Read an unsigned 16-bit integer."""
        return int(self._unpack("<H", 2))

    def read_float(self) -> float:
        """Read a 32-bit IEEE-754 float."""
        return float(self._unpack("<f", 4))

    def read_bytes(self, size: int) -> bytes:
        """Read exactly ``size`` bytes."""
        if size < 0:
            raise ValueError(f"Cannot read a negative byte count: {size}")
        if self.offset + size > len(self._data):
            raise ValueError(
                f"ParseInfo blob ended early at offset {self.offset}; "
                f"need {size} bytes, have {self.remaining}"
            )
        result = self._data[self.offset : self.offset + size]
        self.offset += size
        return result

    def read_text(self) -> str:
        """Read a length-prefixed UTF-8 string from the blob."""
        size = self.read_uint8()
        return self.read_bytes(size).decode("utf-8")

    def _unpack(self, fmt: str, size: int) -> Union[int, float]:
        if self.offset + size > len(self._data):
            raise ValueError(
                f"ParseInfo blob ended early at offset {self.offset}; "
                f"need {size} bytes, have {self.remaining}"
            )
        value = struct.unpack_from(fmt, self._data, self.offset)[0]
        self.offset += size
        return value


def read_oe_header(data_stream: BinaryIO) -> HeaderReadResult:
    """Read a supported ``.oe`` file header from the stream."""

    first_two = data_stream.read(2)
    if len(first_two) < 2:
        raise ValueError("Stream ended before a valid OE header could be read")

    version = struct.unpack("<H", first_two)[0]
    if version <= OE_HEADER_VERSION_2:
        fixed_tail = _read_exact(data_stream, OE_V2_HEADER_SIZE - 2)
        timestamp = struct.unpack("<Q", fixed_tail)[0]
        return HeaderReadResult(
            header=OeFileHeader(
                version=version,
                timestamp=timestamp,
                header_size=OE_V2_HEADER_SIZE,
            )
        )

    if version != OE_HEADER_VERSION_3:
        raise ValueError(f"Unsupported OE header version: {version:#04x}")

    fixed_tail = _read_exact(data_stream, OE_V3_FIXED_HEADER_SIZE - 2)
    timestamp, header_size, parse_info_size, device_id, side = struct.unpack(
        "<QIIQB",
        fixed_tail,
    )
    if header_size < OE_V3_FIXED_HEADER_SIZE:
        raise ValueError(f"Invalid v3 OE header size: {header_size}")
    if parse_info_size != header_size - OE_V3_FIXED_HEADER_SIZE:
        raise ValueError(
            f"Invalid v3 OE parse info size: {parse_info_size}; "
            f"expected {header_size - OE_V3_FIXED_HEADER_SIZE}"
        )

    parse_info_blob = _read_exact(data_stream, parse_info_size)
    parsed_info = parse_parse_info_blob(parse_info_blob)
    return HeaderReadResult(
        header=OeFileHeader(
            version=version,
            timestamp=timestamp,
            header_size=header_size,
            parse_info_size=parse_info_size,
            device_id=device_id,
            side=side,
            sensor_ids=parsed_info.sensor_ids,
            sensor_schemes=parsed_info.sensor_schemes,
            sensor_config_options=parsed_info.sensor_config_options,
        )
    )


@dataclass(frozen=True)
class ParseInfo:
    """Decoded v3 ParseInfo blob contents."""

    sensor_ids: tuple[int, ...]
    sensor_schemes: Dict[int, SensorScheme]
    sensor_config_options: Dict[int, SensorConfigOptions]


def parse_parse_info_blob(blob: bytes) -> ParseInfo:
    """Decode the v3 ParseInfo blob into sensor IDs and schemes."""

    reader = _ByteReader(blob)
    sensor_count = reader.read_uint8()
    sensor_ids = tuple(reader.read_uint8() for _ in range(sensor_count))
    sensor_schemes: Dict[int, SensorScheme] = {}
    config_options: Dict[int, SensorConfigOptions] = {}

    for expected_sid in sensor_ids:
        scheme_size = reader.read_uint16()
        scheme_blob = reader.read_bytes(scheme_size)
        scheme, options = parse_single_sensor_scheme(scheme_blob)
        if scheme.sid != expected_sid:
            raise ValueError(
                f"ParseInfo sensor list expected SID {expected_sid}, "
                f"but scheme encoded SID {scheme.sid}"
            )
        sensor_schemes[scheme.sid] = scheme
        config_options[scheme.sid] = options

    if reader.remaining:
        raise ValueError(f"ParseInfo blob has {reader.remaining} trailing bytes")

    return ParseInfo(
        sensor_ids=sensor_ids,
        sensor_schemes=sensor_schemes,
        sensor_config_options=config_options,
    )


def parse_single_sensor_scheme(blob: bytes) -> tuple[SensorScheme, SensorConfigOptions]:
    """Decode one firmware ``Single Sensor Scheme`` payload."""

    reader = _ByteReader(blob)
    sensor_id = reader.read_uint8()
    sensor_name = reader.read_text()
    component_count = reader.read_uint8()
    groups_by_name: Dict[str, List[SensorComponentScheme]] = {}
    group_order: List[str] = []

    for _ in range(component_count):
        parse_type_id = reader.read_uint8()
        try:
            parse_type = _FIRMWARE_PARSE_TYPES[parse_type_id]
        except KeyError as exc:
            raise ValueError(f"Unsupported firmware parse type: {parse_type_id}") from exc

        group_name = reader.read_text()
        component_name = reader.read_text()
        reader.read_text()  # Unit metadata is not needed for payload decoding.

        if group_name not in groups_by_name:
            groups_by_name[group_name] = []
            group_order.append(group_name)
        groups_by_name[group_name].append(
            SensorComponentScheme(name=component_name, data_type=parse_type)
        )

    options = _read_sensor_config_options(reader)
    if reader.remaining:
        raise ValueError(
            f"Single sensor scheme for SID {sensor_id} has {reader.remaining} trailing bytes"
        )

    groups = [
        SensorComponentGroupScheme(name=name, components=groups_by_name[name])
        for name in group_order
    ]
    return SensorScheme(name=sensor_name, sid=sensor_id, groups=groups), options


def _read_sensor_config_options(reader: _ByteReader) -> SensorConfigOptions:
    available_options = reader.read_uint8()
    frequency_options = None
    if available_options & FREQUENCIES_DEFINED:
        frequency_count = reader.read_uint8()
        default_frequency_index = reader.read_uint8()
        max_ble_frequency_index = reader.read_uint8()
        frequencies = tuple(reader.read_float() for _ in range(frequency_count))
        frequency_options = FrequencyOptions(
            frequency_count=frequency_count,
            default_frequency_index=default_frequency_index,
            max_ble_frequency_index=max_ble_frequency_index,
            frequencies=frequencies,
        )
    return SensorConfigOptions(
        available_options=available_options,
        frequency_options=frequency_options,
    )


def _read_exact(data_stream: BinaryIO, size: int) -> bytes:
    data = data_stream.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected end of stream while reading {size} header bytes")
    return data


def sensor_scheme_labels(scheme: SensorScheme) -> List[str]:
    """Return flattened DataFrame labels produced by ``SchemePayloadParser``."""

    return [
        f"{group.name}.{component.name}"
        for group in scheme.groups
        for component in group.components
    ]


def iter_sensor_scheme_labels(schemes: Iterable[SensorScheme]) -> Dict[int, List[str]]:
    """Build flattened labels for each scheme keyed by sensor ID."""

    return {scheme.sid: sensor_scheme_labels(scheme) for scheme in schemes}
