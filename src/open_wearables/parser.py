"""Backward-compatible parser module.

New code should import from :mod:`open_wearables.parsing`; this module keeps
the historic ``open_wearables.parser`` import path working.
"""

from open_wearables.parsing import (
    FrequencyOptions,
    MicPacket,
    MicPayloadParser,
    OeFileHeader,
    ParseInfo,
    ParseResult,
    Parser,
    PayloadParser,
    SchemePayloadParser,
    SensorConfigOptions,
    interleaved_mic_to_stereo,
    iter_sensor_scheme_labels,
    mic_packet_to_stereo_frames,
    parse_parse_info_blob,
    parse_single_sensor_scheme,
    read_oe_header,
    sensor_scheme_labels,
)
from open_wearables.scheme import ParseType, SensorScheme

__all__ = [
    "FrequencyOptions",
    "MicPacket",
    "MicPayloadParser",
    "OeFileHeader",
    "ParseInfo",
    "ParseResult",
    "Parser",
    "PayloadParser",
    "ParseType",
    "SchemePayloadParser",
    "SensorConfigOptions",
    "SensorScheme",
    "interleaved_mic_to_stereo",
    "iter_sensor_scheme_labels",
    "mic_packet_to_stereo_frames",
    "parse_parse_info_blob",
    "parse_single_sensor_scheme",
    "read_oe_header",
    "sensor_scheme_labels",
]
