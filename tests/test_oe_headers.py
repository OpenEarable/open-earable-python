import io
import os
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

if "websockets" not in sys.modules:
    websockets_module = types.ModuleType("websockets")
    websockets_module.ConnectionClosed = Exception
    websockets_client_module = types.ModuleType("websockets.client")
    websockets_client_module.WebSocketClientProtocol = object
    websockets_client_module.connect = object
    sys.modules["websockets"] = websockets_module
    sys.modules["websockets.client"] = websockets_client_module

from open_wearables.parsing import Parser, parse_single_sensor_scheme
from open_wearables.data.sensor_dataset import SensorDataset
from open_wearables.schema import ParseType, SensorScheme
from open_wearables.schema.types import group


def _text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<B", len(encoded)) + encoded


def _single_sensor_scheme(
    *,
    sid: int,
    sensor_name: str,
    group_name: str,
    component_name: str,
    parse_type: int,
    frequencies: tuple[float, ...] = (),
    default_frequency_index: int = 0,
) -> bytes:
    options = struct.pack("<B", 0)
    if frequencies:
        options = (
            struct.pack(
                "<BBBB",
                0x10,
                len(frequencies),
                default_frequency_index,
                len(frequencies) - 1,
            )
            + struct.pack(f"<{len(frequencies)}f", *frequencies)
        )

    return b"".join(
        [
            struct.pack("<B", sid),
            _text(sensor_name),
            struct.pack("<B", 1),
            struct.pack("<B", parse_type),
            _text(group_name),
            _text(component_name),
            _text("unit"),
            options,
        ]
    )


def _parse_info_blob(sensor_ids: list[int], schemes: list[bytes]) -> bytes:
    blob = bytearray()
    blob.extend(struct.pack("<B", len(sensor_ids)))
    blob.extend(sensor_ids)
    for scheme in schemes:
        blob.extend(struct.pack("<H", len(scheme)))
        blob.extend(scheme)
    return bytes(blob)


def _v3_header(parse_info_blob: bytes) -> bytes:
    fixed_size = 27
    header_size = fixed_size + len(parse_info_blob)
    return (
        struct.pack(
            "<HQIIQB",
            3,
            123456789,
            header_size,
            len(parse_info_blob),
            0x0102030405060708,
            1,
        )
        + parse_info_blob
    )


class OeHeaderTests(unittest.TestCase):
    def test_raw_packet_stream_without_file_header_is_rejected(self):
        scheme = SensorScheme(
            name="test",
            sid=9,
            groups=[group("sample", [("value", ParseType.FLOAT)])],
        )
        stream = io.BytesIO(struct.pack("<BBQf", 9, 4, 1_000_000, 1.25))

        with self.assertRaisesRegex(ValueError, "Unsupported OE header version"):
            Parser.from_sensor_schemes({9: scheme}).parse(stream)

    def test_version_2_header_is_skipped_before_packet_parsing(self):
        scheme = SensorScheme(
            name="test",
            sid=9,
            groups=[group("sample", [("value", ParseType.FLOAT)])],
        )
        stream = io.BytesIO(
            struct.pack("<HQ", 2, 42)
            + struct.pack("<BBQf", 9, 4, 1_000_000, 2.5)
        )

        result = Parser.from_sensor_schemes({9: scheme}).parse(stream)

        self.assertEqual(result.file_header.version, 2)
        self.assertEqual(result.file_header.timestamp, 42)
        self.assertEqual(result.sensor_dfs[9].iloc[0]["sample.value"], 2.5)

    def test_version_3_header_supplies_embedded_sensor_scheme(self):
        sensor_scheme = _single_sensor_scheme(
            sid=9,
            sensor_name="Dynamic Sensor",
            group_name="dynamic",
            component_name="value",
            parse_type=6,
        )
        parse_info = _parse_info_blob([9], [sensor_scheme])
        stream = io.BytesIO(
            _v3_header(parse_info)
            + struct.pack("<BBQf", 9, 4, 2_000_000, 3.75)
        )

        result = Parser(parsers={}).parse(stream)

        self.assertEqual(result.file_header.version, 3)
        self.assertEqual(result.file_header.device_id, 0x0102030405060708)
        self.assertEqual(result.file_header.side, 1)
        self.assertEqual(result.file_header.sensor_ids, (9,))
        self.assertEqual(
            result.file_header.sensor_schemes[9].name,
            "Dynamic Sensor",
        )
        self.assertEqual(result.sensor_dfs[9].iloc[0]["dynamic.value"], 3.75)

    def test_sensor_dataset_builds_parser_after_reading_v3_header(self):
        sensor_scheme = _single_sensor_scheme(
            sid=0,
            sensor_name="Dynamic IMU",
            group_name="dynamic",
            component_name="value",
            parse_type=6,
            frequencies=(25.0, 50.0),
            default_frequency_index=1,
        )
        parse_info = _parse_info_blob([0], [sensor_scheme])
        content = (
            _v3_header(parse_info)
            + struct.pack("<BBQf", 0, 4, 2_000_000, 4.25)
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".oe") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            with mock.patch(
                "open_wearables.data.sensor_dataset.build_default_sensor_schemes",
                side_effect=AssertionError("default schemes must not be used for v3"),
            ):
                dataset = SensorDataset(temp_path)
        finally:
            os.unlink(temp_path)

        self.assertEqual(dataset.file_header.version, 3)
        self.assertEqual(dataset.imu.df.iloc[0]["dynamic.value"], 4.25)
        self.assertEqual(dataset.get_sampling_rate("imu"), 50.0)
        self.assertEqual(dataset.get_sampling_rate(0), 50.0)
        self.assertEqual(dataset.get_sampling_rates()["imu"], 50.0)
        self.assertIsNone(dataset.get_sampling_rates()["ppg"])

    def test_sensor_dataset_sampling_rate_returns_none_without_metadata(self):
        scheme = SensorScheme(
            name="test",
            sid=0,
            groups=[group("sample", [("value", ParseType.FLOAT)])],
        )
        content = (
            struct.pack("<HQ", 2, 42)
            + struct.pack("<BBQf", 0, 4, 1_000_000, 2.5)
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".oe") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            with mock.patch(
                "open_wearables.data.sensor_dataset.build_default_sensor_schemes",
                return_value={0: scheme},
            ):
                dataset = SensorDataset(temp_path)
        finally:
            os.unlink(temp_path)

        self.assertIsNone(dataset.get_sampling_rate("imu"))
        self.assertIsNone(dataset.get_sampling_rate(0))

    def test_single_sensor_scheme_parses_frequency_options(self):
        scheme = b"".join(
            [
                struct.pack("<B", 9),
                _text("Dynamic Sensor"),
                struct.pack("<B", 1),
                struct.pack("<B", 6),
                _text("dynamic"),
                _text("value"),
                _text("unit"),
                struct.pack("<BBBB", 0x10, 2, 1, 1),
                struct.pack("<ff", 25.0, 50.0),
            ]
        )

        parsed_scheme, options = parse_single_sensor_scheme(scheme)

        self.assertEqual(parsed_scheme.sid, 9)
        self.assertEqual(options.available_options, 0x10)
        self.assertEqual(options.frequency_options.frequencies, (25.0, 50.0))


if __name__ == "__main__":
    unittest.main()
