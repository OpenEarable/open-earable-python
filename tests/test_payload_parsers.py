import struct
import unittest

from open_wearables.parsing import SchemePayloadParser
from open_wearables.schema import ParseType, SensorScheme
from open_wearables.schema.types import group


class SchemePayloadParserTests(unittest.TestCase):
    def test_parse_packet_supports_all_scalar_types(self):
        scheme = SensorScheme(
            name="all_types",
            sid=1,
            groups=[
                group(
                    "sample",
                    [
                        ("uint8", ParseType.UINT8),
                        ("uint16", ParseType.UINT16),
                        ("uint32", ParseType.UINT32),
                        ("int8", ParseType.INT8),
                        ("int16", ParseType.INT16),
                        ("int32", ParseType.INT32),
                        ("float", ParseType.FLOAT),
                        ("double", ParseType.DOUBLE),
                    ],
                )
            ],
        )
        payload = struct.pack(
            "<BHIbhifd",
            255,
            65_535,
            4_000_000_000,
            -12,
            -32_000,
            -2_000_000_000,
            1.25,
            2.5,
        )

        parser = SchemePayloadParser(scheme)
        result = parser.parse(payload)

        self.assertEqual(parser.expected_size, len(payload))
        self.assertEqual(
            result,
            [
                {
                    "sample": {
                        "uint8": 255,
                        "uint16": 65_535,
                        "uint32": 4_000_000_000,
                        "int8": -12,
                        "int16": -32_000,
                        "int32": -2_000_000_000,
                        "float": 1.25,
                        "double": 2.5,
                    }
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
