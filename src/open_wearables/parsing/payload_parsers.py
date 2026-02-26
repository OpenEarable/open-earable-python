import struct
from typing import List

from open_wearables.schema import ParseType, SensorScheme

from .base import PayloadParser


class MicPayloadParser(PayloadParser):
    """Payload parser for microphone packets (int16 PCM samples)."""

    def __init__(self, sample_count: int, verbose: bool = False):
        self.sample_count = sample_count
        self.expected_size = sample_count * 2
        self.verbose = verbose

    def parse(self, data: bytes, **kwargs) -> List[dict]:
        if len(data) != self.expected_size and self.verbose:
            print(
                f"Mic payload size {len(data)} bytes does not match expected "
                f"{self.expected_size} bytes (sample_count={self.sample_count})."
            )

        if len(data) % 2 != 0 and self.verbose:
            print(f"Mic payload has odd size {len(data)}; last byte will be ignored.")

        n_samples = len(data) // 2
        format_str = f"<{n_samples}h"
        samples = struct.unpack_from(format_str, data, 0)
        return [{"samples": samples}]

    def should_build_df(self) -> bool:
        return False


class SchemePayloadParser(PayloadParser):
    def __init__(self, sensor_scheme: SensorScheme):
        self.sensor_scheme = sensor_scheme

        size = 0
        for group in self.sensor_scheme.groups:
            for component in group.components:
                if component.data_type in (ParseType.UINT8, ParseType.INT8):
                    size += 1
                elif component.data_type in (ParseType.UINT16, ParseType.INT16):
                    size += 2
                elif component.data_type in (
                    ParseType.UINT32,
                    ParseType.INT32,
                    ParseType.FLOAT,
                ):
                    size += 4
                elif component.data_type == ParseType.DOUBLE:
                    size += 8
                else:
                    raise ValueError(
                        f"Unsupported data type in scheme: {component.data_type}"
                    )
        self.expected_size = size

    def check_size(self, data: bytes) -> None:
        size = len(data)
        if size != self.expected_size and not (
            size > self.expected_size and (size - 2) % self.expected_size == 0
        ):
            raise ValueError(
                f"Payload size {size} bytes does not match expected size "
                f"{self.expected_size} bytes for sensor '{self.sensor_scheme.name}'"
            )

    def is_buffered(self, data: bytes) -> bool:
        size = len(data)
        return size > self.expected_size and (size - 2) % self.expected_size == 0

    def parse(self, data: bytes, **kwargs) -> List[dict]:
        self.check_size(data)
        if self.is_buffered(data):
            results = []
            t_delta = struct.unpack_from("<H", data, len(data) - 2)[0]
            payload = data[:-2]
            n_packets = len(payload) // self.expected_size
            for i in range(n_packets):
                packet_data = payload[i * self.expected_size : (i + 1) * self.expected_size]
                parsed_packet = self.parse_packet(packet_data)
                parsed_packet["t_delta"] = t_delta
                results.append(parsed_packet)
            return results
        return [self.parse_packet(data)]

    def parse_packet(self, data: bytes) -> dict:
        parsed_data = {}
        offset = 0

        for group in self.sensor_scheme.groups:
            group_data = {}
            for component in group.components:
                if component.data_type == ParseType.UINT8:
                    value = struct.unpack_from("<B", data, offset)[0]
                    offset += 1
                elif component.data_type == ParseType.UINT16:
                    value = struct.unpack_from("<H", data, offset)[0]
                    offset += 2
                elif component.data_type == ParseType.UINT32:
                    value = struct.unpack_from("<I", data, offset)[0]
                    offset += 4
                elif component.data_type == ParseType.INT8:
                    value = struct.unpack_from("<b", data, offset)[0]
                    offset += 1
                elif component.data_type == ParseType.INT16:
                    value = struct.unpack_from("<h", data, offset)[0]
                    offset += 2
                elif component.data_type == ParseType.INT32:
                    value = struct.unpack_from("<i", data, offset)[0]
                    offset += 4
                elif component.data_type == ParseType.FLOAT:
                    value = struct.unpack_from("<f", data, offset)[0]
                    offset += 4
                elif component.data_type == ParseType.DOUBLE:
                    value = struct.unpack_from("<d", data, offset)[0]
                    offset += 8
                else:
                    raise ValueError(f"Unsupported data type: {component.data_type}")

                group_data[component.name] = value
            parsed_data[group.name] = group_data

        return parsed_data
