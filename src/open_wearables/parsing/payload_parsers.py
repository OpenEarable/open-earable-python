import struct
from typing import List

from open_wearables.schema import ParseType, SensorScheme

from .base import PayloadParser


_PARSE_TYPE_FORMATS: dict[ParseType, tuple[str, int]] = {
    ParseType.UINT8: ("<B", 1),
    ParseType.UINT16: ("<H", 2),
    ParseType.UINT32: ("<I", 4),
    ParseType.INT8: ("<b", 1),
    ParseType.INT16: ("<h", 2),
    ParseType.INT32: ("<i", 4),
    ParseType.FLOAT: ("<f", 4),
    ParseType.DOUBLE: ("<d", 8),
}


class MicPayloadParser(PayloadParser):
    """Payload parser for microphone packets (int16 PCM samples)."""

    def __init__(self, sample_count: int, verbose: bool = False):
        """Create a microphone payload parser.

        Parameters
        ----------
        sample_count:
            Expected number of little-endian int16 samples per packet.
        verbose:
            Enables diagnostic output for malformed microphone payloads.
        """
        self.sample_count = sample_count
        self.expected_size = sample_count * 2
        self.verbose = verbose

    def parse(self, data: bytes, **kwargs) -> List[dict]:
        """Decode microphone payload bytes into a sample tuple."""
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
    """Payload parser driven by an OpenEarable ``SensorScheme`` definition."""

    def __init__(self, sensor_scheme: SensorScheme):
        """Create a parser for payloads matching ``sensor_scheme``."""
        self.sensor_scheme = sensor_scheme
        self.expected_size = self._calculate_expected_size(sensor_scheme)

    @staticmethod
    def _calculate_expected_size(sensor_scheme: SensorScheme) -> int:
        """Return the byte size for one unbuffered sample in ``sensor_scheme``."""
        return sum(
            _format_for(component.data_type)[1]
            for group in sensor_scheme.groups
            for component in group.components
        )

    def check_size(self, data: bytes) -> None:
        """Validate that ``data`` is either one sample or a buffered payload."""
        size = len(data)
        if size != self.expected_size and not (
            size > self.expected_size and (size - 2) % self.expected_size == 0
        ):
            raise ValueError(
                f"Payload size {size} bytes does not match expected size "
                f"{self.expected_size} bytes for sensor '{self.sensor_scheme.name}'"
            )

    def is_buffered(self, data: bytes) -> bool:
        """Return whether ``data`` contains multiple samples and a time delta."""
        size = len(data)
        return size > self.expected_size and (size - 2) % self.expected_size == 0

    def parse(self, data: bytes, **kwargs) -> List[dict]:
        """Decode a single-sample or buffered structured sensor payload."""
        self.check_size(data)
        if self.is_buffered(data):
            results = []
            t_delta = struct.unpack_from("<H", data, len(data) - 2)[0]
            payload = data[:-2]
            n_packets = len(payload) // self.expected_size
            for i in range(n_packets):
                packet_data = payload[
                    i * self.expected_size : (i + 1) * self.expected_size
                ]
                parsed_packet = self.parse_packet(packet_data)
                parsed_packet["t_delta"] = t_delta
                results.append(parsed_packet)
            return results
        return [self.parse_packet(data)]

    def parse_packet(self, data: bytes) -> dict:
        """Decode one unbuffered structured sensor sample."""
        parsed_data = {}
        offset = 0

        for group in self.sensor_scheme.groups:
            group_data = {}
            for component in group.components:
                format_string, size = _format_for(component.data_type)
                value = struct.unpack_from(format_string, data, offset)[0]
                offset += size

                group_data[component.name] = value
            parsed_data[group.name] = group_data

        return parsed_data


def _format_for(parse_type: ParseType) -> tuple[str, int]:
    """Return the struct format and byte width for ``parse_type``."""
    try:
        return _PARSE_TYPE_FORMATS[parse_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported data type: {parse_type}") from exc
