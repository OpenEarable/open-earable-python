import struct
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional

import pandas as pd

from open_wearables.schema import SensorScheme

from .audio import MicPacket, mic_samples_to_stereo
from .base import ParseResult, PayloadParser
from .headers import OeFileHeader, read_oe_header
from .payload_parsers import MicPayloadParser, SchemePayloadParser


PACKET_HEADER_SIZE = 10


@dataclass(frozen=True)
class PacketHeader:
    """Decoded OpenEarable packet header."""

    sid: int
    payload_size: int
    timestamp_us: int

    @property
    def timestamp_s(self) -> float:
        """Packet timestamp in seconds."""
        return self.timestamp_us / 1e6


class Parser:
    """Incremental parser for OpenEarable packet streams and ``.oe`` files."""

    def __init__(self, parsers: dict[int, PayloadParser], verbose: bool = False):
        """Create a parser from per-SID payload parser registrations."""
        self.parsers = parsers
        self.verbose = verbose
        self.sensor_schemes: dict[int, SensorScheme] = {}

    @classmethod
    def from_sensor_schemes(
        cls,
        sensor_schemes: dict[int, SensorScheme],
        verbose: bool = False,
    ) -> "Parser":
        """Create a parser that decodes every SID with ``SchemePayloadParser``."""
        parsers: dict[int, PayloadParser] = {
            sid: SchemePayloadParser(scheme) for sid, scheme in sensor_schemes.items()
        }
        parser = cls(parsers=parsers, verbose=verbose)
        parser.sensor_schemes = dict(sensor_schemes)
        return parser

    def parse(
        self,
        data_stream: BinaryIO,
        *,
        chunk_size: int = 4096,
        max_resync_scan_bytes: int = 256,
    ) -> ParseResult:
        """Parse a stream from the beginning of an OE file."""
        header_result = read_oe_header(data_stream)
        file_header = header_result.header
        if file_header is not None:
            self._apply_file_header(file_header)
            if self.verbose:
                print(
                    f"Parsed OE header v{file_header.version}: "
                    f"timestamp={file_header.timestamp}, header_size={file_header.header_size}"
                )

        return self.parse_packets(
            data_stream,
            file_header=file_header,
            initial_packet_bytes=header_result.initial_packet_bytes,
            chunk_size=chunk_size,
            max_resync_scan_bytes=max_resync_scan_bytes,
        )

    def parse_packets(
        self,
        data_stream: BinaryIO,
        *,
        file_header: Optional[OeFileHeader] = None,
        initial_packet_bytes: bytes = b"",
        chunk_size: int = 4096,
        max_resync_scan_bytes: int = 256,
    ) -> ParseResult:
        """Parse packet data from a stream positioned at the first packet.

        Parameters
        ----------
        data_stream:
            Binary stream positioned at packet data, not at the file header.
        file_header:
            Optional file-level header metadata to attach to the result.
        initial_packet_bytes:
            Bytes already consumed by a caller before packet parsing starts.
        """
        rows_by_sid: dict[int, list[dict[str, object]]] = {}
        buffer = bytearray(initial_packet_bytes)
        packet_idx = 0
        mic_samples: List[int] = []
        mic_packets: List[MicPacket] = []

        while True:
            if len(buffer) < PACKET_HEADER_SIZE:
                chunk = data_stream.read(chunk_size)
                if not chunk:
                    if self.verbose and buffer:
                        print(
                            f"End of stream with {len(buffer)} leftover bytes (incomplete header/payload)."
                        )
                    break
                buffer.extend(chunk)
                continue

            header = self._read_packet_header(buffer)

            if self.verbose:
                print(
                    f"Packet #{packet_idx}: SID={header.sid}, "
                    f"size={header.payload_size}, time={header.timestamp_s:.6f}s "
                    f"(buffer_len={len(buffer)})"
                )

            if header.sid not in self.parsers:
                if self.verbose:
                    print(
                        f"Warning: No parser registered for SID={header.sid}. "
                        "Attempting resync..."
                    )
                self._discard_until_resync(
                    buffer, packet_idx, max_scan_bytes=max_resync_scan_bytes
                )
                continue

            if header.payload_size <= 0:
                if self.verbose:
                    print(
                        f"Invalid size={header.payload_size} for SID={header.sid}. "
                        "Attempting resync..."
                    )
                self._discard_until_resync(
                    buffer, packet_idx, max_scan_bytes=max_resync_scan_bytes
                )
                continue

            parser = self.parsers[header.sid]
            needed = PACKET_HEADER_SIZE + header.payload_size
            if len(buffer) < needed:
                chunk = data_stream.read(chunk_size)
                if not chunk:
                    if self.verbose:
                        print(
                            f"Truncated payload at packet #{packet_idx}: need {needed} bytes, "
                            f"have {len(buffer)} bytes and stream ended."
                        )
                    break
                buffer.extend(chunk)
                continue

            payload = bytes(buffer[PACKET_HEADER_SIZE:needed])
            try:
                values_list = parser.parse(payload)
                if isinstance(parser, MicPayloadParser):
                    self._append_mic_values(
                        values_list=values_list,
                        timestamp_s=header.timestamp_s,
                        mic_samples=mic_samples,
                        mic_packets=mic_packets,
                    )
                if self.verbose:
                    if isinstance(parser, MicPayloadParser):
                        print(
                            f"Parsed mic packet #{packet_idx} (SID={header.sid}) successfully: "
                            f"{len(values_list[0].get('samples', [])) if values_list else 0} samples"
                        )
                    else:
                        print(
                            f"Parsed packet #{packet_idx} (SID={header.sid}) successfully: "
                            f"{values_list}"
                        )
            except Exception as exc:
                if self.verbose:
                    print(
                        f"struct.error while parsing payload at packet #{packet_idx} "
                        f"(SID={header.sid}, size={header.payload_size}): {exc}. "
                        "Attempting resync..."
                    )
                self._discard_until_resync(
                    buffer, packet_idx, max_scan_bytes=max_resync_scan_bytes
                )
                continue

            if parser.should_build_df():
                self._append_sensor_rows(
                    rows_by_sid=rows_by_sid,
                    sid=header.sid,
                    timestamp_s=header.timestamp_s,
                    values_list=values_list,
                )

            del buffer[:needed]
            packet_idx += 1

        sensor_dfs = self._rows_to_dataframes(rows_by_sid)
        audio_stereo = mic_samples_to_stereo(mic_samples)
        return ParseResult(
            sensor_dfs=sensor_dfs,
            mic_samples=mic_samples,
            mic_packets=mic_packets,
            audio_stereo=audio_stereo,
            file_header=file_header,
        )

    def _apply_file_header(self, file_header: OeFileHeader) -> None:
        """Update parser registrations with schemes embedded in a v3 file header."""
        if not file_header.sensor_schemes:
            return

        for sid, scheme in file_header.sensor_schemes.items():
            existing_parser = self.parsers.get(sid)
            if isinstance(existing_parser, MicPayloadParser):
                continue
            self.parsers[sid] = SchemePayloadParser(scheme)

        self.sensor_schemes.update(file_header.sensor_schemes)

    def _parse_header(self, header: bytes) -> tuple[int, int, int]:
        sid, size, time = struct.unpack("<BBQ", header)
        return sid, size, time

    def _read_packet_header(self, buffer: bytearray) -> PacketHeader:
        """Decode the packet header at the start of ``buffer``."""
        sid, size, time = self._parse_header(bytes(buffer[:PACKET_HEADER_SIZE]))
        return PacketHeader(sid=sid, payload_size=size, timestamp_us=time)

    def _append_mic_values(
        self,
        *,
        values_list: List[dict],
        timestamp_s: float,
        mic_samples: List[int],
        mic_packets: List[MicPacket],
    ) -> None:
        """Accumulate decoded microphone samples and packet timestamps."""
        for item in values_list:
            samples = item.get("samples")
            if samples is None:
                continue
            mic_samples.extend(list(samples))
            mic_packets.append(
                {
                    "timestamp": timestamp_s,
                    "samples": samples,
                }
            )

    def _append_sensor_rows(
        self,
        *,
        rows_by_sid: dict[int, list[dict[str, object]]],
        sid: int,
        timestamp_s: float,
        values_list: List[dict],
    ) -> None:
        """Flatten parsed sensor values and append DataFrame-ready rows."""
        for values in values_list:
            flat_values, timestamp_s = self._flatten_values(values, timestamp_s)
            row = {
                "timestamp": timestamp_s,
                **flat_values,
            }
            rows_by_sid.setdefault(sid, []).append(row)

    @staticmethod
    def _flatten_values(
        values: dict,
        timestamp_s: float,
    ) -> tuple[dict[str, object], float]:
        """Flatten grouped parser output into ``group.component`` columns."""
        flat_values: dict[str, object] = {}
        for key, val in values.items():
            if key == "t_delta":
                timestamp_s += val / 1e6
                continue
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    flat_values[f"{key}.{sub_key}"] = sub_val
            else:
                flat_values[key] = val
        return flat_values, timestamp_s

    @staticmethod
    def _rows_to_dataframes(
        rows_by_sid: dict[int, list[dict[str, object]]],
    ) -> Dict[int, pd.DataFrame]:
        """Convert accumulated parser rows into timestamp-indexed DataFrames."""
        result: Dict[int, pd.DataFrame] = {}
        for sid, rows in rows_by_sid.items():
            df = pd.DataFrame(rows)
            if not df.empty and "timestamp" in df.columns:
                df.set_index("timestamp", inplace=True)
            result[sid] = df
        return result

    def _discard_until_resync(
        self,
        buffer: bytearray,
        packet_idx: int,
        *,
        max_scan_bytes: int,
    ) -> None:
        """Discard corrupt bytes up to the next plausible packet header."""
        new_offset = self._attempt_resync(
            bytes(buffer),
            0,
            packet_idx,
            max_scan_bytes=max_scan_bytes,
        )
        if new_offset is None:
            del buffer[:1]
        else:
            del buffer[:new_offset]

    def _is_plausible_header(self, sid: int, size: int, remaining: int) -> bool:
        if sid not in self.parsers:
            return False
        if size <= 0 or size > remaining:
            return False

        parser = self.parsers[sid]
        if hasattr(parser, "expected_size") and parser.expected_size is not None:
            if size != parser.expected_size:
                return False

        return True

    def _attempt_resync(
        self,
        data: bytes,
        packet_start: int,
        packet_idx: int,
        max_scan_bytes: int = 64,
    ) -> Optional[int]:
        total_len = len(data)

        if self.verbose:
            print(
                f"Attempting resync after packet #{packet_idx} from offset {packet_start} "
                f"(scan up to {max_scan_bytes} bytes ahead)..."
            )

        for delta in range(1, max_scan_bytes + 1):
            candidate = packet_start + delta
            if candidate + PACKET_HEADER_SIZE > total_len:
                break

            header = data[candidate : candidate + PACKET_HEADER_SIZE]
            try:
                sid, size, time = self._parse_header(header)
            except struct.error:
                continue

            remaining = total_len - (candidate + PACKET_HEADER_SIZE)
            if not self._is_plausible_header(sid, size, remaining):
                continue

            if self.verbose:
                timestamp_s = time / 1e6
                print(
                    f"Resynced at offset {candidate} (skipped {delta} bytes): "
                    f"SID={sid}, size={size}, time={timestamp_s:.6f}s"
                )

            return candidate

        if self.verbose:
            print(
                f"Resync failed within {max_scan_bytes} bytes after packet #{packet_idx}; "
                "giving up on this buffer."
            )
        return None
