import struct
from open_earable_python.scheme import SensorScheme, ParseType
import pandas as pd
from typing import BinaryIO

class PayloadParser:
    """Abstract base class for payload parsers.

    Subclasses must set ``expected_size`` and implement :meth:`parse`.
    """

    expected_size: int

    def parse(self, data: bytes) -> list[dict]:
        """Parse a payload into a dictionary of values.

        Parameters
        ----------
        data:
            Raw payload bytes (without header).
        """
        raise NotImplementedError

class Parser:
    def __init__(self, parsers: dict[int, PayloadParser], verbose: bool = False):
        """Create a Parser from a mapping of SID -> PayloadParser."""
        self.parsers = parsers
        self.verbose = verbose

    @classmethod
    def from_sensor_schemes(
        cls,
        sensor_schemes: dict[int, SensorScheme],
        verbose: bool = False,
    ) -> "Parser":
        """Construct a Parser where each SID uses a SchemePayloadParser.

        This does **not** add a special microphone parser; callers can
        override or extend the parser mapping for microphone SIDs as needed.
        """
        parsers: dict[int, PayloadParser] = {
            sid: SchemePayloadParser(scheme) for sid, scheme in sensor_schemes.items()
        }
        return cls(parsers=parsers, verbose=verbose)

    def parse(
        self,
        data_stream: BinaryIO,
        *,
        chunk_size: int = 4096,
        max_resync_scan_bytes: int = 256,
    ) -> dict[int, pd.DataFrame]:
        """Parse a binary byte stream into per-SID DataFrames.

        This function reads from `data_stream` incrementally in chunks and keeps an
        internal buffer so the entire stream does not need to be loaded into memory.

        Parameters
        ----------
        data_stream:
            A binary stream (file-like object) positioned at the beginning of packet data.
            Note: If this is an .oe file, the caller should have already consumed the
            file header before passing the stream here.
        chunk_size:
            Number of bytes to read per chunk.
        max_resync_scan_bytes:
            How many bytes ahead to scan when attempting to resynchronize after a corrupted
            header/payload.

        Returns
        -------
        dict
            Mapping from SID -> pandas DataFrame indexed by "timestamp".
        """
        rows_by_sid: dict[int, list[dict]] = {}

        header_size = 10
        buffer = bytearray()
        packet_idx = 0

        def flush_to_dataframes() -> dict[int, pd.DataFrame]:
            result: dict[int, pd.DataFrame] = {}
            for sid, rows in rows_by_sid.items():
                df = pd.DataFrame(rows)
                if not df.empty and "timestamp" in df.columns:
                    df.set_index("timestamp", inplace=True)
                result[sid] = df
            return result

        # Main read/parse loop
        while True:
            # Ensure we have enough data for at least a header; if not, read more
            if len(buffer) < header_size:
                chunk = data_stream.read(chunk_size)
                if not chunk:
                    # End of stream
                    if self.verbose and buffer:
                        print(
                            f"End of stream with {len(buffer)} leftover bytes (incomplete header/payload)."
                        )
                    break
                buffer.extend(chunk)
                continue

            # We have at least a header
            header = bytes(buffer[:header_size])
            sid, size, time = self._parse_header(header)

            timestamp_s = time / 1e6

            if self.verbose:
                print(
                    f"Packet #{packet_idx}: SID={sid}, size={size}, time={timestamp_s:.6f}s "
                    f"(buffer_len={len(buffer)})"
                )

            # Basic sanity checks
            if sid not in self.parsers:
                if self.verbose:
                    print(f"Warning: No parser registered for SID={sid}. Attempting resync...")
                # new_offset = self._attempt_resync(bytes(buffer), 0, packet_idx, max_scan_bytes=max_resync_scan_bytes)
                # if new_offset is None:
                #     del buffer[:1]
                # else:
                #     del buffer[:new_offset]
                # continue

            if size <= 0:
                if self.verbose:
                    print(f"Invalid size={size} for SID={sid}. Attempting resync...")
                # new_offset = self._attempt_resync(bytes(buffer), 0, packet_idx, max_scan_bytes=max_resync_scan_bytes)
                # if new_offset is None:
                #     del buffer[:1]
                # else:
                #     del buffer[:new_offset]
                # continue

            if sid not in self.parsers:
                if self.verbose:
                    print(f"Unregistered SID={sid} encountered at packet #{packet_idx}. Skipping...")
                break
            else:
                parser = self.parsers[sid]
                # if hasattr(parser, "expected_size") and parser.expected_size is not None:
                #     if size != parser.expected_size:
                #         if self.verbose:
                #             print(
                #                 f"Size mismatch for SID={sid}: size={size}, expected={parser.expected_size}. "
                #                 "Attempting resync..."
                #             )
                        # new_offset = self._attempt_resync(bytes(buffer), 0, packet_idx, max_scan_bytes=max_resync_scan_bytes)
                        # if new_offset is None:
                        #     del buffer[:1]
                        # else:
                        #     del buffer[:new_offset]
                        # continue

                # Ensure the full payload is available; if not, read more
                needed = header_size + size
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

                payload = bytes(buffer[header_size:needed])
                try:
                    values_list = parser.parse(payload)
                    if self.verbose:
                        print(
                            f"Parsed packet #{packet_idx} (SID={sid}) successfully: {values_list}"
                        )
                except struct.error as e:
                    if self.verbose:
                        print(
                            f"struct.error while parsing payload at packet #{packet_idx} "
                            f"(SID={sid}, size={size}): {e}. Attempting resync..."
                        )
                    # Resync within the current buffer
                    new_offset = self._attempt_resync(bytes(buffer), 0, packet_idx, max_scan_bytes=max_resync_scan_bytes)
                    if new_offset is None:
                        del buffer[:1]
                    else:
                        del buffer[:new_offset]
                    continue
            
            for values in values_list:
                # Flatten nested group structure (group.component -> value)
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

                row = {
                    "timestamp": timestamp_s,
                    **flat_values,
                }
                rows_by_sid.setdefault(sid, []).append(row)

            # Consume this packet from the buffer
            del buffer[:needed]
            packet_idx += 1

        return flush_to_dataframes()

    def _parse_header(self, header: bytes) -> tuple[int, int, int]:
        """Parse a 10-byte packet header into (sid, size, time)."""
        sid, size, time = struct.unpack("<BBQ", header)
        return sid, size, time

    def _is_plausible_header(self, sid: int, size: int, remaining: int) -> bool:
        """Heuristic check whether a (sid, size) looks like a valid header.

        - SID must have a registered PayloadParser
        - size must be positive, not exceed remaining bytes, and match the
          expected payload size from the SensorScheme
        """
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
    ) -> int:
        """Try to recover from a corrupted header by scanning forward.

        Returns a new offset where a plausible header was found, or ``None``
        if no suitable header was located within ``max_scan_bytes``.
        """
        total_len = len(data)
        header_size = 10

        if self.verbose:
            print(
                f"Attempting resync after packet #{packet_idx} from offset {packet_start} "
                f"(scan up to {max_scan_bytes} bytes ahead)..."
            )

        for delta in range(1, max_scan_bytes + 1):
            candidate = packet_start + delta
            if candidate + header_size > total_len:
                break

            header = data[candidate : candidate + header_size]
            try:
                sid, size, time = self._parse_header(header)
            except struct.error:
                continue

            remaining = total_len - (candidate + header_size)
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
                f"giving up on this buffer."
            )
        return None
    
# MARK: - MicParser

class MicPayloadParser(PayloadParser):
    """Payload parser for microphone packets (int16 PCM samples)."""

    def __init__(self, sample_count: int, verbose: bool = False):
        self.sample_count = sample_count
        self.expected_size = sample_count * 2  # int16 samples
        self.verbose = verbose

    def parse(self, data: bytes) -> list[dict]:
        # Allow slight deviations in size but warn if unexpected
        if len(data) != self.expected_size and self.verbose:
            print(
                f"Mic payload size {len(data)} bytes does not match expected "
                f"{self.expected_size} bytes (sample_count={self.sample_count})."
            )

        if len(data) % 2 != 0 and self.verbose:
            print(
                f"Mic payload has odd size {len(data)}; last byte will be ignored."
            )

        n_samples = len(data) // 2
        format_str = f"<{n_samples}h"
        samples = struct.unpack_from(format_str, data, 0)
        return [{"samples": samples}]

# MARK: - SchemePayloadParser

class SchemePayloadParser(PayloadParser):
    def __init__(self, sensor_scheme: SensorScheme):
        self.sensor_scheme = sensor_scheme

        # Precompute expected payload size in bytes for a single packet
        size = 0
        for group in self.sensor_scheme.groups:
            for component in group.components:
                if component.data_type == ParseType.UINT8 or component.data_type == ParseType.INT8:
                    size += 1
                elif component.data_type in (ParseType.UINT16, ParseType.INT16):
                    size += 2
                elif component.data_type == ParseType.UINT32 or component.data_type == ParseType.INT32 or component.data_type == ParseType.FLOAT:
                    size += 4
                elif component.data_type == ParseType.DOUBLE:
                    size += 8
                else:
                    raise ValueError(f"Unsupported data type in scheme: {component.data_type}")
        self.expected_size = size
    
    def check_size(self, data: bytes) -> None:
        size = len(data)
        if size != self.expected_size and size % self.expected_size != 2:
            raise ValueError(
                f"Payload size {size} bytes does not match expected size "
                f"{self.expected_size} bytes for sensor '{self.sensor_scheme.name}'"
            )
        
    def is_buffered(self, data: bytes) -> bool:
        size = len(data)
        return size == self.expected_size

    def parse(self, data: bytes) -> list[dict]:
        self.check_size(data)
        if self.is_buffered(data):
            results = []
            # get the t_delta as an uint16 from the last two bytes
            t_delta = struct.unpack_from("<H", data, len(data) - 2)[0]
            n_packets = len(data) // self.expected_size
            for i in range(n_packets):
                packet_data = data[i * self.expected_size : (i + 1) * self.expected_size]
                parsed_packet = self.parse_packet(packet_data)
                # add t_delta to the parsed packet
                parsed_packet["t_delta"] = t_delta
                results.append(parsed_packet)
            return results
        else:
            return [self.parse_packet(data)]


    def parse_packet(self, data: bytes) -> dict:
        parsed_data = {}
        offset = 0

        for group in self.sensor_scheme.groups:
            group_data = {}
            for component in group.components:
                if component.data_type == "uint8":
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