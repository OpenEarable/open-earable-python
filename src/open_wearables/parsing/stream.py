import struct
from typing import BinaryIO, Dict, List, Optional

import pandas as pd

from open_wearables.schema import SensorScheme

from .audio import MicPacket, mic_samples_to_stereo
from .base import ParseResult, PayloadParser
from .payload_parsers import MicPayloadParser, SchemePayloadParser


class Parser:
    def __init__(self, parsers: dict[int, PayloadParser], verbose: bool = False):
        self.parsers = parsers
        self.verbose = verbose

    @classmethod
    def from_sensor_schemes(
        cls,
        sensor_schemes: dict[int, SensorScheme],
        verbose: bool = False,
    ) -> "Parser":
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
    ) -> ParseResult:
        rows_by_sid: dict[int, list[dict]] = {}

        header_size = 10
        buffer = bytearray()
        packet_idx = 0
        mic_samples: List[int] = []
        mic_packets: List[MicPacket] = []

        def flush_to_dataframes() -> Dict[int, pd.DataFrame]:
            result: Dict[int, pd.DataFrame] = {}
            for sid, rows in rows_by_sid.items():
                df = pd.DataFrame(rows)
                if not df.empty and "timestamp" in df.columns:
                    df.set_index("timestamp", inplace=True)
                result[sid] = df
            return result

        while True:
            if len(buffer) < header_size:
                chunk = data_stream.read(chunk_size)
                if not chunk:
                    if self.verbose and buffer:
                        print(
                            f"End of stream with {len(buffer)} leftover bytes (incomplete header/payload)."
                        )
                    break
                buffer.extend(chunk)
                continue

            header = bytes(buffer[:header_size])
            sid, size, time = self._parse_header(header)
            timestamp_s = time / 1e6

            if self.verbose:
                print(
                    f"Packet #{packet_idx}: SID={sid}, size={size}, time={timestamp_s:.6f}s "
                    f"(buffer_len={len(buffer)})"
                )

            if sid not in self.parsers:
                if self.verbose:
                    print(f"Warning: No parser registered for SID={sid}. Attempting resync...")
                new_offset = self._attempt_resync(
                    bytes(buffer),
                    0,
                    packet_idx,
                    max_scan_bytes=max_resync_scan_bytes,
                )
                if new_offset is None:
                    del buffer[:1]
                else:
                    del buffer[:new_offset]
                continue

            if size <= 0:
                if self.verbose:
                    print(f"Invalid size={size} for SID={sid}. Attempting resync...")
                new_offset = self._attempt_resync(
                    bytes(buffer),
                    0,
                    packet_idx,
                    max_scan_bytes=max_resync_scan_bytes,
                )
                if new_offset is None:
                    del buffer[:1]
                else:
                    del buffer[:new_offset]
                continue

            parser = self.parsers[sid]
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
                if isinstance(parser, MicPayloadParser):
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
                if self.verbose:
                    if isinstance(parser, MicPayloadParser):
                        print(
                            f"Parsed mic packet #{packet_idx} (SID={sid}) successfully: "
                            f"{len(values_list[0].get('samples', [])) if values_list else 0} samples"
                        )
                    else:
                        print(
                            f"Parsed packet #{packet_idx} (SID={sid}) successfully: {values_list}"
                        )
            except Exception as exc:
                if self.verbose:
                    print(
                        f"struct.error while parsing payload at packet #{packet_idx} "
                        f"(SID={sid}, size={size}): {exc}. Attempting resync..."
                    )
                new_offset = self._attempt_resync(
                    bytes(buffer),
                    0,
                    packet_idx,
                    max_scan_bytes=max_resync_scan_bytes,
                )
                if new_offset is None:
                    del buffer[:1]
                else:
                    del buffer[:new_offset]
                continue

            if parser.should_build_df():
                for values in values_list:
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

            del buffer[:needed]
            packet_idx += 1

        sensor_dfs = flush_to_dataframes()
        audio_stereo = mic_samples_to_stereo(mic_samples)
        return ParseResult(
            sensor_dfs=sensor_dfs,
            mic_samples=mic_samples,
            mic_packets=mic_packets,
            audio_stereo=audio_stereo,
        )

    def _parse_header(self, header: bytes) -> tuple[int, int, int]:
        sid, size, time = struct.unpack("<BBQ", header)
        return sid, size, time

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
                "giving up on this buffer."
            )
        return None
