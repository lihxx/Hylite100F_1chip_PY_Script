#!/usr/bin/env python3
"""Read and decode one HYLITE200F frame.

This is the Python counterpart of READ_X.m.  It keeps the MATLAB register
sequence and also includes the processing performed by D_P1.m and
arr1_Mapping.m.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

try:
    import numpy as np
except ImportError:  # Keep --help available before project dependencies are installed.
    np = None  # type: ignore[assignment]


DEFAULT_DEVICE_IP = "192.168.10.16"
DEFAULT_UDP_PORT = 4660
DEFAULT_TCP_PORT = 24
INPUT_BUFFER_SIZE = 300_894
FRAME_BITS_PER_LANE = 128 * 2 * 13 * 32  # 106496 bits
HEADER_SEARCH_BITS = 800
LANE_COUNT = 32
PIXELS_PER_LANE = 256
BITS_PER_PIXEL = 13
LFSR_BITS = 11

# The synchronization pattern from D_P1.m: 8 x FF, 8 x 00, 8 x FF,
# 8 x 00, followed by 12 x FF.
SYNC_BYTES = bytes([0xFF] * 8 + [0x00] * 8 + [0xFF] * 8 + [0x00] * 8 + [0xFF] * 12)


def require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "NumPy is required. Install dependencies from the project directory "
            "with: python -m pip install -r requirements.txt"
        )


def register_packet(register: int, value: int) -> bytes:
    """Build the device's 9-byte UDP register-write command."""
    if not 0 <= register <= 0xFF or not 0 <= value <= 0xFF:
        raise ValueError("register and value must be in the range 0..255")
    return bytes((0xFF, 0x80, 0x51, 0x01, 0, 0, 0, register, value))


def send_register(udp_socket: socket.socket, register: int, value: int) -> None:
    packet = register_packet(register, value)
    sent = udp_socket.send(packet)
    if sent != len(packet):
        raise OSError(f"UDP packet was only partly sent ({sent}/{len(packet)} bytes)")


def open_udp_socket(device_ip: str, udp_port: int, local_ip: str | None) -> socket.socket:
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if local_ip:
        udp_socket.bind((local_ip, 0))
    udp_socket.connect((device_ip, udp_port))
    return udp_socket


def open_tcp_socket(
    device_ip: str,
    tcp_port: int,
    local_ip: str | None,
    connect_timeout: float,
) -> socket.socket:
    source_address = (local_ip, 0) if local_ip else None
    tcp_socket = socket.create_connection(
        (device_ip, tcp_port),
        timeout=connect_timeout,
        source_address=source_address,
    )
    tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, INPUT_BUFFER_SIZE)
    return tcp_socket


def receive_until_idle(
    tcp_socket: socket.socket,
    *,
    maximum_size: int,
    first_byte_timeout: float,
    idle_timeout: float,
) -> bytes:
    """Wait for the first TCP byte, then stop after an in-frame idle period."""
    tcp_socket.settimeout(first_byte_timeout)
    received = bytearray()

    while len(received) < maximum_size:
        try:
            block = tcp_socket.recv(min(65_536, maximum_size - len(received)))
        except socket.timeout:
            if received:
                break
            raise TimeoutError(
                f"No TCP data arrived within {first_byte_timeout:g} seconds"
            ) from None

        if not block:
            break
        received.extend(block)
        # Once a frame has started, MATLAB's 0.1-second Timeout is used to
        # detect its end.  The first byte can legitimately take much longer.
        tcp_socket.settimeout(idle_timeout)

    if not received:
        raise RuntimeError("The TCP connection closed without returning any data")
    return bytes(received)


def acquire_frame(
    args: argparse.Namespace, lookup: np.ndarray
) -> tuple[bytes, np.ndarray, np.ndarray, tuple[int, int]]:
    """Execute READ_X.m, including D_P1 decoding before the final reset."""
    with open_udp_socket(args.device_ip, args.udp_port, args.local_ip) as udp_socket:
        with open_tcp_socket(
            args.device_ip,
            args.tcp_port,
            args.local_ip,
            args.connect_timeout,
        ) as tcp_socket:
            # IDELAY
            time.sleep(1.0)
            send_register(udp_socket, 0x70, 0x01)  # tap0_0
            time.sleep(0.1)
            send_register(udp_socket, 0x80, 0x01)  # tap0_1
            time.sleep(0.1)
            send_register(udp_socket, 0x90, 0x00)  # LD
            time.sleep(0.1)
            send_register(udp_socket, 0x90, 0x00)  # LD
            time.sleep(0.1)

            # Data readout
            send_register(udp_socket, 0x06, 0x00)  # cs = 0
            time.sleep(0.001)
            send_register(udp_socket, 0x0A, 0x01)  # gate = 1
            time.sleep(0.001)
            send_register(udp_socket, 0x0C, 0xFF)  # clkref_en
            time.sleep(0.001)
            send_register(udp_socket, 0x0B, 0xFF)  # fifo_rst = 1
            time.sleep(0.1)
            send_register(udp_socket, 0x0B, 0x00)  # fifo_rst = 0
            time.sleep(0.1)
            send_register(udp_socket, 0x09, 0x00)  # rstb_chip = 0
            time.sleep(0.1)
            send_register(udp_socket, 0x09, 0xFF)  # rstb_chip = 1
            time.sleep(0.1)

            # DDR read sequence
            send_register(udp_socket, 0x0F, 0x01)
            time.sleep(1.0)
            send_register(udp_socket, 0x0F, 0x00)  # DDR reset

            raw = receive_until_idle(
                tcp_socket,
                maximum_size=INPUT_BUFFER_SIZE,
                first_byte_timeout=args.first_byte_timeout,
                idle_timeout=args.idle_timeout,
            )

            # In the current READ_X.m, D_P1 runs before the final 0x0E pulse.
            time.sleep(1.0)
            image, gain_image, positions = decode_frame(raw, lookup)

            send_register(udp_socket, 0x0E, 0x01)
            time.sleep(0.01)
            send_register(udp_socket, 0x0E, 0x00)
            time.sleep(0.01)
            return raw, image, gain_image, positions


def _cell_number(cell: ElementTree.Element, namespace: dict[str, str]) -> int:
    value = cell.find("x:v", namespace)
    if value is None or value.text is None:
        raise ValueError(f"Empty LUT cell {cell.attrib.get('r', '?')}")
    return int(float(value.text))


def load_lfsr_lookup(workbook_path: Path) -> np.ndarray:
    """Load the two numeric columns of LFSR11_dec.xlsx without extra packages."""
    require_numpy()
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    try:
        with ZipFile(workbook_path) as workbook:
            worksheet = ElementTree.fromstring(
                workbook.read("xl/worksheets/sheet1.xml")
            )
    except FileNotFoundError:
        raise RuntimeError(f"LFSR lookup file not found: {workbook_path}") from None
    except (BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise RuntimeError(f"Cannot read LFSR lookup file {workbook_path}: {exc}") from exc

    lookup = np.full(1 << LFSR_BITS, -1, dtype=np.int32)
    for row in worksheet.findall(".//x:sheetData/x:row", namespace):
        cells = row.findall("x:c", namespace)
        if len(cells) < 2:
            continue
        count = _cell_number(cells[0], namespace)
        lfsr_value = _cell_number(cells[1], namespace)
        if not 0 <= lfsr_value < lookup.size:
            raise RuntimeError(f"Invalid 11-bit LFSR value in workbook: {lfsr_value}")
        lookup[lfsr_value] = count

    missing = np.flatnonzero(lookup < 0)
    if missing.size:
        raise RuntimeError(
            f"LFSR lookup is incomplete; first missing value is {int(missing[0])}"
        )
    return lookup


def find_sync(bits: np.ndarray, lane_name: str) -> int:
    """Return the zero-based sync position, matching MATLAB's first 800 starts."""
    require_numpy()
    sync_bits = np.unpackbits(np.frombuffer(SYNC_BYTES, dtype=np.uint8))
    last_start = min(HEADER_SEARCH_BITS, bits.size - sync_bits.size + 1)
    for start in range(max(0, last_start)):
        if np.array_equal(bits[start : start + sync_bits.size], sync_bits):
            return start
    raise RuntimeError(
        f"Synchronization header was not found in the first {HEADER_SEARCH_BITS} "
        f"bits of {lane_name}"
    )


def arr1_mapping(arr_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate arr1_Mapping.m from a 64 x 256 array to a 128 x 128 image."""
    require_numpy()
    expected_shape = (64, 256)
    if arr_data.shape != expected_shape:
        raise ValueError(f"Mapping input must have shape {expected_shape}; got {arr_data.shape}")

    arr_image = np.zeros((128, 128), dtype=arr_data.dtype)
    # arr1_Mapping.m declares this second output but never writes to it.
    arr_img_gain = np.zeros((128, 128), dtype=np.float64)

    for lane in range(32):
        arr_image[:, 2 * lane] = arr_data[lane, :128][::-1]
        arr_image[:, 2 * lane + 1] = arr_data[lane, 128:]

    for lane in range(32, 64):
        arr_image[:, 2 * lane] = arr_data[lane, :128]
        arr_image[:, 2 * lane + 1] = arr_data[lane, 128:][::-1]

    arr_image[:, 64:] = arr_image[::-1, 64:].copy()

    for start in range(0, 128, 32):
        arr_image[:, start : start + 32] = arr_image[:, start : start + 32][
            :, ::-1
        ].copy()

    for start in range(0, 64, 4):
        arr_image[:, start : start + 4] = arr_image[:, start : start + 4][
            :, ::-1
        ].copy()
    for start in range(0, 64, 2):
        arr_image[:, start : start + 2] = arr_image[:, start : start + 2][
            :, ::-1
        ].copy()
    for start in range(64, 128, 4):
        arr_image[:, start : start + 4] = arr_image[:, start : start + 4][
            :, ::-1
        ].copy()

    return arr_image, arr_img_gain


def decode_frame(
    raw: bytes,
    lookup: np.ndarray,
    positions: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Decode a frame, optionally using known (lane 1, lane 0) bit positions."""
    require_numpy()
    raw_array = np.frombuffer(raw, dtype=np.uint8)
    if raw_array.size < len(SYNC_BYTES) * 2:
        raise RuntimeError(f"TCP data is too short to contain both headers: {len(raw)} bytes")

    byte_bits = np.unpackbits(raw_array).reshape(-1, 8)
    lane1_bits = byte_bits[0::2].reshape(-1)
    lane0_bits = byte_bits[1::2].reshape(-1)
    if positions is None:
        lane1_start = find_sync(lane1_bits, "lane 1")
        lane0_start = find_sync(lane0_bits, "lane 0")
    else:
        lane1_start, lane0_start = positions
        if lane1_start < 0 or lane0_start < 0:
            raise ValueError("sync positions must be zero or greater")

    if lane1_start + FRAME_BITS_PER_LANE > lane1_bits.size:
        raise RuntimeError(
            f"Lane 1 is short: {lane1_bits.size - lane1_start} frame bits available, "
            f"{FRAME_BITS_PER_LANE} required"
        )
    if lane0_start + FRAME_BITS_PER_LANE > lane0_bits.size:
        raise RuntimeError(
            f"Lane 0 is short: {lane0_bits.size - lane0_start} frame bits available, "
            f"{FRAME_BITS_PER_LANE} required"
        )

    lane1_frame = lane1_bits[lane1_start : lane1_start + FRAME_BITS_PER_LANE]
    lane0_frame = lane0_bits[lane0_start : lane0_start + FRAME_BITS_PER_LANE]

    lane1_fields = lane1_frame.reshape(-1, LANE_COUNT).T.reshape(
        LANE_COUNT, PIXELS_PER_LANE, BITS_PER_PIXEL
    )
    lane0_fields = lane0_frame.reshape(-1, LANE_COUNT).T.reshape(
        LANE_COUNT, PIXELS_PER_LANE, BITS_PER_PIXEL
    )

    lfsr_weights = 1 << np.arange(LFSR_BITS - 1, -1, -1)
    lane0_codes = lane0_fields[:, :, :LFSR_BITS] @ lfsr_weights
    lane1_codes = lane1_fields[:, :, :LFSR_BITS] @ lfsr_weights
    data0 = lookup[lane0_codes]
    data1 = lookup[lane1_codes]

    arr_data = np.vstack((data0, data1))
    image, gain_image = arr1_mapping(arr_data)
    return image, gain_image, (lane1_start, lane0_start)


def show_image(image: np.ndarray, save_path: Path | None, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    figure, axis = plt.subplots()
    plot = axis.imshow(image, origin="upper", aspect="equal")
    axis.set_title("Decoded frame")
    figure.colorbar(plot, ax=axis)
    figure.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved image: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def port_number(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be in the range 1..65535")
    return value


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Control HYLITE200F, receive one frame, decode it, and display it."
    )
    parser.add_argument("--device-ip", default=DEFAULT_DEVICE_IP, help="device IP")
    parser.add_argument(
        "--local-ip", help="local 10G NIC IP, useful when the PC has multiple NICs"
    )
    parser.add_argument("--udp-port", type=port_number, default=DEFAULT_UDP_PORT)
    parser.add_argument("--tcp-port", type=port_number, default=DEFAULT_TCP_PORT)
    parser.add_argument(
        "--connect-timeout", type=positive_float, default=3.0, help="seconds (default: 3)"
    )
    parser.add_argument(
        "--first-byte-timeout",
        type=positive_float,
        default=3.0,
        help="maximum wait for frame data to start, in seconds (default: 3)",
    )
    parser.add_argument(
        "--idle-timeout",
        type=positive_float,
        default=0.1,
        help="end-of-frame TCP idle timeout in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--lut",
        type=Path,
        default=script_dir / "LFSR11_dec.xlsx",
        help="LFSR lookup workbook",
    )
    parser.add_argument(
        "--raw-input",
        type=Path,
        help="decode an existing RAW1 file instead of connecting to hardware",
    )
    parser.add_argument("--raw-output", type=Path, help="save received RAW1 bytes")
    parser.add_argument("--save-image", type=Path, help="save the decoded image")
    parser.add_argument("--no-show", action="store_true", help="do not open a plot window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        lookup = load_lfsr_lookup(args.lut)
        if args.raw_input is not None:
            raw = args.raw_input.read_bytes()
            print(f"Loaded {len(raw):,} bytes from {args.raw_input}")
            image, _gain_image, positions = decode_frame(raw, lookup)
        else:
            raw, image, _gain_image, positions = acquire_frame(args, lookup)
            print(f"Received {len(raw):,} TCP bytes")

        if args.raw_output is not None:
            args.raw_output.parent.mkdir(parents=True, exist_ok=True)
            args.raw_output.write_bytes(raw)
            print(f"Saved raw frame: {args.raw_output}")

        print(f"Sync positions (lane 1, lane 0): {positions[0]}, {positions[1]} bits")
        show_image(image, args.save_image, show=not args.no_show)
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
