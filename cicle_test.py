#!/usr/bin/env python3
"""Python translation of cicle_test.m and DA_1.m.

For each k value, configure the device, acquire a digital reference frame,
then acquire and save the analog frame using the reference sync positions.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import READ_X as read_x


DEFAULT_K_VALUES = (200,)
DIGITAL_BUFFER_SIZE = 300_894
ANALOG_BUFFER_SIZE = 276_480


def k_value(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 511:
        raise argparse.ArgumentTypeError("k must be in the range 0..511")
    return value


def register_packet(page: int, register: int, value: int) -> bytes:
    if not 0 <= page <= 0xFF:
        raise ValueError("page must be in the range 0..255")
    if not 0 <= register <= 0xFF or not 0 <= value <= 0xFF:
        raise ValueError("register and value must be in the range 0..255")
    return bytes((0xFF, 0x80, 0x51, 0x01, 0, 0, page, register, value))


def send_register(
    udp_socket: socket.socket, register: int, value: int, *, page: int = 0
) -> None:
    packet = register_packet(page, register, value)
    sent = udp_socket.send(packet)
    if sent != len(packet):
        raise OSError(f"UDP packet was only partly sent ({sent}/{len(packet)} bytes)")


def open_tcp_socket(args: argparse.Namespace, buffer_size: int) -> socket.socket:
    source_address = (args.local_ip, 0) if args.local_ip else None
    tcp_socket = socket.create_connection(
        (args.device_ip, args.tcp_port),
        timeout=args.connect_timeout,
        source_address=source_address,
    )
    tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    return tcp_socket


def acquire_digital_frame(args: argparse.Namespace, k: int) -> bytes:
    """Run the first acquisition section of cicle_test.m."""
    high = 0 if k <= 255 else 1
    low = k if k <= 255 else k - 256

    time.sleep(1.0)
    with read_x.open_udp_socket(args.device_ip, args.udp_port, args.local_ip) as udp:
        with open_tcp_socket(args, DIGITAL_BUFFER_SIZE) as tcp:
            send_register(udp, 0x10, high, page=1)
            time.sleep(0.1)
            send_register(udp, 0x11, low, page=1)
            time.sleep(0.1)
            send_register(udp, 0x12, high, page=1)
            time.sleep(0.1)
            send_register(udp, 0x13, low, page=1)
            time.sleep(0.1)
            send_register(udp, 0x1E, 1, page=1)
            time.sleep(0.1)
            send_register(udp, 0x1E, 0, page=1)
            time.sleep(0.1)
            send_register(udp, 0x1F, 1, page=1)
            time.sleep(0.1)
            send_register(udp, 0x1F, 0, page=1)

            time.sleep(1.0)
            send_register(udp, 0x70, 1)
            time.sleep(0.1)
            send_register(udp, 0x80, 1)
            time.sleep(0.1)
            send_register(udp, 0x90, 0)
            time.sleep(0.1)
            send_register(udp, 0x90, 0)
            time.sleep(0.1)

            send_register(udp, 0x06, 0)
            time.sleep(0.001)
            send_register(udp, 0x0A, 1)
            time.sleep(0.001)
            send_register(udp, 0x0C, 255)
            time.sleep(0.001)
            send_register(udp, 0x0B, 255)
            time.sleep(0.1)
            send_register(udp, 0x0B, 0)
            time.sleep(0.1)
            send_register(udp, 0x09, 0)
            time.sleep(0.1)
            send_register(udp, 0x09, 255)
            time.sleep(0.1)

            send_register(udp, 0x0F, 1)
            time.sleep(0.1)
            send_register(udp, 0x0F, 0)

            raw = read_x.receive_until_idle(
                tcp,
                maximum_size=DIGITAL_BUFFER_SIZE,
                first_byte_timeout=args.first_byte_timeout,
                idle_timeout=args.idle_timeout,
            )
            time.sleep(1.0)
            send_register(udp, 0x0E, 1)
            time.sleep(0.01)
            send_register(udp, 0x0E, 0)
            time.sleep(0.01)
            return raw


def decode_frame_at_positions(raw: bytes, lookup, positions: tuple[int, int]):
    """Decode RAW2 at RAW1's sync positions, matching DA_1.m."""
    read_x.require_numpy()
    np = read_x.np
    raw_array = np.frombuffer(raw, dtype=np.uint8)
    byte_bits = np.unpackbits(raw_array).reshape(-1, 8)
    lane1_bits = byte_bits[0::2].reshape(-1)
    lane0_bits = byte_bits[1::2].reshape(-1)
    lane1_start, lane0_start = positions

    if lane1_start < 0 or lane0_start < 0:
        raise ValueError("sync positions must be zero or greater")
    if lane1_start + read_x.FRAME_BITS_PER_LANE > lane1_bits.size:
        raise RuntimeError(
            f"Analog lane 1 is short: {lane1_bits.size - lane1_start} bits "
            f"available, {read_x.FRAME_BITS_PER_LANE} required"
        )
    if lane0_start + read_x.FRAME_BITS_PER_LANE > lane0_bits.size:
        raise RuntimeError(
            f"Analog lane 0 is short: {lane0_bits.size - lane0_start} bits "
            f"available, {read_x.FRAME_BITS_PER_LANE} required"
        )

    lane1_frame = lane1_bits[
        lane1_start : lane1_start + read_x.FRAME_BITS_PER_LANE
    ]
    lane0_frame = lane0_bits[
        lane0_start : lane0_start + read_x.FRAME_BITS_PER_LANE
    ]
    lane1_fields = lane1_frame.reshape(-1, read_x.LANE_COUNT).T.reshape(
        read_x.LANE_COUNT, read_x.PIXELS_PER_LANE, read_x.BITS_PER_PIXEL
    )
    lane0_fields = lane0_frame.reshape(-1, read_x.LANE_COUNT).T.reshape(
        read_x.LANE_COUNT, read_x.PIXELS_PER_LANE, read_x.BITS_PER_PIXEL
    )

    weights = 1 << np.arange(read_x.LFSR_BITS - 1, -1, -1)
    lane0_codes = lane0_fields[:, :, : read_x.LFSR_BITS] @ weights
    lane1_codes = lane1_fields[:, :, : read_x.LFSR_BITS] @ weights
    arr_data = np.vstack((lookup[lane0_codes], lookup[lane1_codes]))
    image, gain_image = read_x.arr1_mapping(arr_data)
    return image, gain_image


def acquire_analog_frame(
    args: argparse.Namespace,
    k: int,
    lookup,
    positions: tuple[int, int],
) -> tuple[bytes, object]:
    """Run the analog acquisition and DA_1 processing sections."""
    with read_x.open_udp_socket(args.device_ip, args.udp_port, args.local_ip) as udp:
        send_register(udp, 0x06, 255)
        time.sleep(0.001)
        send_register(udp, 0x0A, 0)
        time.sleep(0.001)

        with open_tcp_socket(args, ANALOG_BUFFER_SIZE) as tcp:
            send_register(udp, 0x0B, 255)
            time.sleep(0.1)
            send_register(udp, 0x0B, 0)
            time.sleep(0.1)
            send_register(udp, 0x0F, 1)
            time.sleep(1.0)
            send_register(udp, 0x0F, 0)

            raw = read_x.receive_until_idle(
                tcp,
                maximum_size=ANALOG_BUFFER_SIZE,
                first_byte_timeout=args.first_byte_timeout,
                idle_timeout=args.idle_timeout,
            )
            image, _gain_image = decode_frame_at_positions(raw, lookup, positions)

            output_path = args.output_dir / f"15Yimage_{k:02d}.jpg"
            save_frame_image(image, output_path, vmin=0, vmax=650)

            send_register(udp, 0x0E, 1)
            time.sleep(0.01)
            send_register(udp, 0x0E, 0)
            return raw, image


def save_frame_image(
    image, output_path: Path, *, vmin: int | None = None, vmax: int | None = None
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    plot = axis.imshow(image, origin="upper", aspect="equal", vmin=vmin, vmax=vmax)
    figure.colorbar(plot, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image: {output_path}")


def show_last_image(image, k: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required for plotting") from exc

    figure, axis = plt.subplots()
    plot = axis.imshow(image, origin="upper", aspect="equal", vmin=0, vmax=650)
    axis.set_title(f"Analog frame, k = {k}")
    figure.colorbar(plot, ax=axis)
    figure.tight_layout()
    plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python equivalent of cicle_test.m")
    parser.add_argument(
        "--ip",
        "--device-ip",
        dest="device_ip",
        default=read_x.DEFAULT_DEVICE_IP,
        help=f"device IP (default: {read_x.DEFAULT_DEVICE_IP})",
    )
    parser.add_argument(
        "-k",
        "--k",
        dest="k_values",
        type=k_value,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        metavar="K",
        help="one or more k values in the range 0..511 (default: 200)",
    )
    parser.add_argument("--local-ip", help="local 10G NIC IP")
    parser.add_argument(
        "--udp-port", type=read_x.port_number, default=read_x.DEFAULT_UDP_PORT
    )
    parser.add_argument(
        "--tcp-port", type=read_x.port_number, default=read_x.DEFAULT_TCP_PORT
    )
    parser.add_argument(
        "--connect-timeout", type=read_x.positive_float, default=3.0
    )
    parser.add_argument(
        "--first-byte-timeout", type=read_x.positive_float, default=3.0
    )
    parser.add_argument("--idle-timeout", type=read_x.positive_float, default=0.1)
    parser.add_argument(
        "--lut",
        type=Path,
        default=Path(__file__).resolve().parent / "LFSR11_dec.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="image output directory (default: current directory)",
    )
    parser.add_argument("--no-show", action="store_true", help="do not show final image")
    return parser.parse_args(argv)


def run(args: argparse.Namespace):
    lookup = read_x.load_lfsr_lookup(args.lut)
    last_image = None
    last_k = None

    for index, k in enumerate(args.k_values, start=1):
        print(f"[{index}/{len(args.k_values)}] Starting k={k} on {args.device_ip}")
        raw1 = acquire_digital_frame(args, k)

        # D_P_TC.m: decode RAW1, return both sync positions, and save the image.
        digital_image, _gain, positions = read_x.decode_frame(raw1, lookup)
        digital_path = args.output_dir / f"15YYimage_{k:02d}.jpg"
        save_frame_image(digital_image, digital_path)
        print(
            f"k={k}: digital frame {len(raw1):,} bytes; "
            f"sync positions (lane 1, lane 0): {positions[0]}, {positions[1]}"
        )
        time.sleep(1.0)

        raw2, last_image = acquire_analog_frame(args, k, lookup, positions)
        print(f"k={k}: analog frame {len(raw2):,} bytes")
        last_k = k
        time.sleep(1.0)

    return last_image, last_k


def main() -> int:
    args = parse_args()
    try:
        image, last_k = run(args)
        if not args.no_show:
            show_last_image(image, last_k)
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
