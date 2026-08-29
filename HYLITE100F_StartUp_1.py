#!/usr/bin/env python3
"""Python port of one_chip/HYLITE100F_StartUp_1.m -- Chip GDAC configuration.

Reads the first column of the GDAC register .xlsx and runs the complete
UDP register sequence exactly like the MATLAB script:
    clkref_en -> rstb_chip_t (low/high) -> cs=0/config_mod=0/gate=0
    -> GDAC burst -> rstb_matlab -> cs=1/gate=1
The device IP (--ip) and the config file (--file) are configurable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chip_net import (  # noqa: E402
    close_udp,
    gdac_packet,
    open_udp,
    register_packet,
    send,
)
from xlsx_util import first_column_int  # noqa: E402

DEFAULT_DEVICE_IP = "192.168.10.16"
DEFAULT_UDP_PORT = 4660
DEFAULT_FILE = "../ConfigFiles/GDAC_regVcali1u_LowSpeed_128GclK.xlsx"

# The MATLAB script points at ../ConfigFiles/...; on this machine the file
# lives elsewhere, so several candidates are tried in order.
FILE_CANDIDATES = (
    DEFAULT_FILE,
    "../ConfigFiles/GDAC_regVcali1u_LowSpeed_128Gclk.xlsx",
    "GDAC_regVcali1u_LowSpeed_128Gclk.xlsx",
    "../GDAC_regVcali1u_LowSpeed_128Gclk.xlsx",
)


def resolve_config_file(given: Path | None) -> Path:
    """Return an existing GDAC .xlsx: --file if given, else first candidate."""
    if given is not None:
        if not given.exists():
            raise FileNotFoundError(f"GDAC config file not found: {given}")
        return given
    here = Path(__file__).resolve().parent
    for candidate in FILE_CANDIDATES:
        path = Path(candidate)
        if path.exists() or (not path.is_absolute() and (here / path).exists()):
            return path if path.exists() else here / path
    raise FileNotFoundError(
        "GDAC config .xlsx not found; pass --file. Tried: "
        + ", ".join(FILE_CANDIDATES)
    )


def run_startup(device_ip: str, udp_port: int, gdac_values: list[int]) -> None:
    """Send the full startup register sequence over UDP.

    MATLAB: u = udp(ip, 4660); fopen(u); ...register writes...; fclose(u).
    The UDP connection is explicitly closed at the end so cicle_test can
    open fresh connections afterwards.
    """
    for value in gdac_values:
        if not 0 <= value <= 0xFF:
            raise ValueError(f"GDAC value {value} is out of the 0..255 range")

    udp_socket = open_udp(device_ip, udp_port)  # fopen(u)
    try:
        def write(register: int, value: int, delay: float) -> None:
            send(udp_socket, register_packet(register, value))
            time.sleep(delay)

        time.sleep(0.001)  # MATLAB pause(0.001) after fopen
        write(0x0C, 0xFF, 0.001)  # clkref_en
        write(0x08, 0x00, 0.001)  # rstb_chip_t low
        write(0x08, 0xFF, 0.001)  # rstb_chip_t high
        write(0x06, 0x00, 0.001)  # cs = 0
        write(0x05, 0x00, 0.001)  # config_mod = 0
        write(0x0A, 0x00, 0.001)  # gate = 0
        send(udp_socket, gdac_packet(bytes(gdac_values)))  # GDAC burst
        time.sleep(1.0)
        write(0x00, 0xFF, 0.1)  # rstb_matlab
        write(0x01, 0xFF, 0.1)  # rstb_matlab
        write(0x06, 0x01, 0.001)  # cs = 1
        write(0x0A, 0xFF, 0.001)  # gate = 1
    finally:
        close_udp(udp_socket, device_ip, udp_port)  # fclose(u); echoudp('off')


def port_number(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be in the range 1..65535")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chip GDAC configuration for HYLITE100F: read the first column of "
            "the GDAC .xlsx and run the startup register sequence over UDP."
        )
    )
    parser.add_argument(
        "--ip",
        default=DEFAULT_DEVICE_IP,
        help=f"device IP address (default: {DEFAULT_DEVICE_IP})",
    )
    parser.add_argument(
        "--udp-port",
        type=port_number,
        default=DEFAULT_UDP_PORT,
        help=f"UDP port (default: {DEFAULT_UDP_PORT})",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="GDAC configuration .xlsx (default: search known locations)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = resolve_config_file(args.file)
        values = first_column_int(config)
        if not values:
            raise ValueError(f"no numeric values found in column A of {config}")
        print(f"Loaded {len(values):,} GDAC values from {config} "
              f"(min {min(values)}, max {max(values)})")
        print(f"Sending startup sequence to {args.ip}:{args.udp_port}")
        run_startup(args.ip, args.udp_port, values)
        print("HYLITE100F startup sequence finished")
        return 0
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
