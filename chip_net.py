#!/usr/bin/env python3
"""UDP/TCP helpers -- ports of the MATLAB echoudp/echotcpip + fopen/fwrite/fclose.

The open/close helpers print what they do so the connection lifecycle can be
verified against the MATLAB scripts:
    HYLITE100F_StartUp_1:  fopen(u) ... fclose(u)
    cicle_test:            fopen(u), fopen(t) ... fclose(u), fclose(t)
                           then phase B reopens fresh connections.
"""

from __future__ import annotations

import socket


def register_packet(register: int, value: int, flag: int = 0) -> bytes:
    """9-byte register write: FF 80 51 01 00 00 <flag> <reg> <val>."""
    if not 0 <= register <= 0xFF or not 0 <= value <= 0xFF or not 0 <= flag <= 0xFF:
        raise ValueError("register, value and flag must all be in the range 0..255")
    return bytes((0xFF, 0x80, 0x51, 0x01, 0, 0, flag, register, value))


def gdac_packet(gdac_values: bytes) -> bytes:
    """GDAC burst: FF 80 51 46 00 00 00 10 <values> (4th byte = 70)."""
    return bytes((0xFF, 0x80, 0x51, 70, 0, 0, 0, 0x10)) + gdac_values


def open_udp(device_ip: str, udp_port: int) -> socket.socket:
    """MATLAB: u = udp(ip, port); fopen(u)."""
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.connect((device_ip, udp_port))
    print(f"Opened UDP connection to {device_ip}:{udp_port}")
    return udp_socket


def close_udp(
    udp_socket: socket.socket | None, device_ip: str, udp_port: int
) -> None:
    """MATLAB: fclose(u); echoudp('off')."""
    if udp_socket is not None:
        try:
            udp_socket.shutdown(socket.SHUT_RDWR)  # best effort on UDP
        except OSError:
            pass
        udp_socket.close()
        print(f"Closed UDP connection to {device_ip}:{udp_port}")


def open_tcp(device_ip: str, tcp_port: int, connect_timeout: float = 3.0) -> socket.socket:
    """MATLAB: t = tcpip(ip, port, 'Timeout', ...); fopen(t)."""
    tcp_socket = socket.create_connection((device_ip, tcp_port), timeout=connect_timeout)
    print(f"Opened TCP connection to {device_ip}:{tcp_port}")
    return tcp_socket


def close_tcp(
    tcp_socket: socket.socket | None, device_ip: str, tcp_port: int
) -> None:
    """MATLAB: fclose(t); echotcpip('off').  Sends FIN to the device."""
    if tcp_socket is not None:
        try:
            tcp_socket.shutdown(socket.SHUT_RDWR)  # send FIN
        except OSError:
            pass
        tcp_socket.close()
        print(f"Closed TCP connection to {device_ip}:{tcp_port}")


def send(udp_socket: socket.socket, packet: bytes) -> None:
    """fwrite(u, packet) over the connected UDP socket."""
    sent = udp_socket.send(packet)
    if sent != len(packet):
        raise OSError(f"UDP packet was only partly sent ({sent}/{len(packet)} bytes)")


def read_tcp(
    tcp_socket: socket.socket,
    idle_timeout: float,
    first_timeout: float = 5.0,
) -> bytes:
    """Read until an idle gap (port of MATLAB fread with Timeout).

    The first byte may take a while after the DDR read trigger, so it is
    awaited for up to `first_timeout`; afterwards the frame is drained
    until no data arrives for `idle_timeout`.
    """
    tcp_socket.settimeout(first_timeout)
    try:
        block = tcp_socket.recv(1_048_576)
    except socket.timeout:
        raise TimeoutError(
            f"no TCP data arrived within {first_timeout:g} seconds; "
            f"the device may need more time to start streaming "
            f"(increase --first-timeout)"
        ) from None
    if not block:
        raise TimeoutError("TCP connection closed before any data arrived")

    chunks = [block]
    tcp_socket.settimeout(idle_timeout)
    while True:
        try:
            block = tcp_socket.recv(1_048_576)
        except socket.timeout:
            break
        if not block:
            break
        chunks.append(block)
    return b"".join(chunks)
