"""Полное время взаимодействия устройства, периферии и облака (глава 12.1).

T_serial = S_u/B_u + R + T_c + S_d/B_d; ускорение модели помогает, только
если T_c лежит на критическом пути.
"""

from __future__ import annotations

from .roofline import _non_negative, _positive


def serial_seconds(
    upload_bytes: float,
    upload_bandwidth: float,
    rtt: float,
    compute_seconds: float,
    download_bytes: float,
    download_bandwidth: float,
) -> float:
    _non_negative("upload_bytes", upload_bytes)
    _positive("upload_bandwidth", upload_bandwidth)
    _non_negative("rtt", rtt)
    _non_negative("compute_seconds", compute_seconds)
    _non_negative("download_bytes", download_bytes)
    _positive("download_bandwidth", download_bandwidth)
    return (
        upload_bytes / upload_bandwidth
        + rtt
        + compute_seconds
        + download_bytes / download_bandwidth
    )


def compression_gain_seconds(
    original_bytes: float,
    compressed_bytes: float,
    upload_bandwidth: float,
    codec_seconds: float,
) -> float:
    """Выигрыш от сжатия перед отправкой; отрицателен, если кодек дороже сэкономленной передачи."""
    _non_negative("original_bytes", original_bytes)
    _non_negative("compressed_bytes", compressed_bytes)
    _positive("upload_bandwidth", upload_bandwidth)
    _non_negative("codec_seconds", codec_seconds)
    return (original_bytes - compressed_bytes) / upload_bandwidth - codec_seconds
