"""Полное время взаимодействия устройства, периферии и облака (глава 12.1).

T_serial = S_u/B_u + R + T_c + S_d/B_d; ускорение модели помогает, только
если T_c лежит на критическом пути.
"""

from __future__ import annotations

from .checks import require_non_negative, require_positive


def serial_seconds(
    upload_bytes: float,
    upload_bandwidth: float,
    rtt: float,
    compute_seconds: float,
    download_bytes: float,
    download_bandwidth: float,
) -> float:
    require_non_negative("upload_bytes", upload_bytes)
    require_positive("upload_bandwidth", upload_bandwidth)
    require_non_negative("rtt", rtt)
    require_non_negative("compute_seconds", compute_seconds)
    require_non_negative("download_bytes", download_bytes)
    require_positive("download_bandwidth", download_bandwidth)
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
    require_non_negative("original_bytes", original_bytes)
    require_non_negative("compressed_bytes", compressed_bytes)
    require_positive("upload_bandwidth", upload_bandwidth)
    require_non_negative("codec_seconds", codec_seconds)
    return (original_bytes - compressed_bytes) / upload_bandwidth - codec_seconds
