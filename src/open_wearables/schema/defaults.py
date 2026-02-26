from typing import Dict, Mapping

from .types import ParseType, SensorScheme, group


def build_default_sensor_schemes(sensor_sid: Mapping[str, int]) -> Dict[int, SensorScheme]:
    """Build default non-microphone sensor schemes keyed by SID."""
    return {
        sensor_sid["imu"]: SensorScheme(
            name="imu",
            sid=sensor_sid["imu"],
            groups=[
                group(
                    "acc",
                    [
                        ("x", ParseType.FLOAT),
                        ("y", ParseType.FLOAT),
                        ("z", ParseType.FLOAT),
                    ],
                ),
                group(
                    "gyro",
                    [
                        ("x", ParseType.FLOAT),
                        ("y", ParseType.FLOAT),
                        ("z", ParseType.FLOAT),
                    ],
                ),
                group(
                    "mag",
                    [
                        ("x", ParseType.FLOAT),
                        ("y", ParseType.FLOAT),
                        ("z", ParseType.FLOAT),
                    ],
                ),
            ],
        ),
        sensor_sid["barometer"]: SensorScheme(
            name="barometer",
            sid=sensor_sid["barometer"],
            groups=[
                group(
                    "barometer",
                    [
                        ("temperature", ParseType.FLOAT),
                        ("pressure", ParseType.FLOAT),
                    ],
                )
            ],
        ),
        sensor_sid["ppg"]: SensorScheme(
            name="ppg",
            sid=sensor_sid["ppg"],
            groups=[
                group(
                    "ppg",
                    [
                        ("red", ParseType.UINT32),
                        ("ir", ParseType.UINT32),
                        ("green", ParseType.UINT32),
                        ("ambient", ParseType.UINT32),
                    ],
                )
            ],
        ),
        sensor_sid["optical_temp"]: SensorScheme(
            name="optical_temp",
            sid=sensor_sid["optical_temp"],
            groups=[group("optical_temp", [("optical_temp", ParseType.FLOAT)])],
        ),
        sensor_sid["bone_acc"]: SensorScheme(
            name="bone_acc",
            sid=sensor_sid["bone_acc"],
            groups=[
                group(
                    "bone_acc",
                    [
                        ("x", ParseType.INT16),
                        ("y", ParseType.INT16),
                        ("z", ParseType.INT16),
                    ],
                )
            ],
        ),
    }
