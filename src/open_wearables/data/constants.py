from typing import Dict, List

SENSOR_SID: Dict[str, int] = {
    "imu": 0,
    "barometer": 1,
    "microphone": 2,
    "ppg": 4,
    "optical_temp": 6,
    "bone_acc": 7,
}

SID_NAMES: Dict[int, str] = {
    0: "imu",
    1: "barometer",
    2: "microphone",
    4: "ppg",
    6: "optical_temp",
    7: "bone_acc",
}

SENSOR_FORMATS: Dict[int, str] = {
    SENSOR_SID["imu"]: "<9f",
    SENSOR_SID["barometer"]: "<2f",
    SENSOR_SID["ppg"]: "<4I",
    SENSOR_SID["optical_temp"]: "<f",
    SENSOR_SID["bone_acc"]: "<3h",
}

LABELS: Dict[str, List[str]] = {
    "imu": [
        "acc.x",
        "acc.y",
        "acc.z",
        "gyro.x",
        "gyro.y",
        "gyro.z",
        "mag.x",
        "mag.y",
        "mag.z",
    ],
    "barometer": ["barometer.temperature", "barometer.pressure"],
    "ppg": ["ppg.red", "ppg.ir", "ppg.green", "ppg.ambient"],
    "bone_acc": ["bone_acc.x", "bone_acc.y", "bone_acc.z"],
    "optical_temp": ["optical_temp"],
    "microphone": ["mic.inner", "mic.outer"],
}

COLORS: Dict[str, List[str]] = {
    "ppg": ["red", "darkred", "green", "gray"],
}
