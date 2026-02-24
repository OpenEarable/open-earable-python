import enum
from typing import Dict, Mapping, Sequence

class ParseType(enum.Enum):
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    FLOAT = "float"
    DOUBLE = "double"

class SensorComponentScheme:
    def __init__(self, name: str, data_type: ParseType):
        self.name = name
        self.data_type = data_type

    def __repr__(self):
        return f"SensorComponentScheme(name={self.name}, data_type={self.data_type})"

class SensorComponentGroupScheme:
    def __init__(self, name: str, components: list[SensorComponentScheme]):
        self.name = name
        self.components = components
    
    def __repr__(self):
        return f"SensorComponentGroupScheme(name={self.name}, components={self.components})"

class SensorScheme:
    """
    A class representing the schema for sensor data in an earable device.
    """

    def __init__(self, name: str, sid: int, groups: list[SensorComponentGroupScheme]):
        self.name = name
        self.sid = sid
        self.groups = groups

    def __repr__(self):
        return f"SensorScheme(name={self.name}, sid={self.sid}, groups={self.groups})"


def _group(
    name: str,
    components: Sequence[tuple[str, ParseType]],
) -> SensorComponentGroupScheme:
    return SensorComponentGroupScheme(
        name=name,
        components=[
            SensorComponentScheme(component_name, parse_type)
            for component_name, parse_type in components
        ],
    )


def build_default_sensor_schemes(sensor_sid: Mapping[str, int]) -> Dict[int, SensorScheme]:
    """Build default non-microphone sensor schemes keyed by SID."""
    return {
        sensor_sid["imu"]: SensorScheme(
            name="imu",
            sid=sensor_sid["imu"],
            groups=[
                _group(
                    "acc",
                    [("x", ParseType.FLOAT), ("y", ParseType.FLOAT), ("z", ParseType.FLOAT)],
                ),
                _group(
                    "gyro",
                    [("x", ParseType.FLOAT), ("y", ParseType.FLOAT), ("z", ParseType.FLOAT)],
                ),
                _group(
                    "mag",
                    [("x", ParseType.FLOAT), ("y", ParseType.FLOAT), ("z", ParseType.FLOAT)],
                ),
            ],
        ),
        sensor_sid["barometer"]: SensorScheme(
            name="barometer",
            sid=sensor_sid["barometer"],
            groups=[
                _group(
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
                _group(
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
            groups=[_group("optical_temp", [("optical_temp", ParseType.FLOAT)])],
        ),
        sensor_sid["bone_acc"]: SensorScheme(
            name="bone_acc",
            sid=sensor_sid["bone_acc"],
            groups=[
                _group(
                    "bone_acc",
                    [("x", ParseType.INT16), ("y", ParseType.INT16), ("z", ParseType.INT16)],
                )
            ],
        ),
    }
