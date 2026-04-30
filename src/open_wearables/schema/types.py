import enum
from typing import Optional, Sequence


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

    def __repr__(self) -> str:
        return f"SensorComponentScheme(name={self.name}, data_type={self.data_type})"


class SensorComponentGroupScheme:
    def __init__(self, name: str, components: list[SensorComponentScheme]):
        self.name = name
        self.components = components

    def __repr__(self) -> str:
        return f"SensorComponentGroupScheme(name={self.name}, components={self.components})"


class SensorScheme:
    """Schema definition for one OpenEarable sensor stream."""

    def __init__(
        self,
        name: str,
        sid: int,
        groups: list[SensorComponentGroupScheme],
        sampling_rate: Optional[float] = None,
    ):
        self.name = name
        self.sid = sid
        self.groups = groups
        self.sampling_rate = sampling_rate

    def __repr__(self) -> str:
        return f"SensorScheme(name={self.name}, sid={self.sid}, groups={self.groups}, sampling_rate={self.sampling_rate})"


def group(
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
