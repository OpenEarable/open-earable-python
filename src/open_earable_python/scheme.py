import enum

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
