import warnings

from .dataset import (
    SensorDataset,
    load_recordings,
)

_DEPRECATION_MESSAGE = (
    "open-earable-python is deprecated and has been renamed to open-wearable. "
    "Install open-wearable and migrate imports from `open_earable_python` to "
    "`open_wearable`. This package will no longer be developed or maintained,"
    " and will eventually be removed from PyPI. "
)

# Use FutureWarning so migration guidance is visible by default.
warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)

__all__ = [
    "SensorDataset",
    "load_recordings",
]
