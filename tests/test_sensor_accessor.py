import pandas as pd

from open_earable_python.dataset import LABELS, SensorAccessor


def make_imu_df():
    data = {
        "acc.x": [1.0, 2.0],
        "acc.y": [3.0, 4.0],
        "gyro.x": [5.0, 6.0],
        "gyro.y": [7.0, 8.0],
        "mag.z": [9.0, 10.0],
    }
    df = pd.DataFrame(data)
    df.index = [0.0, 1.0]
    return df


def test_group_and_channel_accessors():
    df = make_imu_df()
    accessor = SensorAccessor(df, LABELS["imu"])

    # Full dataframe-style access
    assert list(accessor.columns) == list(df.columns)

    # Group access
    acc_df = accessor["acc"]
    assert list(acc_df.columns) == ["x", "y", "z"]

    gyro_df = accessor.gyro
    assert list(gyro_df.columns) == ["x", "y", "z"]

    # Channel access
    assert accessor["acc.x"].iloc[0] == 1.0
    assert accessor.acc["x"].iloc[1] == 2.0
    assert accessor.gyro["y"].iloc[0] == 7.0


def test_missing_channel_raises_key_error():
    df = make_imu_df()
    accessor = SensorAccessor(df, LABELS["imu"])

    try:
        _ = accessor["nonexistent"]
    except KeyError:
        return

    assert False, "Accessing a missing channel should raise KeyError"
