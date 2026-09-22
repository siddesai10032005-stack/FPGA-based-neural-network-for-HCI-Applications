import pandas as pd
import numpy as np

INPUT = "data/processed/gaze_labels.csv"
OUTPUT = "data/processed/final_gaze_labels.csv"


def rotation_matrix_xyz(rx, ry, rz):
    """
    Rotation matrix using X -> Y -> Z rotations.

    Angles are given in degrees.
    """

    rx, ry, rz = np.radians([rx, ry, rz])

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


df = pd.read_csv(INPUT)

relative_yaw = []
relative_pitch = []

relative_x = []
relative_y = []
relative_z = []

for _, row in df.iterrows():

    # World-space gaze direction
    direction_world = np.array([
        row["look_at_x"] - row["head_pos_x"],
        row["look_at_y"] - row["head_pos_y"],
        row["look_at_z"] - row["head_pos_z"]
    ])

    # Normalize
    direction_world = (
        direction_world /
        np.linalg.norm(direction_world)
    )

    # Head rotation
    R = rotation_matrix_xyz(
        row["head_rot_x"],
        row["head_rot_y"],
        row["head_rot_z"]
    )

    # Transform world direction into head coordinates
    direction_head = R.T @ direction_world

    x, y, z = direction_head

    relative_x.append(x)
    relative_y.append(y)
    relative_z.append(z)

    yaw = np.degrees(
        np.arctan2(x, -z)
    )

    pitch = np.degrees(
        np.arctan2(
            y,
            np.sqrt(x*x + z*z)
        )
    )

    relative_yaw.append(yaw)
    relative_pitch.append(pitch)


df["relative_gaze_x"] = relative_x
df["relative_gaze_y"] = relative_y
df["relative_gaze_z"] = relative_z

df["relative_gaze_yaw"] = relative_yaw
df["relative_gaze_pitch"] = relative_pitch


df.to_csv(OUTPUT, index=False)


print("=" * 65)
print("HEAD-RELATIVE GAZE LABEL GENERATION COMPLETE")
print("=" * 65)

print(f"Rows: {len(df)}")
print(f"Output: {OUTPUT}")

print("\nRelative gaze yaw:")
print(
    f"min = {df['relative_gaze_yaw'].min():.2f}°"
)
print(
    f"max = {df['relative_gaze_yaw'].max():.2f}°"
)

print("\nRelative gaze pitch:")
print(
    f"min = {df['relative_gaze_pitch'].min():.2f}°"
)
print(
    f"max = {df['relative_gaze_pitch'].max():.2f}°"
)

print("\nFirst 15 samples:")
print(
    df[
        [
            "sample_id",
            "head_rot_x",
            "head_rot_y",
            "head_rot_z",
            "look_at_x",
            "look_at_y",
            "relative_gaze_yaw",
            "relative_gaze_pitch"
        ]
    ].head(15).to_string(index=False)
)

print("=" * 65)
