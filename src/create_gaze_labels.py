import pandas as pd
import numpy as np

INPUT = "data/processed/eye_dataset.csv"
OUTPUT = "data/processed/gaze_labels.csv"

df = pd.read_csv(INPUT)

# ---------------------------------------------------------
# 1. Build the world-space vector from the head position
#    toward the LookAtPoint.
# ---------------------------------------------------------

dx = df["look_at_x"] - df["head_pos_x"]
dy = df["look_at_y"] - df["head_pos_y"]
dz = df["look_at_z"] - df["head_pos_z"]

# ---------------------------------------------------------
# 2. Normalize the gaze vector
# ---------------------------------------------------------

norm = np.sqrt(dx**2 + dy**2 + dz**2)

df["gaze_world_x"] = dx / norm
df["gaze_world_y"] = dy / norm
df["gaze_world_z"] = dz / norm

# ---------------------------------------------------------
# 3. World-space yaw/pitch
#
# These are geometric descriptors, NOT yet the final
# head-relative gaze labels.
# ---------------------------------------------------------

df["gaze_world_yaw"] = np.degrees(
    np.arctan2(df["gaze_world_x"], -df["gaze_world_z"])
)

df["gaze_world_pitch"] = np.degrees(
    np.arctan2(
        df["gaze_world_y"],
        np.sqrt(
            df["gaze_world_x"]**2 +
            df["gaze_world_z"]**2
        )
    )
)

# ---------------------------------------------------------
# 4. Simple categorical representation of the target grid
#
# The 15-point U2Eyes grid is:
#
# X = -11, 0, +11
# Y = -11, -5.5, 0, +5.5, +11
#
# We keep the continuous values too.
# ---------------------------------------------------------

def horizontal_label(x):
    if x < -5.5:
        return "LEFT"
    elif x > 5.5:
        return "RIGHT"
    return "CENTER"


def vertical_label(y):
    if y < -2.75:
        return "DOWN"
    elif y > 2.75:
        return "UP"
    return "CENTER"


df["target_horizontal"] = df["look_at_x"].apply(
    horizontal_label
)

df["target_vertical"] = df["look_at_y"].apply(
    vertical_label
)

df["target_direction"] = (
    df["target_vertical"] + "_" +
    df["target_horizontal"]
)

# ---------------------------------------------------------
# 5. Relative head/gaze descriptor
#
# This is intentionally NOT used as the neural-network
# target yet. It is retained for later fusion.
# ---------------------------------------------------------

df["head_yaw"] = df["head_rot_y"]
df["head_pitch"] = df["head_rot_x"]
df["head_roll"] = df["head_rot_z"]

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("GAZE LABEL GENERATION COMPLETE")
print("=" * 60)

print(f"Input rows : {len(df)}")
print(f"Output     : {OUTPUT}")

print("\nWorld gaze yaw range:")
print(
    f"{df['gaze_world_yaw'].min():.2f} "
    f"to "
    f"{df['gaze_world_yaw'].max():.2f} degrees"
)

print("\nWorld gaze pitch range:")
print(
    f"{df['gaze_world_pitch'].min():.2f} "
    f"to "
    f"{df['gaze_world_pitch'].max():.2f} degrees"
)

print("\nTarget horizontal:")
print(df["target_horizontal"].value_counts())

print("\nTarget vertical:")
print(df["target_vertical"].value_counts())

print("\nTarget direction:")
print(df["target_direction"].value_counts())

print("\nFirst 10 samples:")
print(
    df[
        [
            "sample_id",
            "head_rot_x",
            "head_rot_y",
            "look_at_x",
            "look_at_y",
            "gaze_world_yaw",
            "gaze_world_pitch",
            "target_direction"
        ]
    ].head(10).to_string(index=False)
)

print("=" * 60)
