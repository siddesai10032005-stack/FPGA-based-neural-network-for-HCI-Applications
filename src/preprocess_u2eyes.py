import zipfile
import xml.etree.ElementTree as ET
import csv
import os
from PIL import Image

ZIP_PATH = "data/raw/User_01.zip"

LEFT_DIR = "data/processed/eyes/left"
RIGHT_DIR = "data/processed/eyes/right"
CSV_PATH = "data/processed/eye_dataset.csv"

IMAGE_SIZE = (128, 64)

# Extra context around the annotated eye boundary
MARGIN_X = 30
MARGIN_Y = 25

os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def get_float(parent, name):
    return float(parent.find(name).text)


def get_eye_crop(img, poi_def):
    margin = poi_def.find("InteriorMargin")
    points = margin.find("Point2D").findall("Vector2")

    xs = [get_float(p, "x") for p in points]
    ys = [get_float(p, "y") for p in points]

    xmin = max(0, int(min(xs)) - MARGIN_X)
    xmax = min(img.width, int(max(xs)) + MARGIN_X)
    ymin = max(0, int(min(ys)) - MARGIN_Y)
    ymax = min(img.height, int(max(ys)) + MARGIN_Y)

    crop = img.crop((xmin, ymin, xmax, ymax))
    crop = crop.resize(IMAGE_SIZE)

    return crop, xmin, ymin, xmax, ymax


def get_pupil(poi_def):
    p = poi_def.find("PupilCenter2D")
    return (
        get_float(p, "x"),
        get_float(p, "y")
    )


def get_iris(poi_def):
    p = poi_def.find("IrisCenter2D")
    return (
        get_float(p, "x"),
        get_float(p, "y")
    )


rows = []
image_count = 0

with zipfile.ZipFile(ZIP_PATH) as z:

    for hp_num in range(1, 126):

        hp = f"HP_{hp_num:03d}"

        base = f"User_01/Grid_15/{hp}/"

        # ------------------------------------------------
        # Read head-pose annotation
        # ------------------------------------------------
        head_xml = ET.fromstring(
            z.read(base + "many_headpose.xml")
        )

        head_records = (
            head_xml.find("Headpose")
            .findall("HeadposeDef")
        )

        # ------------------------------------------------
        # Read eye annotation
        # ------------------------------------------------
        poi_xml = ET.fromstring(
            z.read(base + "many_poi_data.xml")
        )

        left_records = (
            poi_xml.find("POILeft")
            .findall("POIDef")
        )

        right_records = (
            poi_xml.find("POIRight")
            .findall("POIDef")
        )

        if not (
            len(head_records) == 15
            and len(left_records) == 15
            and len(right_records) == 15
        ):
            raise RuntimeError(
                f"Annotation count mismatch in {hp}: "
                f"head={len(head_records)}, "
                f"left={len(left_records)}, "
                f"right={len(right_records)}"
            )

        # ------------------------------------------------
        # Process 15 images
        # ------------------------------------------------
        for idx in range(15):

            image_number = idx + 1

            image_path = (
                f"{base}{image_number:02d}.png"
            )

            with z.open(image_path) as f:
                img = Image.open(f).convert("RGB")

                # Force loading before ZIP file moves on
                img.load()

            left_poi = left_records[idx]
            right_poi = right_records[idx]
            head = head_records[idx]

            # ------------------------------------------------
            # Eye crops
            # ------------------------------------------------
            left_crop, llx1, lly1, llx2, lly2 = get_eye_crop(
                img,
                left_poi
            )

            right_crop, rlx1, rly1, rlx2, rly2 = get_eye_crop(
                img,
                right_poi
            )

            sample_id = f"{hp_num:03d}_{image_number:02d}"

            left_filename = sample_id + ".png"
            right_filename = sample_id + ".png"

            left_crop.save(
                os.path.join(LEFT_DIR, left_filename)
            )

            right_crop.save(
                os.path.join(RIGHT_DIR, right_filename)
            )

            # ------------------------------------------------
            # Head pose
            # ------------------------------------------------
            rotation = head.find("Rotation")
            position = head.find("Position")
            look_at = head.find("LookAtPoint")

            head_rot_x = get_float(rotation, "x")
            head_rot_y = get_float(rotation, "y")
            head_rot_z = get_float(rotation, "z")

            head_pos_x = get_float(position, "x")
            head_pos_y = get_float(position, "y")
            head_pos_z = get_float(position, "z")

            look_at_x = get_float(look_at, "x")
            look_at_y = get_float(look_at, "y")
            look_at_z = get_float(look_at, "z")

            # ------------------------------------------------
            # Eye centers
            # ------------------------------------------------
            left_pupil_x, left_pupil_y = get_pupil(left_poi)
            right_pupil_x, right_pupil_y = get_pupil(right_poi)

            left_iris_x, left_iris_y = get_iris(left_poi)
            right_iris_x, right_iris_y = get_iris(right_poi)

            rows.append([
                sample_id,
                hp,
                image_number,

                left_filename,
                right_filename,

                head_rot_x,
                head_rot_y,
                head_rot_z,

                head_pos_x,
                head_pos_y,
                head_pos_z,

                look_at_x,
                look_at_y,
                look_at_z,

                left_pupil_x,
                left_pupil_y,

                right_pupil_x,
                right_pupil_y,

                left_iris_x,
                left_iris_y,

                right_iris_x,
                right_iris_y,

                llx1,
                lly1,
                llx2,
                lly2,

                rlx1,
                rly1,
                rlx2,
                rly2
            ])

            image_count += 1

            if image_count % 100 == 0:
                print(
                    f"Processed {image_count}/1875 images"
                )


# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------

header = [
    "sample_id",
    "hp_folder",
    "image_number",

    "left_image",
    "right_image",

    "head_rot_x",
    "head_rot_y",
    "head_rot_z",

    "head_pos_x",
    "head_pos_y",
    "head_pos_z",

    "look_at_x",
    "look_at_y",
    "look_at_z",

    "left_pupil_x",
    "left_pupil_y",

    "right_pupil_x",
    "right_pupil_y",

    "left_iris_x",
    "left_iris_y",

    "right_iris_x",
    "right_iris_y",

    "left_crop_xmin",
    "left_crop_ymin",
    "left_crop_xmax",
    "left_crop_ymax",

    "right_crop_xmin",
    "right_crop_ymin",
    "right_crop_xmax",
    "right_crop_ymax"
]

with open(CSV_PATH, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)


print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")
print(f"Images processed : {image_count}")
print(f"Left crops       : {len(os.listdir(LEFT_DIR))}")
print(f"Right crops      : {len(os.listdir(RIGHT_DIR))}")
print(f"CSV rows         : {len(rows)}")
print(f"CSV file         : {CSV_PATH}")
print("Crop size        : 128 x 64")
print("========================================")
