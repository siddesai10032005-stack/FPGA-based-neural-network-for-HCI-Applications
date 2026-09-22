import zipfile
import xml.etree.ElementTree as ET
import csv
import os


ZIP_PATH = "data/raw/User_01.zip"
OUTPUT_CSV = "data/processed/u2eyes_metadata.csv"


def get_xyz(parent, tag):
    """Read x, y, z values from an XML element."""
    node = parent.find(tag)

    return (
        float(node.find("x").text),
        float(node.find("y").text),
        float(node.find("z").text),
    )


def get_xy(parent, tag):
    """Read x, y values from an XML element."""
    node = parent.find(tag)

    return (
        float(node.find("x").text),
        float(node.find("y").text),
    )


def parse_headpose(xml_bytes):
    """Parse head-pose XML and return one record per image."""

    root = ET.fromstring(xml_bytes)

    records = root.find("Headpose").findall("HeadposeDef")

    output = []

    for record in records:

        rot_x, rot_y, rot_z = get_xyz(record, "Rotation")
        pos_x, pos_y, pos_z = get_xyz(record, "Position")
        look_x, look_y, look_z = get_xyz(record, "LookAtPoint")

        output.append({
            "head_rot_x": rot_x,
            "head_rot_y": rot_y,
            "head_rot_z": rot_z,
            "head_pos_x": pos_x,
            "head_pos_y": pos_y,
            "head_pos_z": pos_z,
            "look_at_x": look_x,
            "look_at_y": look_y,
            "look_at_z": look_z,
        })

    return output


def parse_eye_records(xml_bytes):
    """Parse left and right eye POI information."""

    root = ET.fromstring(xml_bytes)

    left_records = root.find("POILeft").findall("POIDef")
    right_records = root.find("POIRight").findall("POIDef")

    if len(left_records) != len(right_records):
        raise ValueError("Left/right eye record counts do not match.")

    left_output = []
    right_output = []

    for record in left_records:

        iris_x, iris_y = get_xy(record, "IrisCenter2D")
        pupil_x, pupil_y = get_xy(record, "PupilCenter2D")
        cornea_x, cornea_y = get_xy(record, "CorneaCenter2D")
        globe_x, globe_y = get_xy(record, "GlobeCenter2D")

        left_output.append({
            "left_iris_x": iris_x,
            "left_iris_y": iris_y,
            "left_pupil_x": pupil_x,
            "left_pupil_y": pupil_y,
            "left_cornea_x": cornea_x,
            "left_cornea_y": cornea_y,
            "left_globe_x": globe_x,
            "left_globe_y": globe_y,
        })

    for record in right_records:

        iris_x, iris_y = get_xy(record, "IrisCenter2D")
        pupil_x, pupil_y = get_xy(record, "PupilCenter2D")
        cornea_x, cornea_y = get_xy(record, "CorneaCenter2D")
        globe_x, globe_y = get_xy(record, "GlobeCenter2D")

        right_output.append({
            "right_iris_x": iris_x,
            "right_iris_y": iris_y,
            "right_pupil_x": pupil_x,
            "right_pupil_y": pupil_y,
            "right_cornea_x": cornea_x,
            "right_cornea_y": cornea_y,
            "right_globe_x": globe_x,
            "right_globe_y": globe_y,
        })

    return left_output, right_output


def main():

    print("Opening:", ZIP_PATH)

    rows = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:

        names = z.namelist()

        hp_folders = sorted({
            name.split("/")[-2]
            for name in names
            if "/Grid_15/HP_" in name
            and name.endswith("/")
        })

        print("Found HP folders:", len(hp_folders))

        for folder_index, hp_folder in enumerate(hp_folders, start=1):

            base = f"User_01/Grid_15/{hp_folder}/"

            headpose_path = base + "many_headpose.xml"
            poi_path = base + "many_poi_data.xml"

            headpose = parse_headpose(
                z.read(headpose_path)
            )

            left_eye, right_eye = parse_eye_records(
                z.read(poi_path)
            )

            image_names = sorted([
                name for name in names
                if name.startswith(base)
                and name.endswith(".png")
            ])

            if not (
                len(image_names)
                == len(headpose)
                == len(left_eye)
                == len(right_eye)
            ):
                raise ValueError(
                    f"Record mismatch in {hp_folder}: "
                    f"images={len(image_names)}, "
                    f"headpose={len(headpose)}, "
                    f"left={len(left_eye)}, "
                    f"right={len(right_eye)}"
                )

            for i, image_path in enumerate(image_names):

                row = {
                    "image_path": image_path,
                    "hp_folder": hp_folder,
                }

                row.update(headpose[i])
                row.update(left_eye[i])
                row.update(right_eye[i])

                rows.append(row)

            if folder_index % 10 == 0:
                print(
                    f"Processed {folder_index}/{len(hp_folders)} folders"
                )

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    fieldnames = list(rows[0].keys())

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("====================================")
    print("Dataset parsing completed")
    print("====================================")
    print("Total rows:", len(rows))
    print("Output:", OUTPUT_CSV)
    print("Columns:", len(fieldnames))


if __name__ == "__main__":
    main()
