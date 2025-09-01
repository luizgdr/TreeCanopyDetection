import json
import os
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Any
import matplotlib.pyplot as plt

# TODO Import the zip files from elsewhere
# TODO Extract zip


def points_from_segmentation(segmentation: list[float]) -> NDArray[Any]:
    points = []

    for i in range(0, len(segmentation), 2):
        points.append((segmentation[i], segmentation[i + 1]))

    return np.array(points, dtype=np.int32)


DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
ANNOTATION_PATH = os.path.join(DATA_DIR, "train_annotations.json")

# Open the file and load its content
with open(ANNOTATION_PATH, "r") as file:
    annotation_data = json.load(file)

for image_data in annotation_data["images"]:
    file_path = os.path.join(IMG_DIR, image_data["file_name"])

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Couldn't open {image_data['file_name']}")

    for annotation in image_data["annotations"]:
        label = annotation["class"]
        points = points_from_segmentation(annotation["segmentation"])

        cv2.polylines(
            img,
            [points],
            isClosed=True,
            color=(0, 0, 255) if annotation["class"] == "individual_tree" else (255, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    break  # debug only
