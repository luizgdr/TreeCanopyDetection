import json, os, cv2, numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from numpy.typing import NDArray
from typing import Any, Callable


DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
ANNOTATION_PATH = os.path.join(DATA_DIR, "train_annotations.json")
INDIVIDUAL_TREE_COLOR = (0, 0, 255)
TREE_GROUP_COLOR = (255, 0, 0)


def points_from_segmentation(seg: list[float]) -> NDArray[Any]:
    pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
    return np.array(pts, dtype=np.int32)


def clamp(i: int) -> int:
    return max(0, min(n - 1, i))


def annotate_image(img_bgr: np.ndarray, image_data: dict) -> np.ndarray:
    out = img_bgr.copy()
    for ann in image_data.get("annotations", []):
        pts = points_from_segmentation(ann["segmentation"])
        if len(pts) >= 3:
            color = (
                INDIVIDUAL_TREE_COLOR
                if ann["class"] == "individual_tree"
                else TREE_GROUP_COLOR
            )
            cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
    return out


def show_pair(i: int):
    global idx
    idx = clamp(i)
    image_data = images[idx]
    file_path = os.path.join(IMG_DIR, image_data["file_name"])

    base = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(f"Couldn't open {image_data['file_name']}")

    left_bgr = annotate_image(base, image_data)
    right_bgr = process_image(base.copy(), image_data)

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    ax_left.clear()
    ax_left.axis("off")
    ax_right.clear()
    ax_right.axis("off")

    ax_left.imshow(left_rgb)
    ax_right.imshow(right_rgb)

    ax_left.set_title(f"{idx+1}/{n} — {image_data['file_name']} (annotated)")
    ax_right.set_title("handled copy")

    fig.canvas.draw_idle()
    text_box.set_val(str(idx + 1))


def on_key(event):
    if event.key in ("right", "n"):
        show_pair(idx + 1)
    elif event.key in ("left", "p"):
        show_pair(idx - 1)
    elif event.key == "r":  # re-run handler on current image
        show_pair(idx)
    elif event.key == "q":
        plt.close(fig)
    elif event.key == "g":  # prompt jump
        try:
            val = int(input(f"Go to index (1-{n}): ").strip())
            show_pair(val - 1)
        except Exception:
            pass


def submit(text: str):
    try:
        val = int(text)
        show_pair(val - 1)
    except ValueError:
        text_box.set_val(str(idx + 1))
    text_box.on_submit(submit)


def process_image(img_bgr: np.ndarray, image_data: dict) -> np.ndarray:
    return img_bgr


with open(ANNOTATION_PATH, "r") as f:
    annotation_data = json.load(f)

images = annotation_data["images"]
n = len(images)
idx = 0

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.15)  # room for controls

fig.canvas.mpl_connect("key_press_event", on_key)

ax_box = plt.axes([0.15, 0.04, 0.25, 0.07])  # type: ignore
text_box = TextBox(ax_box, "Go to #", initial="1")

ax_go = plt.axes([0.42, 0.04, 0.08, 0.07])  # type: ignore
btn_go = Button(ax_go, "Go")
btn_go.on_clicked(lambda _: submit(text_box.text))

ax_re = plt.axes([0.52, 0.04, 0.12, 0.07])  # type: ignore
btn_re = Button(ax_re, "Reprocess (r)")
btn_re.on_clicked(lambda _: show_pair(idx))

show_pair(idx)
plt.show()
