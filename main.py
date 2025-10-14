import json
import os
import shutil
import cv2
import numpy as np
import albumentations as A
import random
from pathlib import Path
from typing import Any, Dict, Iterable
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
import torch

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
ANNOTATION_PATH = os.path.join(DATA_DIR, "train_annotations.json")
OUT_MASK_DIR = os.path.join(DATA_DIR, "masks")
AUG_IMG_DIR = os.path.join(DATA_DIR, "augmented", "images")
AUG_MSK_DIR = os.path.join(DATA_DIR, "augmented", "masks")
DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")
IMG_TRAIN_DIR = os.path.join(DATASET_DIR, "images", "train")
IMG_VAL_DIR = os.path.join(DATASET_DIR, "images", "val")
LABEL_TRAIN_DIR = os.path.join(DATASET_DIR, "labels", "train")
LABEL_VAL_DIR = os.path.join(DATASET_DIR, "labels", "val")
AUG_TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "images", "train", "aug")
AUG_TRAIN_LABEL_DIR = os.path.join(DATASET_DIR, "labels", "train", "aug")
AUG_TRAIN_MSK_DIR = os.path.join(DATASET_DIR, "masks", "train", "aug")

for dir_path in [
    OUT_MASK_DIR,
    AUG_IMG_DIR,
    AUG_MSK_DIR,
    IMG_TRAIN_DIR,
    IMG_VAL_DIR,
    LABEL_TRAIN_DIR,
    LABEL_VAL_DIR,
    AUG_TRAIN_IMG_DIR,
    AUG_TRAIN_LABEL_DIR,
    AUG_TRAIN_MSK_DIR,
]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


DEBUG_MAX_IMAGES = None
AUG_PER_IMAGE = 5

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


FORCE_REPREP = False


data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")
if os.path.exists(data_yaml_path) and not FORCE_REPREP:
    print(
        "Dataset already exists! Skipping data preparation. Set FORCE_REPREP=True to regenerate."
    )
else:
    print("Preparing dataset...")

    def points_from_segmentation(segmentation: Iterable[float]) -> NDArray[Any]:
        pts = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
        return pts.astype(np.int32)

    def make_mask(shape_hw: tuple[int, int], annotations: list) -> NDArray[np.uint8]:
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)

        if not annotations:
            return mask

        for ann in annotations:
            segmentation = ann["segmentation"]
            pts = points_from_segmentation(segmentation).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)

        return mask

    def colorize_mask(
        mask: NDArray, id_to_color: Dict[int, tuple[int, int, int]]
    ) -> NDArray:
        h, w = mask.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)

        color[mask == 255] = [0, 255, 0]
        return color

    def overlay_mask_on_image_bgr(
        img_bgr: NDArray,
        mask: NDArray,
        id_to_color: Dict[int, tuple[int, int, int]],
        alpha: float = 0.4,
    ) -> NDArray:
        color_mask_bgr = colorize_mask(mask, id_to_color)
        overlay = cv2.addWeighted(img_bgr, 1.0, color_mask_bgr, alpha, 0)
        return overlay

    class_to_id = {"tree": 0}
    id_to_color = {1: (0, 255, 0)}

    with open(ANNOTATION_PATH, "r") as file:
        annotation_data = json.load(file)

    images_list = annotation_data["images"]
    train_img_data, val_img_data = train_test_split(
        images_list, test_size=0.2, random_state=RANDOM_SEED
    )

    def get_annotations_for_image(image_data: dict):
        annotations = image_data.get("annotations", [])

        return [a for a in annotations if a.get("class", "") == "individual_tree"]

    def polygons_to_yolo_label(
        image_width: int, image_height: int, annotations: list
    ) -> str:
        lines = []
        if not annotations:
            return ""

        for ann in annotations:
            cls_id = 0
            segmentation = ann["segmentation"]
            pts = np.array(segmentation, dtype=np.float32).reshape(-1, 2)

            norm_pts = pts / np.array([image_width, image_height])
            norm_pts = norm_pts.flatten()

            line = f"{cls_id} {' '.join(f'{x:.6f}' for x in norm_pts)}"
            lines.append(line)

        return "\n".join(lines)

    def process_split(
        img_data_list: list,
        img_out_dir: str,
        label_out_dir: str,
        is_train: bool = False,
    ):
        processed_count = 0
        for image_data in img_data_list:
            file_name = image_data["file_name"]
            file_path = os.path.join(IMG_DIR, file_name)
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                print(f"Warning: Could not load {file_name}")
                continue

            h = image_data["height"]
            w = image_data["width"]
            annotations = get_annotations_for_image(image_data)

            base_name = Path(file_name).stem
            img_out_path = os.path.join(img_out_dir, f"{base_name}.jpg")
            cv2.imwrite(img_out_path, img_bgr)

            label_content = polygons_to_yolo_label(w, h, annotations)
            label_out_path = os.path.join(label_out_dir, f"{base_name}.txt")
            with open(label_out_path, "w") as f:
                f.write(label_content)

            mask = make_mask((h, w), annotations)
            mask_path = os.path.join(OUT_MASK_DIR, f"{base_name}.png")
            cv2.imwrite(mask_path, mask)

            processed_count += 1
            if DEBUG_MAX_IMAGES is not None and processed_count >= DEBUG_MAX_IMAGES:
                break

        print(
            f"Processed {len(img_data_list)} images for {'train' if is_train else 'val'} split."
        )

    process_split(train_img_data, IMG_TRAIN_DIR, LABEL_TRAIN_DIR, is_train=True)
    process_split(val_img_data, IMG_VAL_DIR, LABEL_VAL_DIR, is_train=False)

    print(
        f"Original Train: {len(train_img_data)} images, Val: {len(val_img_data)} images"
    )

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )

    def raster_mask_to_yolo_polygons(
        mask_path: str, img_width: int, img_height: int, output_txt_path: str
    ):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            pts = contour.squeeze().astype(np.float32)
            if len(pts) < 3:
                continue

            norm_pts = pts / np.array([img_width, img_height])
            norm_pts = norm_pts.flatten()
            line = f"0 {' '.join(f'{x:.6f}' for x in norm_pts)}"
            lines.append(line)

        with open(output_txt_path, "w") as f:
            f.write("\n".join(lines))

    processed = 0
    aug_count = 0
    for image_data in train_img_data:
        file_name = image_data["file_name"]
        file_path = os.path.join(IMG_DIR, file_name)
        img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)

        if img_bgr is None:
            continue

        h = image_data["height"]
        w = image_data["width"]
        annotations = get_annotations_for_image(image_data)
        mask = make_mask((h, w), annotations)

        base_name = Path(file_name).stem

        base_mask_path = os.path.join(OUT_MASK_DIR, f"{base_name}.png")
        cv2.imwrite(base_mask_path, mask)

        for i in range(AUG_PER_IMAGE):
            out = transform(image=img_bgr, mask=mask)
            aug_img_bgr = out["image"]
            aug_msk = out["mask"]

            aug_img_out_path = os.path.join(
                AUG_TRAIN_IMG_DIR, f"{base_name}_aug_{i:02d}.jpg"
            )
            aug_msk_out_path = os.path.join(
                AUG_TRAIN_MSK_DIR, f"{base_name}_aug_{i:02d}.png"
            )
            cv2.imwrite(aug_img_out_path, aug_img_bgr)
            cv2.imwrite(aug_msk_out_path, aug_msk)

            aug_label_out_path = os.path.join(
                AUG_TRAIN_LABEL_DIR, f"{base_name}_aug_{i:02d}.txt"
            )
            raster_mask_to_yolo_polygons(aug_msk_out_path, w, h, aug_label_out_path)

            aug_count += 1

        processed += 1
        if DEBUG_MAX_IMAGES is not None and processed >= DEBUG_MAX_IMAGES:
            break

    print(f"Generated {aug_count} augmented samples for train.")

    shutil.rmtree(AUG_TRAIN_MSK_DIR)
    print("Temp augmented masks cleaned up.")

    with open(data_yaml_path, "w") as f:
        f.write(
            f"""path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val

nc: 1
names: ['tree']
"""
        )

    print(f"data.yaml created at: {data_yaml_path}")

print("Data prep complete or skipped. Ready for training/validation/inference.")


dataset_dir = Path(DATASET_DIR)
train_imgs = len(list(dataset_dir.glob("images/train/**/*.jpg")))
val_imgs = len(list(dataset_dir.glob("images/val/*.jpg")))
train_labels = len(list(dataset_dir.glob("labels/train/**/*.txt")))
val_labels = len(list(dataset_dir.glob("labels/val/*.txt")))

print(f"Train images: {train_imgs}")
print(f"Val images: {val_imgs}")
print(f"Train labels: {train_labels}")
print(f"Val labels: {val_labels}")

sample_label = next(dataset_dir.glob("labels/train/*.txt"), None)
if sample_label:
    with open(sample_label) as f:
        content = f.read()
    print(f"Sample label content: {content[:100]}...")
else:
    print("No sample label found—check paths!")


def train_model():
    from ultralytics import YOLO

    model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=320,
        batch=8,
        name="tree_seg",
        seed=RANDOM_SEED,
        patience=10,
        save=True,
        plots=True,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
    )

    print(
        "Training completed. Best model saved at: runs/segment/tree_seg/weights/best.pt"
    )
    return model


def validate_model(model_path: str):
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml_path)
    print(f"Validation Results:")
    print(f"Box mAP@0.5: {metrics.box.map50}")
    print(f"Box mAP@0.5:0.95: {metrics.box.map}")
    print(f"Seg mAP@0.5: {metrics.seg.map50}")
    print(f"Seg mAP@0.5:0.95: {metrics.seg.map}")

    print("Validation plots saved in runs/segment/val/")


def run_inference(model_path: str, source_path: str):
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import os
    from pathlib import Path

    if not os.path.exists(source_path):
        print(f"Source path not found: {source_path}. Skipping inference.")
        return

    output_dir = "runs/segment/generated"
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    results = model.predict(
        source=source_path,
        save=True,
        save_txt=True,
        save_conf=True,
        conf=0.5,
        iou=0.7,
        imgsz=640,
        verbose=True,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    if not isinstance(results, list):
        print("No results to process. Ensure the source path is valid.")
        return

    for idx, result in enumerate(results):
        img_path = result.path
        try:
            orig_img = cv2.imread(img_path)
            if orig_img is None:
                print(f"Could not load {img_path}. Skipping this image.")
                continue

            orig_h, orig_w = orig_img.shape[:2]
            print(
                f"Processing image: {os.path.basename(img_path)} with size: {orig_w}x{orig_h}"
            )

            masks = result.masks.data  # type: ignore
            if masks is not None and len(masks) > 0:
                combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

                total_tree_pixels = 0
                for i, mask in enumerate(masks):
                    tree_mask = mask.cpu().numpy().astype(np.uint8)

                    tree_mask_resized = cv2.resize(
                        tree_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                    ).astype(np.uint8)

                    combined_mask = cv2.bitwise_or(
                        combined_mask, tree_mask_resized * 255
                    )

                    tree_pixels = np.sum(tree_mask_resized > 0)
                    total_tree_pixels += tree_pixels

                    if result.boxes is not None and i < len(result.boxes.conf):
                        conf = result.boxes.conf[i].item()
                        print(
                            f"Tree {i} in {os.path.basename(img_path)}: Canopy area = {tree_pixels} pixels, Confidence = {conf:.2f}"
                        )
                    else:
                        print(
                            f"Tree {i} in {os.path.basename(img_path)}: Canopy area = {tree_pixels} pixels"
                        )

                base_name = Path(img_path).stem
                combined_mask_path = os.path.join(
                    output_dir, f"{base_name}_combined_mask.png"
                )
                cv2.imwrite(combined_mask_path, combined_mask)
                print(f"Saved combined mask for {base_name} at {combined_mask_path}")

                overlay = orig_img.copy()
                overlay_bgra = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
                tree_areas = combined_mask > 0
                overlay_bgra[tree_areas] = [
                    0,
                    255,
                    0,
                    128,
                ]

                overlay_path = os.path.join(
                    output_dir, f"{base_name}_semi_transparent_overlay.png"
                )
                cv2.imwrite(overlay_path, overlay_bgra)
                print(
                    f"Saved semi-transparent overlay for {base_name} at {overlay_path}"
                )

                canopy_coverage = (total_tree_pixels / (orig_h * orig_w)) * 100
                print(
                    f"Total tree canopy coverage for {base_name}: {canopy_coverage:.2f}%"
                )

            polygons = result.masks.xy  # type: ignore
            for i, poly in enumerate(polygons):
                if len(poly) > 0:
                    poly_area = cv2.contourArea(poly.astype(np.int32))
                    print(f"Tree {i} polygon area in {base_name}: {poly_area} pixels")

        except PermissionError as e:
            print(f"Permission error for {img_path}: {e}. Skipping this image.")
        except Exception as e:
            print(f"Error processing {img_path}: {e}. Skipping to next.")

    print(
        "Inference outputs saved in runs/segment/predict/ and custom files in runs/segment/generated/"
    )


def export_model(model_path: str, format_type: str = "onnx"):
    from ultralytics import YOLO

    model = YOLO(model_path)
    exported_path = model.export(format=format_type)
    print(f"Model exported to: {exported_path}")
    return exported_path


if __name__ == "__main__":
    model = train_model()

    validate_model("runs/segment/tree_seg/weights/best.pt")

    run_inference(
        "runs/segment/tree_seg/weights/best.pt",
        IMG_VAL_DIR,
    )

    exported_path = export_model("runs/segment/tree_seg/weights/best.pt", "onnx")

    print(f"Exported model to path '{exported_path}'")
