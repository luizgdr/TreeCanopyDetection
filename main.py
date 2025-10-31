import logging
import os
import cv2
import json
from shapely.geometry import Polygon
from deepforest.main import deepforest  # type: ignore
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import shutil
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
from numpy.typing import NDArray
from typing import Any, Iterable
from sklearn.model_selection import train_test_split


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


def get_annotations_for_image(image_data: dict):
    return image_data.get("annotations", [])


def polygons_to_yolo_label(
    image_width: int, image_height: int, annotations: list
) -> str:
    lines = []
    if not annotations:
        return ""

    for ann in annotations:
        if ann["class"] == "individual_tree":
            cls_id = 0
        elif ann["class"] == "group_of_trees":
            cls_id = 1
        else:
            continue

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
            logger.error(f"Could not load {file_name}")
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

    logger.info(
        f"Processed {len(img_data_list)} images for {'train' if is_train else 'val'} split."
    )


def prep_dataset():
    if os.path.exists(DATA_YAML_PATH) and not FORCE_REPREP:
        logger.info(
            "Dataset already exists! Skipping data preparation. Set FORCE_REPREP=True to regenerate."
        )
    else:
        logger.info("Preparing dataset...")

        with open(ANNOTATION_PATH, "r") as file:
            annotation_data = json.load(file)

        images_list = annotation_data["images"]
        train_img_data, val_img_data = train_test_split(
            images_list, test_size=0.2, random_state=RANDOM_SEED
        )

        process_split(train_img_data, IMG_TRAIN_DIR, LABEL_TRAIN_DIR, is_train=True)
        process_split(val_img_data, IMG_VAL_DIR, LABEL_VAL_DIR, is_train=False)

        logger.info(
            f"Original Train: {len(train_img_data)} images, Val: {len(val_img_data)} images"
        )

        processed = 0
        aug_count = 0
        for image_data in train_img_data:
            file_name = image_data["file_name"]
            file_path = os.path.join(IMG_DIR, file_name)
            img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            annotations = get_annotations_for_image(image_data)
            mask = make_mask((h, w), annotations)

            base_name = Path(file_name).stem

            base_mask_path = os.path.join(OUT_MASK_DIR, f"{base_name}.png")
            cv2.imwrite(base_mask_path, mask)

            quadrants = [
                (0, h // 2, 0, w // 2, "top_left"),
                (0, h // 2, w // 2, w, "top_right"),
                (h // 2, h, 0, w // 2, "bottom_left"),
                (h // 2, h, w // 2, w, "bottom_right"),
            ]
            for _, (y_start, y_end, x_start, x_end, quad_name) in enumerate(quadrants):
                quad_img = img_bgr[y_start:y_end, x_start:x_end]
                quad_h, quad_w = quad_img.shape[:2]
                quad_mask = mask[y_start:y_end, x_start:x_end]

                quad_annotations = []
                quad_bbox = Polygon(
                    [
                        (x_start, y_start),
                        (x_end, y_start),
                        (x_end, y_end),
                        (x_start, y_end),
                    ]
                )
                for ann in annotations:
                    segmentation = ann["segmentation"]
                    pts = np.array(segmentation, dtype=np.float32).reshape(-1, 2)

                    poly = Polygon(pts)
                    if not poly.is_valid:
                        poly = poly.buffer(0)

                    clipped = poly.intersection(quad_bbox)
                    if clipped.is_empty or clipped.geom_type != "Polygon":
                        continue

                    clipped_pts = np.array(clipped.exterior.coords)  # type: ignore

                    if len(clipped_pts) < 3 or not clipped.is_valid:
                        continue

                    clipped_pts[:, 0] -= x_start
                    clipped_pts[:, 1] -= y_start

                    clipped_pts[:, 0] = np.clip(clipped_pts[:, 0], 0, quad_w)
                    clipped_pts[:, 1] = np.clip(clipped_pts[:, 1], 0, quad_h)

                    adjusted_ann = ann.copy()
                    adjusted_ann["segmentation"] = clipped_pts.flatten().tolist()
                    quad_annotations.append(adjusted_ann)

                aug_img_out_path = os.path.join(
                    AUG_TRAIN_IMG_DIR, f"{base_name}_quad_{quad_name}.jpg"
                )

                cv2.imwrite(aug_img_out_path, quad_img)
                aug_msk_out_path = os.path.join(
                    AUG_TRAIN_MSK_DIR, f"{base_name}_quad_{quad_name}.png"
                )

                cv2.imwrite(aug_msk_out_path, quad_mask)
                aug_label_out_path = os.path.join(
                    AUG_TRAIN_LABEL_DIR, f"{base_name}_quad_{quad_name}.txt"
                )

                label_content = polygons_to_yolo_label(quad_w, quad_h, quad_annotations)
                with open(aug_label_out_path, "w") as f:
                    f.write(label_content)
                aug_count += 1

            processed += 1
            if DEBUG_MAX_IMAGES is not None and processed >= DEBUG_MAX_IMAGES:
                break

        logger.info(f"Generated {aug_count} quadrant samples for train.")

        if REMOVE_TEMP_MASKS:
            shutil.rmtree(AUG_TRAIN_MSK_DIR)
            logger.info("Temp augmented masks cleaned up.")

        with open(DATA_YAML_PATH, "w") as f:
            f.write(
                f"path: {os.path.abspath(DATASET_DIR)}\n"
                + "train: images/train\n"
                + "val: images/val\n\n"
                + "nc: 2\n"
                + "names: ['individual_tree', 'group_of_trees']\n"
            )

        logger.info(f"data.yaml created at: {DATA_YAML_PATH}")

    logger.info(
        "Data prep complete or skipped. Ready for training/validation/inference."
    )


def validate_dataset():
    dataset_dir = Path(DATASET_DIR)
    train_imgs = len(list(dataset_dir.glob("images/train/**/*.jpg")))
    val_imgs = len(list(dataset_dir.glob("images/val/*.jpg")))
    train_labels = len(list(dataset_dir.glob("labels/train/**/*.txt")))
    val_labels = len(list(dataset_dir.glob("labels/val/*.txt")))

    logger.info(f"Train images: {train_imgs}")
    logger.info(f"Val images: {val_imgs}")
    logger.info(f"Train labels: {train_labels}")
    logger.info(f"Val labels: {val_labels}")

    sample_label = next(dataset_dir.glob("labels/train/*.txt"), None)
    if sample_label:
        with open(sample_label) as f:
            content = f.read()
        logger.info(f"Sample label content: {content[:100]}...")
    else:
        logger.warning("No sample label found—check paths!")


def train_model():
    model = YOLO("yolo11m-seg.pt")

    device = "cpu"
    if torch.cuda.is_available():
        logger.info("CUDA is available! Using CUDA on training.")
        device = 0

    model.train(
        data=DATA_YAML_PATH,
        epochs=100,
        imgsz=1024,
        batch=2,
        name="tree_seg",
        accumulate=4,
        seed=RANDOM_SEED,
        patience=10,
        save=True,
        plots=True,
        device=device,
        workers=4,
        augment=True,
    )

    logger.info(
        "Training completed. Best model saved at: runs/segment/tree_seg/weights/best.pt"
    )


def validate_model(model_path: str):
    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML_PATH)
    logger.info(f"Validation Results:")
    logger.info(f"Box mAP@0.5: {metrics.box.map50}")
    logger.info(f"Box mAP@0.5:0.95: {metrics.box.map}")
    logger.info(f"Seg mAP@0.5: {metrics.seg.map50}")
    logger.info(f"Seg mAP@0.5:0.95: {metrics.seg.map}")

    logger.info("Validation plots saved in runs/segment/val/")


def make_class_mask_for_image(
    annotation_data: dict, base_name: str, width: int, height: int
):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not annotation_data:
        return mask

    images = annotation_data.get("images", [])
    img_entry = None
    for im in images:
        if Path(im["file_name"]).stem == base_name:
            img_entry = im
            break
    if img_entry is None:
        return mask

    anns = img_entry.get("annotations", [])
    for ann in anns:
        cls = ann.get("class", "")
        if cls == "individual_tree":
            cls_id = 1
        elif cls == "group_of_trees":
            cls_id = 2
        else:
            continue

        segmentation = ann.get("segmentation", [])
        if not segmentation:
            continue
        pts = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
        pts = pts.reshape(-1, 1, 2)
        try:
            cv2.fillPoly(mask, [pts], color=int(cls_id))
        except Exception:
            continue
    return mask


def run_inference(model_path: str, source_path: str):
    """
    Run two independent inference pipelines and evaluations:
      1) YOLO segmentation pipeline (uses YOLOv8 segmentation masks / boxes)
      2) DeepForest -> SAM2 prompting pipeline (DeepForest boxes -> SAM2 masks)

    Produces per-pipeline overlays and evaluation_summary_<pipeline>.json files
    under runs/segment/generated/<pipeline>/
    """
    if not os.path.exists(source_path):
        logger.warning(f"Source path not found: {source_path}. Skipping inference.")
        return

    base_output = Path("runs/segment/generated")
    yolo_out_dir = base_output / "yolo"
    sam2_df_out_dir = base_output / "sam2_deepforest"
    yolo_out_dir.mkdir(parents=True, exist_ok=True)
    sam2_df_out_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = YOLO(model_path)

    deepforest_model = deepforest()
    deepforest_model.use_release()

    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    sam2_config = os.path.join(os.getcwd(), "sam2.1_hiera_l.yaml")
    sam2_model = build_sam2(sam2_config, sam2_checkpoint)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    annotation_data = {}
    if os.path.exists(ANNOTATION_PATH):
        try:
            with open(ANNOTATION_PATH, "r") as f:
                annotation_data = json.load(f)
        except Exception as e:
            logger.warning(f"Couldn't load annotations from {ANNOTATION_PATH}: {e}")
            annotation_data = {}
    else:
        logger.warning(
            f"ANNOTATION_PATH {ANNOTATION_PATH} not found. Per-class GT won't be available."
        )

    image_paths = []
    if os.path.isdir(source_path):
        for ext in ["*.jpg", "*.png", "*.tif"]:
            image_paths.extend(sorted(Path(source_path).glob(ext)))
    else:
        image_paths = [Path(source_path)]

    eval_yolo = {
        "per_class_iou": {"individual_tree": [], "group_of_trees": []},
        "per_class_ap_iou_0_4": {"individual_tree": [], "group_of_trees": []},
    }
    eval_sam2 = {
        "per_class_iou": {"individual_tree": []},
        "per_class_ap_iou_0_4": {"individual_tree": []},
    }

    COLOR_RED = (0, 0, 255)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_PURPLE = (128, 0, 128)
    COLOR_GREEN = (0, 255, 0)

    legend_entries = [
        (COLOR_RED, "Predicted something (was background)"),
        (COLOR_ORANGE, "Predicted wrong class (pred != GT)"),
        (COLOR_PURPLE, "GT present, not predicted (miss)"),
        (COLOR_GREEN, "Correct prediction"),
    ]

    for img_path in image_paths:
        base_name = img_path.stem
        try:
            orig_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if orig_img is None:
                logger.warning(f"Could not load {img_path}. Skipping this image.")
                continue

            orig_h, orig_w = orig_img.shape[:2]
            logger.info(f"Processing image: {img_path.name} size: {orig_w}x{orig_h}")

            predicted_class_map_yolo = np.zeros((orig_h, orig_w), dtype=np.uint8)

            try:
                yolo_results = yolo_model.predict(
                    source=str(img_path), conf=0.5, iou=0.7, verbose=False
                )
                if not yolo_results:
                    yolo_res = None
                else:
                    yolo_res = yolo_results[0]
            except Exception as e:
                logger.warning(f"YOLO prediction failed on {img_path}: {e}")
                yolo_res = None

            if yolo_res is not None:

                masks_assigned = False
                try:
                    if hasattr(yolo_res, "masks") and yolo_res.masks is not None:

                        y_masks = yolo_res.masks.data.cpu().numpy()  # type: ignore
                        y_cls = (
                            yolo_res.boxes.cls.cpu().numpy()  # type: ignore
                            if hasattr(yolo_res, "boxes")
                            else np.zeros((y_masks.shape[0],))
                        )

                        for i_mask, mask_np in enumerate(y_masks):

                            if mask_np.shape != (orig_h, orig_w):
                                try:
                                    mask_rsz = cv2.resize(
                                        mask_np.astype(np.uint8),
                                        (orig_w, orig_h),
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                                except Exception:
                                    mask_rsz = (mask_np > 0.5).astype(np.uint8)
                            else:
                                mask_rsz = (mask_np > 0.5).astype(np.uint8)

                            cls_id = int(y_cls[i_mask]) if len(y_cls) > i_mask else 0
                            predicted_class_map_yolo[mask_rsz > 0] = int(cls_id) + 1
                        masks_assigned = True
                except Exception:
                    masks_assigned = False

                if not masks_assigned:
                    try:
                        if hasattr(yolo_res, "boxes") and yolo_res.boxes is not None:
                            boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()  # type: ignore
                            classes = yolo_res.boxes.cls.cpu().numpy()  # type: ignore
                            for i_box, box in enumerate(boxes_xyxy):
                                x1, y1, x2, y2 = box.astype(int)
                                cls_id = (
                                    int(classes[i_box]) if len(classes) > i_box else 0
                                )
                                cv2.rectangle(
                                    predicted_class_map_yolo,
                                    (x1, y1),
                                    (x2, y2),
                                    color=int(cls_id) + 1,
                                    thickness=cv2.FILLED,
                                )
                    except Exception as e:
                        logger.warning(f"YOLO fallback (rasterize boxes) failed: {e}")

            predicted_class_map_sam2 = np.zeros((orig_h, orig_w), dtype=np.uint8)

            try:
                df_preds = deepforest_model.predict_image(
                    path=str(img_path), return_plot=False
                )

                boxes = (
                    df_preds[["xmin", "ymin", "xmax", "ymax"]].values  # type: ignore
                    if not df_preds.empty  # type: ignore
                    else np.array([])
                )
            except Exception as e:
                logger.warning(f"DeepForest failed on {img_path}: {e}")
                boxes = np.array([])

            if boxes is not None and len(boxes) > 0:
                sam2_predictor.set_image(orig_img)
                for i_box, box in enumerate(boxes):
                    try:
                        box_prompt = box.astype(int)
                        masks, scores, logits = sam2_predictor.predict(box=box_prompt)
                        if masks is None or len(masks) == 0:
                            continue
                        best_mask = masks[np.argmax(scores)]

                        if best_mask.shape != (orig_h, orig_w):
                            best_mask = cv2.resize(
                                best_mask.astype(np.uint8),
                                (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST,
                            )

                        predicted_class_map_sam2[best_mask > 0] = 1
                    except Exception as e:
                        logger.debug(
                            f"SAM2 prediction failed for a box on {img_path}: {e}"
                        )
                        continue

            gt_class_mask = make_class_mask_for_image(
                annotation_data, base_name, orig_w, orig_h
            )

            def _make_overlay_and_save(predicted_class_map, out_path):
                tp_mask = (
                    (predicted_class_map != 0)
                    & (gt_class_mask != 0)
                    & (predicted_class_map == gt_class_mask)
                )
                fp_mask = (predicted_class_map != 0) & (gt_class_mask == 0)
                wrong_mask = (
                    (predicted_class_map != 0)
                    & (gt_class_mask != 0)
                    & (predicted_class_map != gt_class_mask)
                )
                fn_mask = (gt_class_mask != 0) & (predicted_class_map == 0)

                overlay_bgra = cv2.cvtColor(orig_img, cv2.COLOR_BGR2BGRA)  # type: ignore
                overlay_layer = np.zeros_like(overlay_bgra, dtype=np.uint8)
                mask_alpha = 0.45
                overlay_layer[tp_mask] = (
                    COLOR_GREEN[0],
                    COLOR_GREEN[1],
                    COLOR_GREEN[2],
                    int(255 * mask_alpha),
                )
                overlay_layer[fp_mask] = (
                    COLOR_RED[0],
                    COLOR_RED[1],
                    COLOR_RED[2],
                    int(255 * mask_alpha),
                )
                overlay_layer[wrong_mask] = (
                    COLOR_ORANGE[0],
                    COLOR_ORANGE[1],
                    COLOR_ORANGE[2],
                    int(255 * mask_alpha),
                )
                overlay_layer[fn_mask] = (
                    COLOR_PURPLE[0],
                    COLOR_PURPLE[1],
                    COLOR_PURPLE[2],
                    int(255 * mask_alpha),
                )

                base_bgra = overlay_bgra.astype(np.float32)
                layer = overlay_layer.astype(np.float32)
                alpha_layer = layer[..., 3:4] / 255.0
                comp = base_bgra.copy()
                comp[..., :3] = (1 - alpha_layer) * base_bgra[
                    ..., :3
                ] + alpha_layer * layer[..., :3]
                comp[..., 3] = 255.0
                comp = comp.astype(np.uint8)

                foreground_mask = (gt_class_mask != 0) | (predicted_class_map != 0)
                if np.any(foreground_mask):
                    correct_foreground = np.sum(
                        (predicted_class_map == gt_class_mask) & foreground_mask
                    )
                    total_foreground = np.sum(foreground_mask)
                    fg_accuracy = (correct_foreground / total_foreground) * 100.0
                else:
                    fg_accuracy = 100.0
                text = f"Acc (fg): {fg_accuracy:.2f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.6, min(orig_w, orig_h) / 1000.0)
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness
                )
                pad = 8
                rect_tl = (10, 10)
                rect_br = (10 + text_w + 2 * pad, 10 + text_h + 2 * pad)
                vis = comp.copy()
                rect_color = (0, 0, 0, 200)
                cv2.rectangle(vis, rect_tl, rect_br, rect_color, thickness=cv2.FILLED)
                text_org = (rect_tl[0] + pad, rect_tl[1] + text_h + pad - 4)
                cv2.putText(
                    vis,
                    text,
                    text_org,
                    font,
                    font_scale,
                    (255, 255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                if vis.shape[2] == 4:
                    cv2.imwrite(out_path, vis)
                else:
                    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_BGR2BGRA))

            try:
                for cls_id, cls_name in [(1, "individual_tree"), (2, "group_of_trees")]:
                    gt_cls = gt_class_mask == cls_id
                    pred_cls = predicted_class_map_yolo == cls_id
                    inter = np.sum(gt_cls & pred_cls)
                    uni = np.sum(gt_cls | pred_cls)
                    iou = inter / uni if uni > 0 else 0.0
                    eval_yolo["per_class_iou"][cls_name].append(iou)

                iou_threshold = 0.4
                for cls_id, cls_name in [(1, "individual_tree"), (2, "group_of_trees")]:
                    gt_instances = [
                        ann
                        for ann in annotation_data.get("images", [])
                        if Path(ann["file_name"]).stem == base_name
                        for ann in ann.get("annotations", [])
                        if ann.get("class") == cls_name
                    ]

                    pred_mask = predicted_class_map_yolo == cls_id

                    num_labels, labels_im = cv2.connectedComponents(
                        pred_mask.astype(np.uint8)
                    )
                    pred_instances = list(range(1, num_labels))
                    tp, fp, fn = 0, 0, len(gt_instances)
                    matched = set()
                    for lbl in pred_instances:
                        pred_inst_mask = labels_im == lbl
                        best_iou = 0.0
                        best_gt_idx = -1
                        for idx, gt_ann in enumerate(gt_instances):
                            gt_mask = np.zeros_like(pred_inst_mask, dtype=np.uint8)

                            seg = gt_ann.get("segmentation", [])
                            if seg:
                                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                                gt_poly = np.zeros_like(gt_mask, dtype=np.uint8)
                                cv2.fillPoly(gt_poly, [pts.reshape(-1, 1, 2)], 1)
                                inter = np.sum(pred_inst_mask & (gt_poly > 0))
                                uni = np.sum(pred_inst_mask | (gt_poly > 0))
                                iou = inter / uni if uni > 0 else 0.0
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = idx
                        if best_iou >= iou_threshold and best_gt_idx not in matched:
                            tp += 1
                            fn -= 1
                            matched.add(best_gt_idx)
                        else:
                            fp += 1
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    ap = (precision + recall) / 2 if (precision + recall) > 0 else 0.0
                    eval_yolo["per_class_ap_iou_0_4"][cls_name].append(ap)

                out_path_yolo = str(yolo_out_dir / f"{base_name}.png")
                _make_overlay_and_save(predicted_class_map_yolo, out_path_yolo)
                logger.info(f"Saved YOLO overlay for {base_name} at {out_path_yolo}")

            except Exception as e:
                logger.error(
                    f"YOLO evaluation/visualization failed on {base_name}: {e}"
                )

            try:

                gt_cls = gt_class_mask == 1
                pred_cls = predicted_class_map_sam2 == 1
                inter = np.sum(gt_cls & pred_cls)
                uni = np.sum(gt_cls | pred_cls)
                iou = inter / uni if uni > 0 else 0.0
                eval_sam2["per_class_iou"]["individual_tree"].append(iou)

                iou_threshold = 0.4
                gt_instances = [
                    ann
                    for ann in annotation_data.get("images", [])
                    if Path(ann["file_name"]).stem == base_name
                    for ann in ann.get("annotations", [])
                    if ann.get("class") == "individual_tree"
                ]

                num_labels, labels_im = cv2.connectedComponents(
                    (predicted_class_map_sam2 == 1).astype(np.uint8)
                )
                pred_instances = list(range(1, num_labels))
                tp, fp, fn = 0, 0, len(gt_instances)
                matched = set()
                for lbl in pred_instances:
                    pred_inst_mask = labels_im == lbl
                    best_iou = 0.0
                    best_gt_idx = -1
                    for idx, gt_ann in enumerate(gt_instances):
                        seg = gt_ann.get("segmentation", [])
                        if seg:
                            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                            gt_poly = np.zeros_like(pred_inst_mask, dtype=np.uint8)
                            cv2.fillPoly(gt_poly, [pts.reshape(-1, 1, 2)], 1)
                            inter = np.sum(pred_inst_mask & (gt_poly > 0))
                            uni = np.sum(pred_inst_mask | (gt_poly > 0))
                            iou_inst = inter / uni if uni > 0 else 0.0
                            if iou_inst > best_iou:
                                best_iou = iou_inst
                                best_gt_idx = idx
                    if best_iou >= iou_threshold and best_gt_idx not in matched:
                        tp += 1
                        fn -= 1
                        matched.add(best_gt_idx)
                    else:
                        fp += 1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                ap = (precision + recall) / 2 if (precision + recall) > 0 else 0.0
                eval_sam2["per_class_ap_iou_0_4"]["individual_tree"].append(ap)

                out_path_sam2 = str(sam2_df_out_dir / f"{base_name}.png")
                _make_overlay_and_save(predicted_class_map_sam2, out_path_sam2)
                logger.info(
                    f"Saved SAM2+DeepForest overlay for {base_name} at {out_path_sam2}"
                )

            except Exception as e:
                logger.error(
                    f"SAM2+DeepForest evaluation/visualization failed on {base_name}: {e}"
                )

        except PermissionError as e:
            logger.error(f"Permission error for {img_path}: {e}. Skipping this image.")
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}. Skipping to next.")

    def _finalize_eval(eval_dict, out_json_path):
        summary = {}

        summary["per_class_iou"] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in eval_dict["per_class_iou"].items()
        }
        summary["per_class_ap_iou_0_4"] = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in eval_dict["per_class_ap_iou_0_4"].items()
        }

        summary["overall"] = {
            "mean_iou": (
                float(np.mean(list(summary["per_class_iou"].values())))
                if summary["per_class_iou"]
                else 0.0
            ),
            "mean_ap": (
                float(np.mean(list(summary["per_class_ap_iou_0_4"].values())))
                if summary["per_class_ap_iou_0_4"]
                else 0.0
            ),
        }
        with open(out_json_path, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Saved evaluation summary to {out_json_path}")

    _finalize_eval(eval_yolo, str(yolo_out_dir / "evaluation_summary_yolo.json"))
    _finalize_eval(
        eval_sam2, str(sam2_df_out_dir / "evaluation_summary_sam2_deepforest.json")
    )

    logger.info("Inference completed. See runs/segment/generated/ for outputs.")


def export_model(model_path: str, format_type: str = "onnx"):
    model = YOLO(model_path)
    exported_path = model.export(format=format_type)

    logger.info(f"Model exported to: {exported_path}")


DATA_DIR = "data"
RUN_DIR = "runs/segment"
IMG_DIR = os.path.join(DATA_DIR, "images")
OUT_MASK_DIR = os.path.join(DATA_DIR, "masks")
DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")
IMG_TRAIN_DIR = os.path.join(DATASET_DIR, "images", "train")
IMG_VAL_DIR = os.path.join(DATASET_DIR, "images", "val")
LABEL_TRAIN_DIR = os.path.join(DATASET_DIR, "labels", "train")
LABEL_VAL_DIR = os.path.join(DATASET_DIR, "labels", "val")
AUG_TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "images", "train", "aug")
AUG_TRAIN_LABEL_DIR = os.path.join(DATASET_DIR, "labels", "train", "aug")
AUG_TRAIN_MSK_DIR = os.path.join(DATASET_DIR, "masks", "train", "aug")
LOG_DIR = os.path.join(RUN_DIR, "logs")
GENERATED_DIR = os.path.join(RUN_DIR, "generated")

for dir_path in [
    DATA_DIR,
    RUN_DIR,
    GENERATED_DIR,
    LOG_DIR,
    IMG_DIR,
    OUT_MASK_DIR,
    DATASET_DIR,
    IMG_TRAIN_DIR,
    IMG_VAL_DIR,
    LABEL_TRAIN_DIR,
    LABEL_VAL_DIR,
    AUG_TRAIN_IMG_DIR,
    AUG_TRAIN_LABEL_DIR,
    AUG_TRAIN_MSK_DIR,
]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

ANNOTATION_PATH = os.path.join(DATA_DIR, "train_annotations.json")
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")
LOG_PATH = os.path.join(LOG_DIR, "training.log")
MODEL_PATH = "runs/segment/tree_seg/weights/best.pt"

Path(LOG_PATH).touch(exist_ok=True)

RANDOM_SEED = 42
FORCE_REPREP = False
REMOVE_TEMP_MASKS = True
DEBUG_MAX_IMAGES = None

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    # prep_dataset()
    # validate_dataset()

    # train_model()

    # validate_model(MODEL_PATH)

    run_inference(
        MODEL_PATH,
        IMG_VAL_DIR,
    )

    # export_model(MODEL_PATH, "onnx")


if __name__ == "__main__":
    main()
