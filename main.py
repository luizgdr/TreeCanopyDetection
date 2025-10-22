import json
import os
import shutil
import cv2
import numpy as np
import random
from pathlib import Path
from typing import Any, Iterable
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]


def prep_dataset():
    if os.path.exists(DATA_YAML_PATH) and not FORCE_REPREP:
        print(
            "Dataset already exists! Skipping data preparation. Set FORCE_REPREP=True to regenerate."
        )
    else:
        print("Preparing dataset...")

        def points_from_segmentation(segmentation: Iterable[float]) -> NDArray[Any]:
            pts = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
            return pts.astype(np.int32)

        def make_mask(
            shape_hw: tuple[int, int], annotations: list
        ) -> NDArray[np.uint8]:
            h, w = shape_hw
            mask = np.zeros((h, w), dtype=np.uint8)

            if not annotations:
                return mask

            for ann in annotations:
                segmentation = ann["segmentation"]
                pts = points_from_segmentation(segmentation).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts], 255)

            return mask

        with open(ANNOTATION_PATH, "r") as file:
            annotation_data = json.load(file)

        images_list = annotation_data["images"]
        train_img_data, val_img_data = train_test_split(
            images_list, test_size=0.2, random_state=RANDOM_SEED
        )

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

            for i, (y_start, y_end, x_start, x_end, quad_name) in enumerate(quadrants):
                quad_img = img_bgr[y_start:y_end, x_start:x_end]
                quad_h, quad_w = quad_img.shape[:2]

                quad_mask = mask[y_start:y_end, x_start:x_end]

                quad_annotations = []
                for ann in annotations:
                    segmentation = ann["segmentation"]
                    pts = np.array(segmentation, dtype=np.float32).reshape(-1, 2)

                    pts[:, 0] -= x_start
                    pts[:, 1] -= y_start

                    pts[:, 0] = np.clip(pts[:, 0], 0, quad_w)
                    pts[:, 1] = np.clip(pts[:, 1], 0, quad_h)

                    if len(pts) > 2 and np.any(
                        (pts[:, 0] >= 0)
                        & (pts[:, 0] <= quad_w)
                        & (pts[:, 1] >= 0)
                        & (pts[:, 1] <= quad_h)
                    ):
                        adjusted_ann = ann.copy()
                        adjusted_ann["segmentation"] = pts.flatten().tolist()
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

        print(f"Generated {aug_count} quadrant samples for train.")

        if REMOVE_TEMP_MASKS:
            shutil.rmtree(AUG_TRAIN_MSK_DIR)
            print("Temp augmented masks cleaned up.")

        with open(DATA_YAML_PATH, "w") as f:
            f.write(
                f"path: {os.path.abspath(DATASET_DIR)}\n"
                + "train: images/train\n"
                + "val: images/val\n\n"
                + "nc: 2\n"
                + "names: ['individual_tree', 'group_of_trees']\n"
            )

        print(f"data.yaml created at: {DATA_YAML_PATH}")

    print("Data prep complete or skipped. Ready for training/validation/inference.")


def validate_dataset():
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
    model = YOLO("yolov8n-seg.pt")

    device = "cpu"
    if torch.cuda.is_available():
        print("CUDA is available! Using CUDA on training.")
        device = 0

    model.train(
        data=DATA_YAML_PATH,
        epochs=100,
        imgsz=768,
        batch=4,
        name="tree_seg",
        seed=RANDOM_SEED,
        patience=10,
        save=True,
        plots=True,
        device=device,
        workers=4,
        augment=True,
    )

    print(
        "Training completed. Best model saved at: runs/segment/tree_seg/weights/best.pt"
    )


def validate_model(model_path: str):
    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML_PATH)
    print(f"Validation Results:")
    print(f"Box mAP@0.5: {metrics.box.map50}")
    print(f"Box mAP@0.5:0.95: {metrics.box.map}")
    print(f"Seg mAP@0.5: {metrics.seg.map50}")
    print(f"Seg mAP@0.5:0.95: {metrics.seg.map}")

    print("Validation plots saved in runs/segment/val/")


# def run_inference(model_path: str, source_path: str):
#     if not os.path.exists(source_path):
#         print(f"Source path not found: {source_path}. Skipping inference.")
#         return

#     output_dir = "runs/segment/generated"
#     os.makedirs(output_dir, exist_ok=True)

#     model = YOLO(model_path)

#     results = model.predict(
#         source=source_path,
#         save=True,
#         save_txt=True,
#         save_conf=True,
#         conf=0.5,
#         iou=0.7,
#         imgsz=640,
#         verbose=True,
#         device=0 if torch.cuda.is_available() else "cpu",
#     )

#     if not isinstance(results, list):
#         print("No results to process. Ensure the source path is valid.")
#         return

#     for idx, result in enumerate(results):
#         img_path = result.path
#         base_name = Path(img_path).stem
#         try:
#             orig_img = cv2.imread(img_path)
#             if orig_img is None:
#                 print(f"Could not load {img_path}. Skipping this image.")
#                 continue

#             orig_h, orig_w = orig_img.shape[:2]
#             print(
#                 f"Processing image: {os.path.basename(img_path)} with size: {orig_w}x{orig_h}"
#             )

#             masks = result.masks.data  # type: ignore
#             if masks is not None and len(masks) > 0:
#                 combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

#                 total_tree_pixels = 0
#                 for i, mask in enumerate(masks):
#                     tree_mask = mask.cpu().numpy().astype(np.uint8)

#                     tree_mask_resized = cv2.resize(
#                         tree_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
#                     ).astype(np.uint8)

#                     combined_mask = cv2.bitwise_or(
#                         combined_mask, tree_mask_resized * 255
#                     )

#                     tree_pixels = np.sum(tree_mask_resized > 0)
#                     total_tree_pixels += tree_pixels

#                     if result.boxes is not None and i < len(result.boxes.conf):
#                         conf = result.boxes.conf[i].item()
#                         print(
#                             f"Tree {i} in {os.path.basename(img_path)}: Canopy area = {tree_pixels} pixels, Confidence = {conf:.2f}"
#                         )
#                     else:
#                         print(
#                             f"Tree {i} in {os.path.basename(img_path)}: Canopy area = {tree_pixels} pixels"
#                         )

#                 combined_mask_path = os.path.join(
#                     output_dir, f"{base_name}_combined_mask.png"
#                 )
#                 cv2.imwrite(combined_mask_path, combined_mask)
#                 print(f"Saved combined mask for {base_name} at {combined_mask_path}")

#                 overlay = orig_img.copy()
#                 overlay_bgra = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
#                 tree_areas = combined_mask > 0
#                 overlay_bgra[tree_areas] = [
#                     0,
#                     255,
#                     0,
#                     128,
#                 ]

#                 overlay_path = os.path.join(
#                     output_dir, f"{base_name}_semi_transparent_overlay.png"
#                 )
#                 cv2.imwrite(overlay_path, overlay_bgra)
#                 print(
#                     f"Saved semi-transparent overlay for {base_name} at {overlay_path}"
#                 )

#                 canopy_coverage = (total_tree_pixels / (orig_h * orig_w)) * 100
#                 print(
#                     f"Total tree canopy coverage for {base_name}: {canopy_coverage:.2f}%"
#                 )

#             polygons = result.masks.xy  # type: ignore
#             for i, poly in enumerate(polygons):
#                 if len(poly) > 0:
#                     poly_area = cv2.contourArea(poly.astype(np.int32))
#                     print(f"Tree {i} polygon area in {base_name}: {poly_area} pixels")

#         except PermissionError as e:
#             print(f"Permission error for {img_path}: {e}. Skipping this image.")
#         except Exception as e:
#             print(f"Error processing {img_path}: {e}. Skipping to next.")

#     print(
#         "Inference outputs saved in runs/segment/predict/ and custom files in runs/segment/generated/"
#     )


def run_inference(model_path: str, source_path: str):
    if not os.path.exists(source_path):
        print(f"Source path not found: {source_path}. Skipping inference.")
        return

    output_dir = "runs/segment/generated"
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    # Load annotations JSON so we can rasterize per-class GT masks
    annotation_data = {}
    if os.path.exists(ANNOTATION_PATH):
        try:
            with open(ANNOTATION_PATH, "r") as f:
                annotation_data = json.load(f)
        except Exception as e:
            print(f"Warning: couldn't load annotations from {ANNOTATION_PATH}: {e}")
            annotation_data = {}
    else:
        print(f"Warning: ANNOTATION_PATH {ANNOTATION_PATH} not found. Per-class GT won't be available.")

    def make_class_mask_for_image(base_name: str, width: int, height: int):
        """
        Returns a HxW uint8 mask with:
          0 => background
          1 => individual_tree
          2 => group_of_trees
        """
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

    results = model.predict(
        source=source_path,
        save=False,
        save_txt=False,
        conf=0.5,
        iou=0.7,
        imgsz=640,
        verbose=True,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    if not isinstance(results, list):
        print("No results to process. Ensure the source path is valid.")
        return

    # BGR colors for swatches (OpenCV uses BGR)
    COLOR_RED = (0, 0, 255)       # false positive
    COLOR_ORANGE = (0, 165, 255)  # wrong class
    COLOR_PURPLE = (128, 0, 128)  # false negative
    COLOR_GREEN = (0, 255, 0)     # correct guess

    legend_entries = [
        (COLOR_RED, "Predicted something (was background)"),
        (COLOR_ORANGE, "Predicted wrong class (pred != GT)"),
        (COLOR_PURPLE, "GT present, not predicted (miss)"),
        (COLOR_GREEN, "Correct prediction"),
    ]

    for idx, result in enumerate(results):
        img_path = result.path
        base_name = Path(img_path).stem
        try:
            orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if orig_img is None:
                print(f"Could not load {img_path}. Skipping this image.")
                continue

            orig_h, orig_w = orig_img.shape[:2]

            gt_class_mask = make_class_mask_for_image(base_name, orig_w, orig_h)

            predicted_class_map = np.zeros((orig_h, orig_w), dtype=np.uint8)
            pred_masks = result.masks.data if hasattr(result.masks, "data") else None # type: ignore

            pred_confs = []
            pred_cls_ids = []
            if result.boxes is not None:
                try:
                    pred_cls_ids = [int(x) for x in result.boxes.cls.tolist()]
                except Exception:
                    pred_cls_ids = [int(x.item()) for x in result.boxes.cls]
                try:
                    pred_confs = [float(x) for x in result.boxes.conf.tolist()]
                except Exception:
                    pred_confs = [float(x.item()) for x in result.boxes.conf]
            else:
                pred_cls_ids = []
                pred_confs = []

            mask_entries = []
            if pred_masks is not None:
                n_masks = len(pred_masks)
                for i in range(n_masks):
                    conf = pred_confs[i] if i < len(pred_confs) else 1.0
                    cls_id = pred_cls_ids[i] if i < len(pred_cls_ids) else 0
                    mask_entries.append((i, conf, cls_id))
                mask_entries.sort(key=lambda x: x[1], reverse=True)

                for i, conf, cls_id in mask_entries:
                    m = pred_masks[i].cpu().numpy().astype(np.uint8) # type: ignore
                    if m.shape[0] != orig_h or m.shape[1] != orig_w:
                        try:
                            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        except Exception:
                            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    mask_pixels = m > 0
                    pred_class_value = int(cls_id) + 1
                    predicted_class_map[mask_pixels] = pred_class_value

            tp_mask = (predicted_class_map != 0) & (gt_class_mask != 0) & (predicted_class_map == gt_class_mask)
            fp_mask = (predicted_class_map != 0) & (gt_class_mask == 0)
            wrong_mask = (predicted_class_map != 0) & (gt_class_mask != 0) & (predicted_class_map != gt_class_mask)
            fn_mask = (gt_class_mask != 0) & (predicted_class_map == 0)

            overlay_bgra = cv2.cvtColor(orig_img, cv2.COLOR_BGR2BGRA)
            overlay_layer = np.zeros_like(overlay_bgra, dtype=np.uint8)

            mask_alpha = 0.45
            overlay_layer[tp_mask] = (COLOR_GREEN[0], COLOR_GREEN[1], COLOR_GREEN[2], int(255 * mask_alpha))
            overlay_layer[fp_mask] = (COLOR_RED[0], COLOR_RED[1], COLOR_RED[2], int(255 * mask_alpha))
            overlay_layer[wrong_mask] = (COLOR_ORANGE[0], COLOR_ORANGE[1], COLOR_ORANGE[2], int(255 * mask_alpha))
            overlay_layer[fn_mask] = (COLOR_PURPLE[0], COLOR_PURPLE[1], COLOR_PURPLE[2], int(255 * mask_alpha))

            base_bgra = overlay_bgra.astype(np.float32)
            layer = overlay_layer.astype(np.float32)
            alpha_layer = (layer[..., 3:4] / 255.0)
            comp = base_bgra.copy()
            comp[..., :3] = (1 - alpha_layer) * base_bgra[..., :3] + alpha_layer * layer[..., :3]
            comp[..., 3] = 255.0
            comp = comp.astype(np.uint8)

            foreground_mask = (gt_class_mask != 0) | (predicted_class_map != 0)
            if np.any(foreground_mask):
                correct_foreground = np.sum((predicted_class_map == gt_class_mask) & foreground_mask)
                total_foreground = np.sum(foreground_mask)
                fg_accuracy = (correct_foreground / total_foreground) * 100.0
            else:
                fg_accuracy = 100.0

            # Put accuracy text top-left (with small background)
            text = f"Acc (fg): {fg_accuracy:.2f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.6, min(orig_w, orig_h) / 1000.0)
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            pad = 8
            rect_tl = (10, 10)
            rect_br = (10 + text_w + 2 * pad, 10 + text_h + 2 * pad)
            vis = comp.copy()
            # draw semi-transparent rectangle for text background
            # we'll draw directly into vis (which is BGRA)
            rect_color = (0, 0, 0, 200)
            cv2.rectangle(vis, rect_tl, rect_br, rect_color, thickness=cv2.FILLED)
            text_org = (rect_tl[0] + pad, rect_tl[1] + text_h + pad - 4)
            cv2.putText(vis, text, text_org, font, font_scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)

            # --- Legend overlay (bottom-right) ---
            legend_margin = 12
            swatch_size = int(max(18, min(orig_w, orig_h) / 40))  # size of color box
            spacing = 6
            line_height = max(swatch_size, int(text_h * 0.9)) + spacing
            # build legend text sizes to compute width
            legend_font_scale = font_scale * 0.9
            legend_thickness = 1
            legend_text_sizes = [cv2.getTextSize(lbl, font, legend_font_scale, legend_thickness)[0] for (_, lbl) in legend_entries]
            legend_item_widths = [swatch_size + spacing + w for (w, _h) in legend_text_sizes]
            legend_w = max(legend_item_widths) + 2 * pad
            legend_h = len(legend_entries) * line_height + 2 * pad

            # position legend bottom-right
            legend_br_x = orig_w - legend_margin
            legend_br_y = orig_h - legend_margin
            legend_tl_x = legend_br_x - legend_w
            legend_tl_y = legend_br_y - legend_h
            legend_tl = (int(legend_tl_x), int(legend_tl_y))
            legend_br = (int(legend_br_x), int(legend_br_y))

            # Create legend layer (BGRA)
            legend_layer = np.zeros_like(vis, dtype=np.uint8)  # same size, zeros (transparent)
            # fill background rectangle with semi-transparent black
            bg_alpha = 0.75
            bg_color_bgra = (0, 0, 0, int(255 * bg_alpha))
            cv2.rectangle(legend_layer, legend_tl, legend_br, bg_color_bgra, thickness=cv2.FILLED)

            # draw each swatch + text
            y = legend_tl[1] + pad
            text_x = legend_tl[0] + pad + swatch_size + spacing
            swatch_x = legend_tl[0] + pad
            for i, (color_bgr, label) in enumerate(legend_entries):
                sw_y1 = int(y + i * line_height)
                sw_y2 = sw_y1 + swatch_size
                sw_x1 = swatch_x
                sw_x2 = sw_x1 + swatch_size
                # draw swatch rectangle (opaque in legend)
                cv2.rectangle(
                    legend_layer,
                    (sw_x1, sw_y1),
                    (sw_x2, sw_y2),
                    (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]), 255),
                    thickness=cv2.FILLED,
                )
                # draw label text (opaque)
                text_pos = (text_x, sw_y1 + swatch_size - 3)
                cv2.putText(
                    legend_layer,
                    label,
                    text_pos,
                    font,
                    legend_font_scale,
                    (255, 255, 255, 255),
                    legend_thickness,
                    cv2.LINE_AA,
                )

            # Composite legend_layer over vis according to legend_layer alpha channel
            legend_float = legend_layer.astype(np.float32)
            vis_float = vis.astype(np.float32)
            alpha_legend = (legend_float[..., 3:4] / 255.0)
            comp2 = vis_float.copy()
            comp2[..., :3] = (1 - alpha_legend) * vis_float[..., :3] + alpha_legend * legend_float[..., :3]
            comp2[..., 3] = 255.0
            comp2 = comp2.astype(np.uint8)

            out_path = os.path.join(output_dir, f"{base_name}.png")
            if comp2.shape[2] == 4:
                cv2.imwrite(out_path, comp2)
            else:
                cv2.imwrite(out_path, cv2.cvtColor(comp2, cv2.COLOR_BGR2BGRA))

        except PermissionError as e:
            print(f"Permission error for {img_path}: {e}. Skipping this image.")
        except Exception as e:
            print(f"Error processing {img_path}: {e}. Skipping to next.")

    print("Inference outputs (with legend) saved in runs/segment/generated/")


def export_model(model_path: str, format_type: str = "onnx"):
    model = YOLO(model_path)
    exported_path = model.export(format=format_type)
    print(f"Model exported to: {exported_path}")
    return exported_path


DATA_DIR = "data"
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

for dir_path in [
    DATA_DIR,
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

RANDOM_SEED = 42
FORCE_REPREP = False
REMOVE_TEMP_MASKS = True
DEBUG_MAX_IMAGES = None

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def main():
    # prep_dataset()
    # validate_dataset()

    # train_model()

    # validate_model("runs/segment/tree_seg/weights/best.pt")

    run_inference(
        "runs/segment/tree_seg/weights/best.pt",
        IMG_VAL_DIR,
    )

    exported_path = export_model("runs/segment/tree_seg/weights/best.pt", "onnx")

    print(f"Exported model to path '{exported_path}'")


if __name__ == "__main__":
    main()
