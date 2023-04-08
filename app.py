import os
import torch
import logging
import numpy as np
import gradio as gr

from einops import repeat
from skimage import color
from skimage.measure import label
from segment_anything import (
    SamAutomaticMaskGenerator,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    SamPredictor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find_dots(image, image_with_keypoints):
    # Check for diffs for any of (R, G, B)
    diff_mask = np.max(image != image_with_keypoints, axis=-1)
    # Assign each connected component as its own label
    diff_labels, num_labels = label(diff_mask, return_num=True)
    points = []
    # Find centre of mass of each keypoint
    for i in range(1, num_labels+1):
        com = np.argwhere(diff_labels == i).mean(axis=0)
        # predictor order is (x, y), numpy order is (y, x)
        points.append([com[1], com[0]])

    return np.array(points)


def create_color_mask(image, annotations):
    if len(annotations) == 0:
        return image

    sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
    mask = np.stack([x["segmentation"] for x in sorted_anns])
    mask = np.argmax(mask, axis=0)

    color_mask = color.label2rgb(mask, image)
    return color_mask


def extract_rgba_masks(image, annotations):
    image_segments = []
    for ann in annotations:
        segment = repeat(ann["segmentation"].astype(np.uint8) * 255, "h w -> h w 1")
        image_segment = np.concatenate([image, segment], axis=-1)
        image_segments.append(image_segment)

    return image_segments


def generate(
    image,
    points_per_side,
    points_per_batch,
    pred_iou_thresh,
    stability_score_thresh,
    stability_score_offset,
    box_nms_thresh,
    crop_n_layers,
    crop_nms_thresh,
    crop_overlap_ratio,
    crop_n_points_downscale_factor,
    min_mask_region_area,
    display_rgba_segments,
):
    global model
    generator = SamAutomaticMaskGenerator(
        model,
        points_per_side,
        points_per_batch,
        pred_iou_thresh,
        stability_score_thresh,
        stability_score_offset,
        box_nms_thresh,
        crop_n_layers,
        crop_nms_thresh,
        crop_overlap_ratio,
        crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )

    annotations = generator.generate(image)
    color_mask = create_color_mask(image, annotations)
    annotation_masks = []
    if display_rgba_segments:
        annotation_masks = extract_rgba_masks(image, annotations)
    return color_mask, annotation_masks


def load_model(name):
    global model, device, checkpoint_name
    if model is None or checkpoint_name != name:
        checkpoint_path = os.path.join("models", name)
        if "vit_b" in name:
            model = build_sam_vit_b(checkpoint_path)
        elif "vit_h" in name:
            model = build_sam_vit_h(checkpoint_path)
        elif "vit_l" in name:
            model = build_sam_vit_l(checkpoint_path)
        else:
            raise ValueError(f"Invalid checkpoint name: {name}")
        checkpoint_name = name
        model.to(device)
        logger.info(f"Loaded model: {checkpoint_name}")


def guided_prediction(image, fg_canvas, bg_canvas):
    global model
    fg_points = find_dots(image, fg_canvas)
    bg_points = find_dots(image, bg_canvas)

    if fg_points.size == 0:
        if bg_points.size == 0:
            point_coords = None
            point_labels = None
        else:
            point_coords = bg_points
            point_labels = np.zeros(len(bg_points))
    elif bg_points.size == 0:
        point_coords = fg_points
        point_labels = np.ones(len(fg_points))
    else:
        point_coords = np.concatenate([fg_points, bg_points])
        point_labels = np.concatenate([np.ones(len(fg_points)), np.zeros(len(bg_points))])

    predictor = SamPredictor(model)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    masks = masks.astype(int)
    # Assign each candidate mask a different color
    mask_colors = [[(1, 0, 0)], [(0, 1, 0)], [(0, 0, 1)]]
    color_masks = [color.label2rgb(masks[i], image, mc) for i, mc in enumerate(mask_colors)]
    return color_masks


def display_detected_keypoints(image, image_with_keypoints):
    diff_mask = np.max(image != image_with_keypoints, axis=-1) * 255
    _, num_labels = label(diff_mask, return_num=True)
    diff_mask = repeat(diff_mask, "... -> ... 3")
    return diff_mask, num_labels


available_models = [x for x in os.listdir("models") if x.endswith(".pth")]
default_model = available_models[0]
model = None
checkpoint_name = None
device = "cuda" if torch.cuda.is_available() else "cpu"
load_model(default_model)

with gr.Blocks() as application:
    gr.Markdown(value="# Segment Anything At Home")
    selected_model = gr.Dropdown(choices=available_models, label="Model",
                                 value=default_model, interactive=True)

    selected_model.change(load_model, inputs=[selected_model], show_progress=True)
    with gr.Tab("Automatic Segmentor"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image = gr.Image(
                        source="upload",
                        label="Input Image",
                        elem_id="image",
                        brush_radius=20,
                    )

                with gr.Row():
                    points_per_side = gr.Number(label="points_per_side", value=32, precision=0)
                    points_per_batch = gr.Number(label="points_per_batch", value=64, precision=0)
                    stability_score_offset = gr.Number(label="stability_score_offset", value=1)
                    crop_n_layers = gr.Number(label="crop_n_layers", precision=0)
                    crop_n_points_downscale_factor = gr.Number(
                        label="crop_n_points_downscale_factor", value=1, precision=0)
                    min_mask_region_area = gr.Number(label="min_mask_region_area", precision=0, value=0)

                with gr.Column():
                    pred_iou_thresh = gr.Slider(label="pred_iou_thresh", minimum=0, maximum=1, value=0.88, step=0.01)
                    stability_score_thresh = gr.Slider(label="stability_score_thresh",
                                                       minimum=0, maximum=1, value=0.95, step=0.01)
                    box_nms_thresh = gr.Slider(label="box_nms_thresh", minimum=0, maximum=1, value=0.7)
                    crop_nms_thresh = gr.Slider(label="crop_nms_thresh", minimum=0, maximum=1, value=0.7)
                    crop_overlap_ratio = gr.Slider(label="crop_overlap_ratio", minimum=0,
                                                   maximum=1, value=512 / 1500, step=0.01)

                    display_rgba_segments = gr.Checkbox(label="Extract RGBA image for each mask")

            with gr.Column():
                output = gr.Image(interactive=False, label="Segmentation Map")
                annotation_masks = gr.Gallery(label="Segment Images")

        submit = gr.Button("Submit")
        submit.click(generate, inputs=[
            image,
            points_per_side,
            points_per_batch,
            pred_iou_thresh,
            stability_score_thresh,
            stability_score_offset,
            box_nms_thresh,
            crop_n_layers,
            crop_nms_thresh,
            crop_overlap_ratio,
            crop_n_points_downscale_factor,
            min_mask_region_area,
            display_rgba_segments,
        ], outputs=[output, annotation_masks])

    with gr.Tab("Predictor"):
        with gr.Row():
            with gr.Column():
                base_image = gr.Image(label="Base Image", source="upload")

                with gr.Row():
                    fg_canvas = gr.Image(label="Foreground Keypoints", tool="color-sketch", brush_radius=30)
                    fg_keypoints = gr.Image(label="Detected Foreground Keypoints", interactive=False)

                with gr.Row():
                    bg_canvas = gr.Image(label="Background Keypoints", tool="color-sketch", brush_radius=30)
                    bg_keypoints = gr.Image(label="Detected Background Keypoints", interactive=False)

                def set_sketch_images(image):
                    return image, image

                base_image.upload(set_sketch_images, inputs=[base_image], outputs=[fg_canvas, bg_canvas])

                with gr.Row():
                    num_fg_keypoints = gr.Text(label="# Detected Foreground Keypoints", interactive=False)
                    num_bg_keypoints = gr.Text(label="# Detected Background Keypoints", interactive=False)

            pred_instructions = """
            ## Instructions
            The predictor tab uses the `SamPredictor` class interface for doing inference. \
            This exists to allow users to insert foreground/background keypoints into the \
            image to guide the segmentation.

            1. Upload the image using the `Base Image` canvas. This will auto-populate the \
            foreground/background canvas
            2. Add Foreground/Background keypoints by placing dots on the respective canvases.
            3. Click on the predict button.

            *Notes: Ensure that the color is not exactly the same as the pixels on the base image \
            where you are placing points. Any color or even multiple colors are allowed, so long \
            as it's different from the point on the base image. There's a heuristic algorithm that \
            locates the points by computing the diff of the base/keypoint images, taking the centre \
            of mass of each label. As such brush strokes/more complex shapes are treated as a single point.*
            """

            with gr.Column():
                gr.Markdown(pred_instructions)
                output_masks = gr.Gallery(label="Output Masks")

        predict = gr.Button("Predict")
        predict.click(
            guided_prediction,
            inputs=[base_image, fg_canvas, bg_canvas],
            outputs=output_masks,
        )
        predict.click(display_detected_keypoints, inputs=[base_image,
                      fg_canvas], outputs=[fg_keypoints, num_fg_keypoints])
        predict.click(display_detected_keypoints, inputs=[base_image,
                      bg_canvas], outputs=[bg_keypoints, num_bg_keypoints])

application.launch()
