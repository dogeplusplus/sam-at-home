import os
import torch
import logging
import numpy as np
import gradio as gr

from einops import repeat
from skimage import color
from segment_anything import SamAutomaticMaskGenerator, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
checkpoint_name = None
available_models = [x for x in os.listdir("models") if x.endswith(".pth")]
default_model = available_models[0]
load_model(default_model)

with gr.Blocks() as application:
    with gr.Row():
        with gr.Column():
            gr.Markdown(value="# Segment Anything At Home")
            with gr.Row():
                selected_model = gr.Dropdown(choices=available_models, label="Model",
                                             value=default_model, interactive=True)
                selected_model.change(load_model, inputs=[selected_model])

            with gr.Row():
                image = gr.Image(source="upload", label="Input Image")

            with gr.Row():
                points_per_side = gr.Number(label="points_per_side", value=32, precision=0)
                points_per_batch = gr.Number(label="points_per_batch", value=64, precision=0)
                stability_score_offset = gr.Number(label="stability_score_offset", value=1)
                crop_n_layers = gr.Number(label="crop_n_layers", minimum=0, precision=0)
                crop_n_points_downscale_factor = gr.Number(label="crop_n_points_downscale_factor", value=1, precision=0)
                min_mask_region_area = gr.Number(label="min_mask_region_area", precision=0, value=0)

            with gr.Column():
                pred_iou_thresh = gr.Slider(label="pred_iou_thresh", minimum=0, maximum=1, value=0.88, step=0.01)
                stability_score_thresh = gr.Slider(label="stability_score_thresh",
                                                   minimum=0, maximum=1, value=0.95, step=0.01)
                box_nms_thresh = gr.Slider(label="box_nms_thresh", minimum=0, maximum=1, value=0.7)
                crop_nms_thresh = gr.Slider(label="crop_nms_thresh", minimum=0, maximum=1, value=0.7)
                crop_overlap_ratio = gr.Slider(label="crop_overlap_ratio", minimum=0,
                                               maximum=1, value=512 / 1500, step=0.01)

        with gr.Column():
            output = gr.Image(interactive=False, label="Segmentation Map")
            annotation_masks = gr.Gallery(label="Segment Images", inputs=[image, output], live=True)

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
    ], outputs=[output, annotation_masks])


application.launch()
