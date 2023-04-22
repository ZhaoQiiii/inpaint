import gradio as gr
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import tempfile
import os
from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from PIL import Image


def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)


def get_masked_img(img, w, h):
    point_labels = [1]
    point_coords = [w, h]
    dilate_kernel_size = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    masks, _, _ = predict_masks_with_sam(
        img,
        [point_coords],
        point_labels,
        model_type="vit_h",
        ckpt_p="pretrained_models/sam_vit_h_4b8939.pth",
        device=device,
    )

    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    else:
        masks = [mask for mask in masks]

    figs = []
    for idx, mask in enumerate(masks):
        # save the pointed and masked image
        tmp_p = mkstemp(".png")
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [point_coords], point_labels,
                    size=(width*0.04)**2)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
        figs.append(fig)
        plt.close()
    return *figs, *masks


def get_inpainted_img(img, mask0, mask1, mask2):
    lama_config = "third_party/lama/configs/prediction/default.yaml"
    lama_ckpt = "pretrained_models/big-lama"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for mask in [mask0, mask1, mask2]:
        if len(mask.shape)==3:
            mask = mask[:,:,0]
        img_inpainted = inpaint_img_with_lama(
            img, mask, lama_config, lama_ckpt, device=device)
        out.append(img_inpainted)
    return out


with gr.Blocks() as demo:
    with gr.Row():
        img = gr.Image(label="Image")
        with gr.Column():
            with gr.Row():
                w = gr.Number(label="Point Coordinate W")
                h = gr.Number(label="Point Coordinate H")
            sam = gr.Button("Predict Mask Using SAM")
            lama = gr.Button("Inpaint Image Using LaMA")

    with gr.Row():
        mask_0 = gr.outputs.Image(type="numpy", label="Segmentation Mask 0")
        mask_1 = gr.outputs.Image(type="numpy", label="Segmentation Mask 1")
        mask_2 = gr.outputs.Image(type="numpy", label="Segmentation Mask 2")

    with gr.Row():
        img_with_mask_0 = gr.Plot(label="Image with Segmentation Mask 0")
        img_with_mask_1 = gr.Plot(label="Image with Segmentation Mask 1")
        img_with_mask_2 = gr.Plot(label="Image with Segmentation Mask 2")

    with gr.Row():
        img_rm_with_mask_0 = gr.outputs.Image(
            type="numpy", label="Image Removed with Segmentation Mask 0")
        img_rm_with_mask_1 = gr.outputs.Image(
            type="numpy", label="Image Removed with Segmentation Mask 1")
        img_rm_with_mask_2 = gr.outputs.Image(
            type="numpy", label="Image Removed with Segmentation Mask 2")

    def get_select_coords(evt: gr.SelectData):
        return evt.index[0], evt.index[1]

    img.select(get_select_coords, [], [w, h])
    sam.click(
        get_masked_img,
        [img, w, h],
        [img_with_mask_0, img_with_mask_1, img_with_mask_2, mask_0, mask_1, mask_2]
    )

    lama.click(
        get_inpainted_img,
        [img, mask_0, mask_1, mask_2],
        [img_rm_with_mask_0, img_rm_with_mask_1, img_rm_with_mask_2]
    )


if __name__ == "__main__":
    demo.launch()