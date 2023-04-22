import gradio as gr
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import tempfile
import os
from omegaconf import OmegaConf
from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama, build_lama_model, inpaint_img_with_builded_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)


def get_sam_feat(img):
    # predictor.set_image(img)
    model['sam'].set_image(img)
    features = model['sam'].features 
    orig_h = model['sam'].orig_h 
    orig_w = model['sam'].orig_w 
    input_h = model['sam'].input_h 
    input_w = model['sam'].input_w 
    return features, orig_h, orig_w, input_h, input_w

 
def get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w):
    point_coords = [w, h]
    point_labels = [1]
    dilate_kernel_size = 15

    # model['sam'].is_image_set = False
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w
    # model['sam'].image_embedding = image_embedding
    # model['sam'].original_size = original_size
    # model['sam'].input_size = input_size
    # model['sam'].is_image_set = True
    
    # model['sam'].set_image(img)
    # masks, _, _ = predictor.predict(
    masks, _, _ = model['sam'].predict(
        point_coords=np.array([point_coords]),
        point_labels=np.array(point_labels),
        multimask_output=True,
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
    # lama_ckpt = "pretrained_models/big-lama"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for mask in [mask0, mask1, mask2]:
        if len(mask.shape)==3:
            mask = mask[:,:,0]
        img_inpainted = inpaint_img_with_builded_lama(
            model['lama'], img, mask, lama_config, device=device)
        out.append(img_inpainted)
    return out


## build models
model = {}
# build the sam model
model_type="vit_h"
ckpt_p="pretrained_models/sam_vit_h_4b8939.pth"
model_sam = sam_model_registry[model_type](checkpoint=ckpt_p)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_sam.to(device=device)
# predictor = SamPredictor(model_sam)
model['sam'] = SamPredictor(model_sam)

# build the lama model
lama_config = "third_party/lama/configs/prediction/default.yaml"
lama_ckpt = "pretrained_models/big-lama"
device = "cuda" if torch.cuda.is_available() else "cpu"
# model_lama = build_lama_model(lama_config, lama_ckpt, device=device)
model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)


with gr.Blocks() as demo:
    features = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

    with gr.Row():
        img = gr.Image(label="Image")
        # img_pointed = gr.Image(label='Pointed Image')
        img_pointed = gr.Plot(label='Pointed Image')
        with gr.Column():
            with gr.Row():
                w = gr.Number(label="Point Coordinate W")
                h = gr.Number(label="Point Coordinate H")
            # sam_feat = gr.Button("Prepare for Segmentation")
            sam_mask = gr.Button("Predict Mask Using SAM")
            lama = gr.Button("Inpaint Image Using LaMA")

    # todo: maybe we can delete this row, for it's unnecessary to show the original mask for customers
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

    def get_select_coords(img, evt: gr.SelectData):
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [[evt.index[0], evt.index[1]]], [1],
                    size=(width*0.04)**2)
        return evt.index[0], evt.index[1], fig

    img.select(get_select_coords, [img], [w, h, img_pointed])
    # sam_feat.click(
    #     get_sam_feat,
    #     [img],
    #     []
    # )
    # img.change(get_sam_feat, [img], [])
    img.upload(get_sam_feat, [img], [features, orig_h, orig_w, input_h, input_w])

    sam_mask.click(
        get_masked_img,
        [img, w, h, features, orig_h, orig_w, input_h, input_w],
        [img_with_mask_0, img_with_mask_1, img_with_mask_2, mask_0, mask_1, mask_2]
    )

    lama.click(
        get_inpainted_img,
        [img, mask_0, mask_1, mask_2],
        [img_rm_with_mask_0, img_rm_with_mask_1, img_rm_with_mask_2]
    )


if __name__ == "__main__":
    demo.queue(concurrency_count=20, max_size=25)
    demo.launch(max_threads=40)
    