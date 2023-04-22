import gradio as gr
import numpy as np

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points


with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input")
        output_img = gr.Image(label="Selected Segment")

    with gr.Row():
        h = gr.Number()
        w = gr.Number()

    def get_select_coords(img, evt: gr.SelectData):
        return evt.index[1], evt.index[0]

    input_img.select(get_select_coords, [input_img,], [h, w])

if __name__ == "__main__":
    demo.launch()