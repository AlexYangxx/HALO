import numpy as np
import torch
import gradio as gr
from PIL import Image
from net.CIDNet import CIDNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import imquality.brisque as brisque
from loss.niqe_utils import *
import argparse

opt_parser = argparse.ArgumentParser(description='App')
opt_parser.add_argument('--cpu', action='store_true', help='CPU-Only')
opt = opt_parser.parse_args()

device = torch.device('cpu' if opt.cpu or not torch.cuda.is_available() else 'cuda')
eval_net = CIDNet().to(device)

eval_net.trans.gated = True
eval_net.trans.gated2 = True


def process_image(input_img, score, model_path, gamma, alpha_s=1.0, alpha_i=1.0):
    torch.set_grad_enabled(False)
    ckpt = os.path.join(directory, model_path)
    try:
        sd = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(ckpt, map_location=device)
    eval_net.load_state_dict(sd)
    eval_net.eval()

    pil2tensor = transforms.Compose([transforms.ToTensor()])
    input = pil2tensor(input_img)
    factor = 8
    h, w = input.shape[1], input.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input = F.pad(input.unsqueeze(0), (0, padw, 0, padh), 'reflect').to(device)
    with torch.no_grad():
        eval_net.trans.alpha_s = alpha_s
        eval_net.trans.alpha = alpha_i
        output = eval_net(input ** gamma)

    output = torch.clamp(output, 0, 1)
    output = output[:, :, :h, :w]
    enhanced_img = transforms.ToPILImage()(output.squeeze(0).cpu())
    if score == 'Yes':
        im1 = enhanced_img.convert('RGB')
        score_brisque = brisque.score(im1)
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)
        return enhanced_img, score_niqe, score_brisque
    else:
        return enhanced_img, 0, 0


def find_pth_files(directory):
    pth_files = []
    for root, dirs, files in os.walk(directory):
        if 'train' in root.split(os.sep):
            continue
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(root, file))
    return pth_files


def pth_paths_relative_to(root, paths):
    return [os.path.relpath(p, root).replace("\\", "/") for p in paths]


directory = "weights"
pth_files = find_pth_files(directory)
pth_files2 = pth_paths_relative_to(directory, pth_files)

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="Low-light Image", type="pil"),
        gr.Radio(choices=['Yes', 'No'], label="Image Score", info='Calculate NIQE and BRISQUE, default is "No".'),
        gr.Radio(choices=pth_files2, label="Model Weights", info='Choose your model. The best models are "SICE.pth" and "generalization.pth".'),
        gr.Slider(0.1, 5, label="gamma curve", step=0.01, value=1.0, info="Lower is lighter, and best range is [0.5,2.5]."),
        gr.Slider(0, 2, label="Alpha-s", step=0.01, value=1.0, info="Higher is more saturated."),
        gr.Slider(0.1, 2, label="Alpha-i", step=0.01, value=1.0, info="Higher is lighter.")
    ],
    outputs=[
        gr.Image(label="Result", type="pil"),
        gr.Textbox(label="NIQE", info="Lower is better."),
        gr.Textbox(label="BRISQUE", info="Lower is better.")
    ],
    title="HVI-CIDNet (Low-Light Image Enhancement)",
    allow_flagging="never"
)

interface.launch(server_port=7862)
