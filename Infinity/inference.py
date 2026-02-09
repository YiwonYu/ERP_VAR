import random
import os
import argparse
import torch
import cv2
import numpy as np
from tools.run_infinity import *
from contextlib import contextmanager
import gc

model_path = "../Infinity/weights/infinity_2b_reg.pth"
vae_path = "../Infinity/weights/infinity_vae_d32reg.pth"
text_encoder_ckpt = "../Infinity/weights/flan-t5-xl-official"
default_prompt = (
    "A 360° equirectangular panorama (ERP), seamless left-right wrap, stable poles. "
    "A high-altitude mountain valley at sunset, glowing clouds, distant snow peaks "
    "forming a continuous horizon, river winding through pine forest, warm rim light, "
    "natural colors, ultra detailed, photorealistic."
)

parser = argparse.ArgumentParser()
parser.add_argument("--pn", default="1M", help="1M, 0.60M, 0.25M, 0.06M")
parser.add_argument("--model_path", default=model_path)
parser.add_argument("--cfg_insertion_layer", type=int, default=0)
parser.add_argument("--vae_type", type=int, default=32)
parser.add_argument("--vae_path", default=vae_path)
parser.add_argument("--add_lvl_embeding_only_first_block", type=int, default=1)
parser.add_argument("--use_bit_label", type=int, default=1)
parser.add_argument("--model_type", default="infinity_2b")
parser.add_argument("--rope2d_each_sa_layer", type=int, default=1)
parser.add_argument("--rope2d_normalized_by_hw", type=int, default=2)
parser.add_argument("--use_scale_schedule_embedding", type=int, default=0)
parser.add_argument("--sampling_per_bits", type=int, default=1)
parser.add_argument("--text_encoder_ckpt", default=text_encoder_ckpt)
parser.add_argument("--text_channels", type=int, default=2048)
parser.add_argument("--apply_spatial_patchify", type=int, default=0)
parser.add_argument("--h_div_w_template", type=float, default=1.000)
parser.add_argument("--use_flex_attn", type=int, default=0)
parser.add_argument("--cache_dir", default="/dev/shm")
parser.add_argument("--checkpoint_type", default="torch")
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--bf16", type=int, default=1)
parser.add_argument("--save_file", default="tmp.jpg")
parser.add_argument("--prompt", default=default_prompt)
parser.add_argument("--cfg", type=float, default=4)
parser.add_argument("--tau", type=float, default=0.5)
parser.add_argument("--h_div_w", type=float, default=0.5)
parser.add_argument("--enable_positive_prompt", type=int, default=0)
args = parser.parse_args()

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load vae
vae = load_visual_tokenizer(args)
# load infinity
infinity = load_transformer(vae, args)

# 16GB memo
prompt = args.prompt
cfg = args.cfg
tau = args.tau
h_div_w = args.h_div_w  # aspect ratio, height:width
seed = args.seed if args.seed >= 0 else random.randint(0, 10000)
enable_positive_prompt = args.enable_positive_prompt

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
print(scale_schedule)
torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


# memory consumption evaluation
@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')


with torch.inference_mode():
    with measure_peak_memory():
        for _ in range(10):
            start_event.record()
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt,
                g_seed=seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=cfg,
                tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=enable_positive_prompt,
            )


save_dir = osp.dirname(osp.abspath(args.save_file))
os.makedirs(save_dir, exist_ok=True)
cv2.imwrite(args.save_file, generated_image.cpu().numpy())
print(f'Save to {osp.abspath(args.save_file)}')



