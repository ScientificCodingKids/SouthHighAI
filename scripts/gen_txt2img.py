# import requests
# import torch
# from PIL import Image
# from io import BytesIO
from uuid import uuid4
from typing import Optional, List
import sys
from diffusers import DiffusionPipeline
from pathlib import Path
from functools import lru_cache
import torch
from PIL import Image

SD_1_5 = "runwayml/stable-diffusion-v1-5"

FINETUNE_0 = r"d:\dev\ai\newer_colors\vrc_robot"



def image_grid(imgs, rows, cols):
    # copied from https://huggingface.co/blog/stable_diffusion
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

class Txt2ImgModel(object):
    def __init__(self, pretrained_path: str, device: Optional[str] = "cuda"):
        # device can be either "cuda" or "cpu"
        self.pipeline = DiffusionPipeline.from_pretrained(pretrained_path,  revision="fp16", torch_dtype=torch.float16).to(device)

    def gen(self, prompt: str, inp_trunk: Optional[str] = None, do_save: Optional[bool] = True):
        N = 4
        res = self.pipeline([prompt, ] * N, num_inference_steps=30).images

        grid = image_grid(res, 2, 2)

        if do_save:
            p = Path(__file__).parent.parent
            inp_trunk = inp_trunk or prompt[0:8].replace(" ", "_")
            out_fn = p / "outputs" / f"{inp_trunk}-{str(uuid4())[0:8]}.jpg"
            grid.save(out_fn)
            return out_fn
        else:
            return grid

vrc_prompts = [
    "vex robot, drivetrain with 4 wheels, a rotation arm with one motor",
]

if __name__ == "__main__":
    std_model = Txt2ImgModel(SD_1_5)
    ft_model = Txt2ImgModel(FINETUNE_0)

    prompt = vrc_prompts[0]

    std_model.gen(prompt)
    ft_model.gen(prompt)

