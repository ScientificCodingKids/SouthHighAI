from diffusers import StableDiffusionPipeline
# import transformers as tf
#
# print(tf.__version__)
import torch
from uuid import uuid4
from typing import Optional, List
import sys

import pandas as pd

from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=2)
def get_pipeline_pair(device):
    ppl = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)  # ("CompVis/stable-diffusion-v1-4")
    #ppl = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    ppl_finetuned = StableDiffusionPipeline.from_pretrained(r"d:\dev\ai\diffusers\examples\text_to_image\sd-pokemon-model", torch_dtype=torch.float16)
    ppl = ppl.to(device)
    ppl_finetuned = ppl_finetuned.to(device)

    return ppl, ppl_finetuned


def txt_to_image(prompt: str, inp_trunk: str, device: Optional[str] = "cuda",
                 required_width: Optional[int] = 768, required_height: Optional[int] = 512
                 ) -> str:
    # for cuda, use py37hf env; for cpu, use py37rob2 env
    tic = pd.Timestamp.now()
    p = Path(__file__).parent.parent

    ppl, ppl_ft = get_pipeline_pair(device)

    generator = torch.Generator(device).manual_seed(1024)  # freeze the randomness
    res_ft = ppl_ft(prompt, height=512, width=512, generator=generator).images

    res = ppl(prompt, height=512, width=512, generator=generator).images

    uuid_str = str(uuid4())[0:8]
    out_fn = p / "outputs" / f"{inp_trunk}-{uuid_str}.jpg"
    out_fn_ft = p / "outputs" / f"{inp_trunk}-{uuid_str}-lora.jpg"

    res[0].save(out_fn)

    res_ft[0].save(out_fn_ft)

    toc = pd.Timestamp.now()

    print(f"{(toc-tic).total_seconds()} secs: {prompt}")

    return out_fn


def run():
    #print(txt_to_image("a photo of a horse riding an astronaut", "astronaut_rides_horse"))

    print(txt_to_image("green colored yoda", "yoda"))

    #print(txt_to_image("a photo of a teenager with a white bunny", "teenager_with_bunny"))

    # print(txt_to_image("vex long c-channel placed on a table", "c_channel"))
    print("ok")


if __name__ == "__main__":
    run()
