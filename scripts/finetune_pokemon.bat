REM https://huggingface.co/docs/diffusers/v0.13.0/en/training/text2image
set MODEL_NAME="runwayml/stable-diffusion-v1-5"
set dataset_name="lambdalabs/pokemon-blip-captions"
REM output will be a subdir of diffusers/examples/text_to_image
set OUTPUT_DIR="lora_sd1.5_pokemon_15k_steps"

set PYTHONPATH=d:\dev\ai\transformers\src;d:\dev\ai\diffusers\src;%PYTHONPATH%

REM cd d:\dev\ai\diffusers\examples\text_to_image

accelerate launch d:\dev\ai\diffusers\examples\text_to_image\train_text_to_image.py ^
  --pretrained_model_name_or_path=%MODEL_NAME% ^
  --dataset_name=%dataset_name% ^
  --use_ema ^
  --resolution=512 --center_crop --random_flip ^
  --train_batch_size=1 ^
  --gradient_accumulation_steps=4 ^
  --gradient_checkpointing ^
  --checkpointing_steps=500 ^
  --mixed_precision="bf16" ^
  --max_train_steps=15000 ^
  --learning_rate=1e-05 ^
  --max_grad_norm=1 ^
  --lr_scheduler="constant" --lr_warmup_steps=0 ^
  --output_dir=%OUTPUT_DIR%