#!/usr/bin/env python3
"""
Worker initialization script for G-Force T4 VMs.

This script runs inside the Batch job container on the GPU worker.
It handles model caching, training, and inference tasks.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gforce-worker")

# Constants
GCS_MOUNT_PATH = Path("/mnt/disks/gcs")
LOCAL_CACHE_PATH = Path("/tmp/model-cache")
HF_CACHE_PATH = Path("/tmp/huggingface")
DEFAULT_MODEL = "TheImposterImposters/URPM-SD1.5-v2.3.inpainting"


def install_dependencies() -> None:
    """Install required ML dependencies."""
    deps = [
        "torch==2.1.2",
        "torchvision",
        "diffusers==0.25.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "xformers==0.0.23.post1",
        "bitsandbytes==0.41.3.post2",
        "safetensors>=0.4.0",
        "peft==0.7.1",
        "huggingface-hub>=0.20.0",
        "pillow>=10.0.0",
    ]

    logger.info("Installing dependencies...")
    for dep in deps:
        logger.info(f"Installing {dep}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", dep],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"Failed to install {dep}: {result.stderr}")
            raise RuntimeError(f"Failed to install {dep}")

    logger.info("Dependencies installed successfully")


def get_model_from_cache_or_download(
    model_id: str,
    cache_path: Path,
) -> Path:
    """Get model from GCS cache or download from HuggingFace.

    Args:
        model_id: HuggingFace model ID
        cache_path: Local path to check for cached model

    Returns:
        Path to the model
    """
    import torch
    from diffusers import StableDiffusionPipeline
    from huggingface_hub import HfApi

    # Check if we have a local cache
    model_cache_key = model_id.replace("/", "--")
    local_model_path = cache_path / model_cache_key

    if local_model_path.exists():
        logger.info(f"Found local cache at {local_model_path}")
        return local_model_path

    # Download from HuggingFace
    logger.info(f"Downloading model {model_id} from HuggingFace...")

    # Use fp16 and CPU offload for T4 compatibility during download
    dtype = torch.float16

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Save to local cache
    logger.info(f"Saving model to {local_model_path}")
    local_model_path.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(local_model_path)

    return local_model_path


def sync_model_to_gcs_cache(
    local_path: Path,
    gcs_cache_path: Path,
) -> None:
    """Sync downloaded model to GCS cache.

    Args:
        local_path: Local path to the model
        gcs_cache_path: GCS path for caching
    """
    logger.info(f"Syncing model to GCS cache at {gcs_cache_path}...")

    # Use gsutil for efficient sync
    result = subprocess.run(
        [
            "gsutil",
            "-m",
            "rsync",
            "-r",
            str(local_path),
            f"gs://{os.environ.get('GCS_BUCKET', 'gforce-assets')}/{gcs_cache_path}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(f"Failed to sync to GCS: {result.stderr}")
    else:
        logger.info("Model synced to GCS cache successfully")


def train_dreambooth(
    dataset_path: str,
    model_id: str,
    output_name: str,
    instance_prompt: str,
    num_steps: int = 1000,
) -> Path:
    """Run DreamBooth training.

    Args:
        dataset_path: Path to training images
        model_id: Base model ID
        output_name: Output checkpoint name
        instance_prompt: Instance prompt (e.g., "photo of sks person")
        num_steps: Number of training steps

    Returns:
        Path to the output checkpoint
    """
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from huggingface_hub import snapshot_download
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import CLIPTextModel, CLIPTokenizer

    logger.info("Starting DreamBooth training...")
    logger.info(f"Model: {model_id}")
    logger.info(f"Steps: {num_steps}")
    logger.info(f"Output: {output_name}")

    # Setup paths
    output_path = GCS_MOUNT_PATH / "outputs" / output_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    dataset_dir = Path(dataset_path.replace("gs://", "").split("/", 1)[1])
    if dataset_path.startswith("gs://"):
        local_dataset_dir = Path("/tmp/dataset")
        local_dataset_dir.mkdir(parents=True, exist_ok=True)
        # Dataset should already be mounted via GCS Fuse
        local_dataset_dir = GCS_MOUNT_PATH / dataset_dir
    else:
        local_dataset_dir = Path(dataset_path)

    # Initialize accelerator with fp16
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=4,
    )

    # Set seed for reproducibility
    set_seed(42)

    # Load models with fp16
    logger.info("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float16,
    )

    # Enable memory efficient attention
    if hasattr(unet, "set_attn_processor"):
        from diffusers.models.attention_processor import AttnProcessor
        unet.set_attn_processor(AttnProcessor())

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Use 8-bit Adam optimizer
    import bitsandbytes as bnb

    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(),
        lr=1e-6,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps,
    )

    # Prepare with accelerator
    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )

    # Training loop
    logger.info("Starting training loop...")
    unet.train()

    # Simple dataset class
    class DreamBoothDataset(Dataset):
        def __init__(
            self,
            dataset_dir: Path,
            instance_prompt: str,
            tokenizer: CLIPTokenizer,
            size: int = 512,
        ):
            self.dataset_dir = dataset_dir
            self.instance_prompt = instance_prompt
            self.tokenizer = tokenizer
            self.size = size
            self.instance_images = list(dataset_dir.glob("*.jpg")) + list(
                dataset_dir.glob("*.png")
            )

        def __len__(self) -> int:
            return len(self.instance_images)

        def __getitem__(self, index: int) -> dict[str, Any]:
            image = Image.open(self.instance_images[index]).convert("RGB")
            image = image.resize((self.size, self.size))

            # Tokenize prompt
            prompt = self.instance_prompt
            inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            return {
                "pixel_values": image,
                "input_ids": inputs.input_ids[0],
            }

    dataset = DreamBoothDataset(local_dataset_dir, instance_prompt, tokenizer)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    global_step = 0
    for epoch in range(1000):  # Large number, will break on step count
        for batch in dataloader:
            if global_step >= num_steps:
                break

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"]
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample timestep
                timesteps = torch.randint(
                    0,
                    1000,  # num_train_timesteps
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to latents
                noisy_latents = latents + noise * 0.5  # Simplified

                # Get text embeddings
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].unsqueeze(0)
                )[0]

                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                # Compute loss
                loss = torch.nn.functional.mse_loss(model_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0:
                logger.info(f"Step {global_step}/{num_steps}, Loss: {loss.item():.4f}")

        if global_step >= num_steps:
            break

    # Save checkpoint
    logger.info("Saving checkpoint...")
    unwrapped_unet = accelerator.unwrap_model(unet)

    # Save as safetensors
    from safetensors.torch import save_file

    checkpoint_path = output_path / f"{output_name}.safetensors"
    save_file(
        unwrapped_unet.state_dict(),
        str(checkpoint_path),
    )

    # Save metadata
    metadata = {
        "model_id": model_id,
        "instance_prompt": instance_prompt,
        "num_steps": num_steps,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training complete! Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def run_inference(
    prompt: str,
    model_path: str | None,
    num_images: int,
    output_prefix: str,
) -> list[Path]:
    """Run batch inference.

    Args:
        prompt: Generation prompt
        model_path: Path to model checkpoint (uses default if None)
        num_images: Number of images to generate
        output_prefix: Prefix for output files

    Returns:
        List of generated image paths
    """
    import torch
    from diffusers import StableDiffusionPipeline

    logger.info(f"Running inference for {num_images} images...")
    logger.info(f"Prompt: {prompt}")

    # Setup output path
    output_dir = GCS_MOUNT_PATH / "outputs" / output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_id = model_path or DEFAULT_MODEL
    logger.info(f"Loading model from {model_id}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Enable memory efficient attention
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Generate images
    generated_paths = []
    for i in range(num_images):
        logger.info(f"Generating image {i + 1}/{num_images}...")

        seed = 42 + i
        generator = torch.Generator(device=device).manual_seed(seed)

        image = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        output_path = output_dir / f"{output_prefix}_{i:04d}.png"
        image.save(output_path)
        generated_paths.append(output_path)

    logger.info(f"Generated {len(generated_paths)} images to {output_dir}")
    return generated_paths


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="G-Force Worker")
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        required=True,
        help="Operation mode",
    )
    parser.add_argument("--dataset", help="Path to training dataset")
    parser.add_argument("--model", help="Model ID or path")
    parser.add_argument("--output", help="Output name")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--prompt", help="Generation prompt")
    parser.add_argument("--num-images", type=int, default=10, help="Number of images")
    parser.add_argument("--output-prefix", default="gen", help="Output prefix")

    args = parser.parse_args()

    # Log environment info
    logger.info(f"G-Force Worker starting...")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"GCS Bucket: {os.environ.get('GCS_BUCKET', 'not set')}")

    # Check GCS mount
    if not GCS_MOUNT_PATH.exists():
        logger.error(f"GCS mount not found at {GCS_MOUNT_PATH}")
        return 1

    logger.info(f"GCS mount available at {GCS_MOUNT_PATH}")

    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return 1

    # Execute requested mode
    try:
        if args.mode == "train":
            if not args.dataset:
                logger.error("--dataset required for training mode")
                return 1

            instance_prompt = os.environ.get("INSTANCE_PROMPT", "")
            if not instance_prompt:
                logger.warning("INSTANCE_PROMPT not set, using default")
                instance_prompt = "photo of sks person"

            model_id = args.model or DEFAULT_MODEL
            output_name = args.output or f"checkpoint-{int(time.time())}"

            checkpoint_path = train_dreambooth(
                dataset_path=args.dataset,
                model_id=model_id,
                output_name=output_name,
                instance_prompt=instance_prompt,
                num_steps=args.steps,
            )

            logger.info(f"Training complete: {checkpoint_path}")

        elif args.mode == "inference":
            prompt = args.prompt or os.environ.get("GENERATION_PROMPT", "")
            if not prompt:
                logger.error("--prompt or GENERATION_PROMPT required for inference")
                return 1

            model_path = args.model or os.environ.get("MODEL_PATH")
            output_prefix = args.output_prefix or f"gen-{int(time.time())}"

            generated = run_inference(
                prompt=prompt,
                model_path=model_path,
                num_images=args.num_images,
                output_prefix=output_prefix,
            )

            logger.info(f"Generated {len(generated)} images")

    except Exception as e:
        logger.exception(f"Error in {args.mode} mode: {e}")
        return 1

    logger.info("G-Force Worker completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
