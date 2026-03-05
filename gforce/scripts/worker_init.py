#!/usr/bin/env python3
"""
Worker initialization script for G-Force T4 VMs.

This script runs inside the Batch job container on the GPU worker.
It handles model caching, training, and inference tasks.

Note: ML dependencies (torch, diffusers, etc.) are pre-installed in the Docker image
for fast boot times. See Dockerfile in project root.
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


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment.
    
    Returns:
        HF token or None if not set
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        # Mask token for logging
        masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
        logger.info(f"HF_TOKEN found: {masked}")
    else:
        logger.info("No HF_TOKEN set - using open access models only")
    return token


def sync_model_from_gcs(
    repo_id: str,
    commit_hash: str | None = None,
) -> Path | None:
    """Sync model from GCS cache to local disk.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "user/model")
        commit_hash: Optional specific commit hash to sync
        
    Returns:
        Path to local model directory if found in cache, None otherwise
    """
    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        logger.warning("GCS_BUCKET not set, skipping GCS cache check")
        return None
    
    # Build GCS cache path
    safe_repo_id = repo_id.replace("/", "--")
    gcs_cache_prefix = f"cache/models/huggingface/{safe_repo_id}"
    
    if commit_hash:
        gcs_path = f"gs://{bucket_name}/{gcs_cache_prefix}/{commit_hash}/"
        local_path = LOCAL_CACHE_PATH / f"{safe_repo_id}_{commit_hash}"
    else:
        # Try to find the latest cached version
        gcs_path = f"gs://{bucket_name}/{gcs_cache_prefix}/"
        local_path = LOCAL_CACHE_PATH / safe_repo_id
    
    # Check if cache exists in GCS
    logger.info(f"Checking GCS cache: {gcs_path}")
    result = subprocess.run(
        ["gsutil", "-q", "ls", gcs_path],
        capture_output=True,
    )
    
    if result.returncode != 0:
        logger.info(f"Model not found in GCS cache: {repo_id}")
        return None
    
    # Sync from GCS to local
    logger.info(f"Syncing model from GCS cache to {local_path}...")
    local_path.mkdir(parents=True, exist_ok=True)
    
    sync_result = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_path, str(local_path)],
        capture_output=True,
        text=True,
    )
    
    if sync_result.returncode != 0:
        logger.error(f"Failed to sync from GCS: {sync_result.stderr}")
        return None
    
    logger.info(f"✓ Model synced from GCS cache to {local_path}")
    return local_path


def sync_model_to_gcs(
    local_path: Path,
    repo_id: str,
    commit_hash: str,
) -> bool:
    """Sync downloaded model to GCS cache.
    
    Args:
        local_path: Local path to the model
        repo_id: HuggingFace repo ID
        commit_hash: Commit hash of the model version
        
    Returns:
        True if sync successful
    """
    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        logger.warning("GCS_BUCKET not set, skipping GCS cache upload")
        return False
    
    safe_repo_id = repo_id.replace("/", "--")
    gcs_path = f"gs://{bucket_name}/cache/models/huggingface/{safe_repo_id}/{commit_hash}/"
    
    logger.info(f"Uploading model to GCS cache: {gcs_path}")
    
    result = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", str(local_path), gcs_path],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to upload to GCS cache: {result.stderr}")
        return False
    
    # Save manifest
    manifest = {
        "provider": "huggingface",
        "repo_id": repo_id,
        "commit_hash": commit_hash,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gcs_path": gcs_path.rstrip("/"),
    }
    manifest_path = f"gs://{bucket_name}/cache/models/huggingface/{safe_repo_id}/{commit_hash}/manifest.json"
    
    manifest_result = subprocess.run(
        ["gsutil", "cp", "-", manifest_path],
        input=json.dumps(manifest, indent=2),
        capture_output=True,
        text=True,
    )
    
    if manifest_result.returncode == 0:
        logger.info(f"✓ Model cached to GCS: {gcs_path}")
    else:
        logger.warning(f"Model uploaded but manifest failed: {manifest_result.stderr}")
    
    return True


def get_model_from_cache_or_download(
    model_id: str,
    cache_path: Path,
) -> tuple[Path, bool]:
    """Get model from GCS cache or download from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID
        cache_path: Local path to check for cached model
        
    Returns:
        Tuple of (model_path, was_cached)
    """
    import torch
    from diffusers import StableDiffusionPipeline
    from huggingface_hub import HfApi, snapshot_download
    
    hf_token = get_hf_token()
    
    # Try GCS cache first
    gcs_cached_path = sync_model_from_gcs(model_id)
    if gcs_cached_path and gcs_cached_path.exists():
        logger.info(f"Using GCS-cached model: {gcs_cached_path}")
        return gcs_cached_path, True
    
    # Check local HF cache
    model_cache_key = model_id.replace("/", "--")
    local_model_path = cache_path / model_cache_key
    
    if local_model_path.exists():
        logger.info(f"Found local HF cache at {local_model_path}")
        return local_model_path, True
    
    # Download from HuggingFace
    logger.info(f"Downloading model {model_id} from HuggingFace...")
    
    try:
        # Get model info for caching
        api = HfApi(token=hf_token)
        model_info = api.model_info(model_id)
        commit_hash = model_info.sha
        logger.info(f"Model commit hash: {commit_hash}")
        
        # Download using snapshot_download for caching
        downloaded_path = snapshot_download(
            model_id,
            cache_dir=str(cache_path),
            token=hf_token,
            resume_download=True,
        )
        
        local_model_path = Path(downloaded_path)
        logger.info(f"Model downloaded to {local_model_path}")
        
        # Upload to GCS cache for future runs
        sync_model_to_gcs(local_model_path, model_id, commit_hash)
        
        return local_model_path, False
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


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

    # Get model from cache or download
    model_path, was_cached = get_model_from_cache_or_download(
        model_id, LOCAL_CACHE_PATH
    )
    logger.info(f"Using model from: {model_path} (cached: {was_cached})")

    # Load models with fp16
    logger.info("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_path,
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
    
    # Get model from cache or download
    if model_path and Path(model_path).exists():
        model_load_path = Path(model_path)
        logger.info(f"Using local model path: {model_load_path}")
    else:
        model_load_path, was_cached = get_model_from_cache_or_download(
            model_id, LOCAL_CACHE_PATH
        )
        logger.info(f"Using model from: {model_load_path} (cached: {was_cached})")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_load_path,
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
    
    # Log HF token status (masked)
    hf_token = get_hf_token()
    if hf_token:
        logger.info("HF_TOKEN: configured (gated models accessible)")
    else:
        logger.info("HF_TOKEN: not set (open models only)")

    # Check GCS mount
    if not GCS_MOUNT_PATH.exists():
        logger.error(f"GCS mount not found at {GCS_MOUNT_PATH}")
        return 1

    logger.info(f"GCS mount available at {GCS_MOUNT_PATH}")

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
