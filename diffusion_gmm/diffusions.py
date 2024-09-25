import os
from typing import Optional, List

import numpy as np
import torch
from diffusers import (
    DDIMPipeline,  
    DiffusionPipeline,  
    StableDiffusionPipeline,
    DiTPipeline,
    DPMSolverMultistepScheduler
)
from tqdm.auto import tqdm

from diffusion_gmm.utils import save_and_plot_samples

# TODO: make class for diffusion models


def generate_ddpm_exposed(
    num_inference_steps: int,
    num_images: int,
    save_dir: str,
    plot_kwargs: dict = {},
    device="cpu",
) -> np.ndarray:
    """
    Generate images using the DDPM model from the Hugging Face Hub
    NOTE: can also drop-in replace with DDPM model
    This differs from generate_ddpm solely in that we expose the components more here

    Args:
        num_inference_steps (int): Number of inference steps to refine the samples
        num_images (int): Number of images to generate
        save_dir (str): Directory to save the generated images
        plot_kwargs (dict): Keyword arguments for saving and plotting the images
        device (str): Device to use for inference (default: 'cpu')

    """
    # Hugging Face hub directory from which we get the pre-trained CIFAR10 DDPM model's config
    model_id = "google/ddpm-cifar10-32"
    print("Model ID:", model_id)

    # Load the DDPM pipeline with the correct model and scheduler
    pipeline = DDIMPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)  # Move to the specified device

    # Generate a random initial sample in float16 to match the model's precision
    samples = torch.randn(
        num_images,
        pipeline.unet.config.in_channels,
        pipeline.unet.config.sample_size,
        pipeline.unet.config.sample_size,
        dtype=torch.float16,
    ).to(device)

    # Set the number of inference steps
    pipeline.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Iteratively refine the samples
    for t in tqdm(pipeline.scheduler.timesteps):
        # Predict the noise residual using autocast for mixed precision
        with torch.no_grad(), torch.cuda.amp.autocast():
            noise_pred = pipeline.unet(samples, t).sample

        # Compute the next less noisy image and update the samples
        samples = pipeline.scheduler.step(noise_pred, t, samples).prev_sample

    # Save the generated images
    print("Saving generated images")
    print("Sample shape:", samples.shape)

    # convert samples to numpy array
    samples_np = samples.cpu().numpy()
    save_and_plot_samples(
        samples_np,
        save_dir,
        **plot_kwargs,
    )

    return samples_np


def generate_image_DiffusionPipe(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    prompt: Optional[str] = None,  # "A photo of a cat, realistic style",
    seed: Optional[int] = None,
    num_images: int = 1,
    save_dir: str = "figs",
    plot_kwargs: dict = {},
) -> np.ndarray:
    """
    Generate images using a text-to-image diffusion model.

    Args:
        model_id (str): The model ID of the pretrained text-to-image diffusion model (default: "CompVis/stable-diffusion-v1-4").
        num_inference_steps (int): Number of denoising steps (default: 50).
        guidance_scale (float): Scale for classifier-free guidance (default: 7.5).
        height (int): Height of the image (default: 512).
        width (int): Width of the image (default: 512).
        prompt (Optional[str]): A text description for the image to generate (default: None).
        seed (Optional[int]): Random seed for reproducibility (default: None).
        num_images (int): Number of images to generate (default: 1).
        save_dir (str): Directory to save the generated images (default: "figs").
    """

    # Load the pretrained model
    pipeline = DiffusionPipeline.from_pretrained(model_id)

    # Move the pipeline to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)

    # Set the random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Generate the images
    samples = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_images=num_images,
    ).images

    # Save the generated images
    print("Saving generated images")
    print("Sample shape:", samples.shape)

    # convert samples to numpy array
    samples = samples.cpu().numpy()
    save_and_plot_samples(
        samples,
        save_dir,
        **plot_kwargs,
    )

    return samples


def generate_DiTPipe(
    class_ids: List[int] = [0],
    num_inference_steps: int = 50, # defaults to 50
    guidance_scale: float = 4.0, # default, high val: image closer to prompt, less image quality
    save_dir: str = "figs",
    plot_kwargs: dict = {},
    rseed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate images using the Diffusion Transformer model from the Hugging Face Hub
    """
    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # set random seed
    generator = torch.manual_seed(rseed)
    output = pipe(
        class_labels=class_ids, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale, 
        generator=generator,
        output_type='npy' # only does anything if 'pil' in which case it returns PIL images
        )
    
    # get the numpy images
    samples = output.images
    # print("saving sample intermediate")
    # np.save("sample_intermediate.npy", samples)
    print("Transposing samples")
    samples = samples.transpose(0, 3, 1, 2)

    # Save the generated images
    print("Saving generated images")
    print("Sample shape:", samples.shape)
    save_and_plot_samples(
        samples,
        save_dir,
        **plot_kwargs,
    )

    return samples


def generate_sb2(
    num_images: int = 1,
    guidance_scale: float = 7.5,
    prompt: str = "",
    save_dir: str = "figs",
    device: str = "cpu",
) -> None:
    """
    Generate images using the Stable Diffusion model from the Hugging Face Hub
    To perform unconditional generation, we can use an empty prompt
    Alternatively, you can manipulate the internal calls to skip text conditioning
    """
    # Load the Stable Diffusion model
    model_id = "stabilityai/stable-diffusion-2"

    # Initialize the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipeline = pipeline.to(device)  # Use GPU if available

    # prompt = "A photo of a cat sitting on a beach"
    images = pipeline([prompt] * num_images, guidance_scale=guidance_scale).images

    # Save the generated images
    for i, img in enumerate(images):
        img.save(os.path.join(save_dir, f"uncond_sb3_sample{i}.png"))


def ldm_pipeline(
    prompt: str = "A painting of a squirrel eating a burger",
    num_inference_steps: int = 50,
    guidance_scale: float = 6,
    save_dir: str = "figs",
    device: str = "cpu",
) -> None:
    """
    Generate images using the Large Diffusion Model from the Hugging Face Hub
    """
    # load model and scheduler
    pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
    pipeline = pipeline.to(device)

    # run pipeline in inference (sample random noise and denoise)
    prompt = "A painting of a squirrel eating a burger"
    images = pipeline(
        [prompt],
        num_inference_steps=num_inference_steps,
        eta=0.3,
        guidance_scale=guidance_scale,
    ).images

    # save images
    for idx, image in enumerate(images):
        image.save(os.path.join(save_dir, f"ldm_256_sample_{idx}.png"))
