import torch
import numpy as np
import os
from tqdm.auto import tqdm
from PIL import Image
from typing import Optional, Tuple

from diffusion_gmm.utils import save_images_grid


from diffusers import StableDiffusionPipeline
def generate_sb2(
    num_images: int = 1,
    guidance_scale: float = 7.5,
    prompt: str = "",
    save_fig_dir: str = "figs",
    device: str = 'cpu'
) -> None:
    """
    Generate images using the Stable Diffusion model from the Hugging Face Hub
    To perform unconditional generation, we can use an empty prompt
    Alternatively, you can manipulate the internal calls to skip text conditioning
    """
    # Load the Stable Diffusion model
    model_id = "stabilityai/stable-diffusion-2"

    # Initialize the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)  # Use GPU if available

    # prompt = "A photo of a cat sitting on a beach"
    images = pipeline([prompt] * num_images, guidance_scale=guidance_scale).images

    # Save the generated images
    for i, img in enumerate(images):
        img.save(os.path.join(save_fig_dir, f"uncond_sb3_sample{i}.png"))

from diffusers import DDIMPipeline #, DDPMPipeline
def generate_ddpm_exposed(
    num_inference_steps: int = 50,
    num_images: int = 1,
    save_grid_shape: Optional[Tuple[int, int]] = None,
    save_fig_dir: str = "figs",
    device='cpu',
) -> None:
    """
    Generate images using the DDPM model from the Hugging Face Hub
    NOTE: can also drop-in replace with DDPM model
    This differs from generate_ddpm solely in that we expose the components more here
    """
    # Hugging Face hub directory from which we get the pre-trained CIFAR10 DDPM model's config
    model_id = "google/ddpm-cifar10-32"

    # Load the DDPM pipeline with the correct model and scheduler
    pipeline = DDIMPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)  # Move to the specified device

    # Generate a random initial sample in float16 to match the model's precision
    sample = torch.randn(
        num_images,
        pipeline.unet.config.in_channels,
        pipeline.unet.config.sample_size,
        pipeline.unet.config.sample_size,
        dtype=torch.float16
    ).to(device)

    # Set the number of inference steps
    pipeline.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Iteratively refine the sample
    for t in tqdm(pipeline.scheduler.timesteps):
        # Predict the noise residual using autocast for mixed precision
        with torch.no_grad(), torch.cuda.amp.autocast():
            noise_pred = pipeline.unet(sample, t).sample

        # Compute the next less noisy image and update the sample
        sample = pipeline.scheduler.step(noise_pred, t, sample).prev_sample

    # Save the generated images
    print("Saving generated images")
    print("Sample shape:", sample.shape)

    if save_grid_shape is not None:
        save_images_grid(
            sample, 
            os.path.join(save_fig_dir, "ddpm_cifar10_sample_grid.png"), 
            grid_shape=save_grid_shape,
        )
    else:
        # Convert the tensor to images and save
        samples = sample.clamp(-1, 1).add(1).div(2).mul(255).to(torch.uint8).cpu()
        for i, img in enumerate(samples):
            img = img.permute(1, 2, 0).numpy()  # Reorder dimensions to HWC
            img = Image.fromarray(img)
            img.save(os.path.join(save_fig_dir, f"ddpm_cifar10_sample_{i}.png"))

from diffusers import DDPMPipeline
def generate_ddpm(
    num_inference_steps: int = 50,
    num_images: int = 1,
    save_grid_shape: Optional[Tuple[int, int]] = None,
    save_fig_dir: str = "figs",
    device='cpu',
) -> None:
    """
    Generate images using the DDPM model from the Hugging Face Hub
    """
    # This example uses a simple DDPM model that is lightweight and suitable for unconditional generation
    model_id = "google/ddpm-cifar10-32"  # A lightweight model, trained on CIFAR-10, closest lightweight option

    # Initialize the diffusion pipeline
    pipeline = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)

    # Generate a batch of images
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Generate all images in one batch call
        sample = pipeline(num_inference_steps=num_inference_steps, batch_size=num_images).images

    # Convert batch of PIL images to a tensor or NumPy array
    # Convert to tensor: shape will be [num_images, channels, height, width]
    sample = torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1) for img in sample]) / 255.0
    # # Alternatively, convert to NumPy array: shape will be [num_images, height, width, channels]
    # images_np = np.stack([np.array(img) for img in batch]) / 255.0

    # Save the generated images
    print("Saving generated images")
    print("Sample shape:", sample.shape)

    # Convert images from [-1, 1] to [0, 255] format and save
    if save_grid_shape is not None:
        save_images_grid(
            sample, 
            os.path.join(save_fig_dir, "another_ddpm_cifar10_sample_grid.png"),
            grid_shape=save_grid_shape
        )
    else:
        for i, img in enumerate(sample):
            img = img.convert("RGB")  # Ensure RGB format
            img.save(os.path.join(save_fig_dir, f"another_ddpm_cifar10_sample_{i}.png"))

from diffusers import DiffusionPipeline
def ldm_pipeline(
    prompt: str = "A painting of a squirrel eating a burger",
    num_inference_steps: int = 50,
    guidance_scale: float = 6,
    save_fig_dir: str = "figs",
    device: str = 'cpu'
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
        guidance_scale=guidance_scale
    ).images

    # save images
    for idx, image in enumerate(images):
        image.save(os.path.join(save_fig_dir, f"ldm_256_sample_{idx}.png"))


def generate_image_DiffusionPipe(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    prompt: Optional[str] = None, # "A photo of a cat, realistic style",
    seed: Optional[int] = None,
    num_images: int = 1,
    save_fig_dir: str = "figs",
    save_grid_shape: Optional[Tuple[int, int]] = None
) -> None:
    """
    Generate images based on ImageNet class descriptions using a text-to-image diffusion model.
    
    Args:
        model_id (str): The model ID of the pretrained text-to-image diffusion model (default: "CompVis/stable-diffusion-v1-4").
        num_inference_steps (int): Number of denoising steps (default: 50).
        guidance_scale (float): Scale for classifier-free guidance (default: 7.5).
        height (int): Height of the image (default: 512).
        width (int): Width of the image (default: 512).
        prompt (Optional[str]): A text description of the ImageNet class.
        seed (Optional[int]): Random seed for reproducibility (default: None).
        num_images (int): Number of images to generate (default: 1).
        save_fig_dir (str): Directory to save the generated images (default: "figs").
        save_grid_shape (Optional[Tuple[int, int]]): Shape of the grid to save the images as a grid (default: None). 
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
    image_samples = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_images=num_images
    ).images

    # Save the generated images
    if save_grid_shape is not None:
        save_images_grid(
            image_samples, 
            os.path.join(save_fig_dir, "generated_sample_grid.png"), 
            grid_shape=save_grid_shape
        )
    else:
        for i, image in enumerate(image_samples):
            image.save(os.path.join(save_fig_dir, f"generated_sample_{i}.png"))