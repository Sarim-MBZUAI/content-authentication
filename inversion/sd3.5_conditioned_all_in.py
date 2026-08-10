import torch
import argparse
import os
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from torchvision import transforms
import re







def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@torch.inference_mode()
def decode_imgs(latents, pipeline):
    imgs = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    imgs = pipeline.vae.decode(imgs)[0]
    imgs = pipeline.image_processor.postprocess(imgs, output_type="pil")
    return imgs

@torch.inference_mode()
def encode_imgs(imgs, pipeline, DTYPE):
    latents = pipeline.vae.encode(imgs).latent_dist.sample()
    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    latents = latents.to(dtype=DTYPE)
    return latents

def generate_eta_values(
    timesteps, 
    start_step, 
    end_step, 
    eta, 
    eta_trend,
):
    assert start_step < end_step and start_step >= 0 and end_step <= len(timesteps), "Invalid start_step and end_step"
    # timesteps are monotonically decreasing, from 1.0 to 0.0
    eta_values = [0.0] * len(timesteps)
    
    if eta_trend == 'constant':
        for i in range(start_step, end_step):
            eta_values[i] = eta
    elif eta_trend == 'linear_increase':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[start_step] - timesteps[i]) / total_time
    elif eta_trend == 'linear_decrease':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[i] - timesteps[end_step - 1]) / total_time
    else:
        raise NotImplementedError(f"Unsupported eta_trend: {eta_trend}")
    
    return eta_values

@torch.inference_mode()
def interpolated_denoise(
    pipeline, 
    img_latents,
    eta_base,                    # base eta value
    eta_trend,                   # constant, linear_increase, linear_decrease
    start_step,                  # 0-based indexing, closed interval
    end_step,                    # 0-based indexing, open interval
    inversed_latents,            # can be none if not using inversed latents
    use_inversed_latents=True,
    guidance_scale=3.5,
    prompt='photo of a tiger',
    DTYPE=torch.bfloat16,
    num_steps=28,
    seed=42
):

    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_steps, pipeline.device)

    # Getting text embedning
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds
    ) = pipeline.encode_prompt(
        prompt=prompt, 
        prompt_2=prompt,
        prompt_3=prompt
    )

    if use_inversed_latents:
        latents = inversed_latents
    else:
        set_seed(seed)
        latents = torch.randn_like(img_latents)
    
    target_img = img_latents.clone().to(torch.float32)

    # get the eta values for each steps in 
    eta_values = generate_eta_values(timesteps, start_step, end_step, eta_base, eta_trend)

    # handle guidance scale if need
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    with pipeline.progress_bar(total=num_steps) as progress_bar:
        for i, t in enumerate(timesteps):

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])

            # Editing text velocity
            pred_velocity = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # perform guidance scale
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = pred_velocity.chunk(2)
                pred_velocity = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Prevents precision issues
            latents = latents.to(torch.float32)
            pred_velocity = pred_velocity.to(torch.float32)

            # Target image velocity
            t_curr = t / pipeline.scheduler.config.num_train_timesteps
            target_velocity = -(target_img - latents) / t_curr

            # interpolated velocity
            eta = eta_values[i]
            interpolate_velocity = pred_velocity + eta * (target_velocity - pred_velocity)

            # denosing
            latents = pipeline.scheduler.step(interpolate_velocity, t, latents, return_dict=False)[0]
            
            latents = latents.to(DTYPE)
            progress_bar.update()
    
    return latents


@torch.inference_mode()
def interpolated_inversion(
    pipeline, 
    latents,
    gamma,
    DTYPE,
    prompt = "",
    num_steps=28,
    seed=42
):

    # using FlowMatchEulerDiscreteScheduler
    pipeline.scheduler.set_timesteps(num_steps, device=pipeline.device)

    # check if it has sigma avaible 
    if not hasattr(pipeline.scheduler, "sigmas"):
        raise Exception("Cannot find sigmas variable in scheduler. Please use FlowMatchEulerDiscreteScheduler to doing RF Inversion")
    
    # we get timestep directy from sigmas
    timesteps = pipeline.scheduler.sigmas
    timesteps = torch.flip(timesteps, dims=[0])

    # Getting null-text embedning
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds
    ) = pipeline.encode_prompt( # null text
        prompt=prompt, 
        prompt_2=prompt,
        prompt_3=prompt,
    )

    # generate gaussain noise with seed
    set_seed(seed)
    target_noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32)

    # # Image inversion with interpolated velocity field.  t goes from 0.0 to 1.0
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((latents.shape[0],), t_curr * 1000, dtype=latents.dtype, device=latents.device)

            # Null-text velocity
            pred_velocity = pipeline.transformer(
                hidden_states=latents,
                timestep=t_vec,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # Prevents precision issues
            latents = latents.to(torch.float32)
            pred_velocity = pred_velocity.to(torch.float32)

            # Target noise velocity
            target_noise_velocity = (target_noise - latents) / (1.0 - t_curr)
            
            # interpolated velocity
            interpolated_velocity = gamma * target_noise_velocity + (1 - gamma) * pred_velocity
            
            # one step Euler, similar to pipeline.scheduler.step but in the forward to noise instead of denosing
            latents = latents + (t_prev - t_curr) * interpolated_velocity
            
            latents = latents.to(DTYPE)
            progress_bar.update()
            
    return latents





















def extract_prompt_from_filename(filename):
    """
    Extracts meaningful prompts from filenames like:
    'A_BMW_motorcycle_parked_on_a_brick_street__e7419eeeee.png'
    Returns the prompt part without the identifier and with proper spacing.
    
    Args:
        filename (str): The filename to process
        
    Returns:
        str: Extracted prompt with proper spacing and formatting
    """
    # Remove the file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Find the last sequence of characters that matches the pattern _[a-f0-9]{10}
    # This removes the unique identifier at the end
    match = re.search(r'_[a-f0-9]{10}$', name_without_ext)
    
    if match:
        # Remove the identifier part
        prompt_part = name_without_ext[:match.start()]
    else:
        prompt_part = name_without_ext
    
    # Replace underscores with spaces and handle double underscores
    prompt = prompt_part.replace('__', ' ').replace('_', ' ').strip()
    
    return prompt

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='Process images with SD3 interpolated_denoise using filename prompts.')
    parser.add_argument('--model_path', type=str, default='stabilityai/stable-diffusion-3.5-medium', help='Path to the pretrained model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output images')
    parser.add_argument('--eta_base', type=float, default=0.95, help='Eta parameter for interpolated_denoise')
    parser.add_argument('--eta_trend', type=str, default='constant', choices=['constant', 'linear_increase', 'linear_decrease'], help='Eta trend for interpolated_denoise')
    parser.add_argument('--start_step', type=int, default=0, help='Start step for eta values')
    parser.add_argument('--end_step', type=int, default=9, help='End step for eta values')
    parser.add_argument('--no_inversion', action='store_true', help='Skip the inversion progress')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='Guidance scale for interpolated_denoise')    
    parser.add_argument('--num_steps', type=int, default=28, help='Number of steps for timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter for interpolated_inversion')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'], help='Data type for computations')
    
    args = parser.parse_args()

    if args.dtype == 'bfloat16':
        DTYPE = torch.bfloat16
    elif args.dtype == 'float16':
        DTYPE = torch.float16
    elif args.dtype == 'float32':
        DTYPE = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the SD3 pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=DTYPE)
    pipe = pipe.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up image transforms for SD3's 1024x1024 resolution
    train_transforms = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Process all images in the input directory
    for filename in os.listdir(args.input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
            
        try:
            print(f"Processing {filename}")
            
            # Extract prompt from filename
            prompt = extract_prompt_from_filename(filename)
            print(f"Extracted prompt: {prompt}")
            
            input_path = os.path.join(args.input_dir, filename)
            
            # Load and preprocess the image
            img = Image.open(input_path)
            img = train_transforms(img).unsqueeze(0).to(device).to(DTYPE)
            img_latent = encode_imgs(img, pipe, DTYPE)

            # Perform inversion if needed
            if not args.no_inversion:
                inversed_latent = interpolated_inversion(
                    pipe, 
                    img_latent,
                    gamma=args.gamma,
                    prompt=prompt,  # Use extracted prompt for inversion
                    DTYPE=DTYPE,
                    num_steps=args.num_steps,
                    seed=args.seed
                )    
            else:
                inversed_latent = None

            # Perform denoising with the extracted prompt
            img_latents = interpolated_denoise(
                pipe, 
                img_latent,
                eta_base=args.eta_base,
                eta_trend=args.eta_trend,
                start_step=args.start_step,
                end_step=args.end_step,
                inversed_latents=inversed_latent,
                use_inversed_latents=not args.no_inversion,
                guidance_scale=args.guidance_scale,
                prompt=prompt,  # Use extracted prompt for denoising
                DTYPE=DTYPE,
                seed=args.seed
            )

            # Decode and save the result
            out = decode_imgs(img_latents, pipe)[0]
            output_path = os.path.join(args.output_dir, filename)
            out.save(output_path)
            print(f"Saved output image to {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

if __name__ == "__main__":
    main()