import numpy as np
import PIL
import torch

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


def prepare_mask_and_masked_image(image, mask):
  # preprocess image
  if isinstance(image, PIL.Image.Image):
    image = [image]

  if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
    image = [np.array(i.convert("RGB"))[None, :] for i in image]
    image = np.concatenate(image, axis=0)

  image = image.transpose(0, 3, 1, 2)
  image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

  # preprocess mask
  if isinstance(mask, PIL.Image.Image):
    mask = [mask]

  if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
    mask = [np.array(m.convert("L"))[None, None, :] for m in mask]
    mask = np.concatenate(mask, axis=0)
    mask = mask.astype(np.float32) / 255.0

  mask[mask < 0.5] = 0
  mask[mask >= 0.5] = 1
  mask = torch.from_numpy(mask)

  masked_image = image * (mask < 0.5)
  return mask, masked_image


class StableDiffusionControlNetInpaintPipeline(StableDiffusionControlNetPipeline):
  def prepare_mask_latents(self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance):
    # resize the mask to latents shape as we concatenate the mask to the latents
    # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
    # and half precision
    mask = torch.nn.functional.interpolate(
      mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
    )
    mask = mask.to(device=device, dtype=dtype)

    masked_image = masked_image.to(device=device, dtype=dtype)

    # encode the mask image into latents space so we can concatenate it to the latents
    if isinstance(generator, list):
      masked_image_latents = [
        self.vae.encode(masked_image[i : i + 1]).latent_dist.sample(generator=generator[i])
        for i in range(batch_size)
      ]
      masked_image_latents = torch.cat(masked_image_latents, dim=0)
    else:
      masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
    masked_image_latents = self.vae.config.scaling_factor * masked_image_latents

    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
    if mask.shape[0] < batch_size:
      if not batch_size % mask.shape[0] == 0:
        raise ValueError(
          "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
          f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
          " of masks that you pass is divisible by the total requested batch size."
        )
      mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
    if masked_image_latents.shape[0] < batch_size:
      if not batch_size % masked_image_latents.shape[0] == 0:
        raise ValueError(
          "The passed images and the required batch size don't match. Images are supposed to be duplicated"
          f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
          " Make sure the number of images that you pass is divisible by the total requested batch size."
        )
      masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

    mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
    masked_image_latents = (
      torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
    )

    # aligning device to prevent device errors when concating it with the latent model input
    masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
    return mask, masked_image_latents

  @torch.no_grad()
  def __call__(
    self,
    prompt=None,
    image=None,
    control_image=None,
    mask_image=None,
    height=None,
    width=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt=None,
    num_images_per_prompt=1,
    eta=0.0,
    generator=None,
    latents=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    output_type="pil",
    return_dict=True,
    callback=None,
    callback_steps=1,
    cross_attention_kwargs=None,
    controlnet_conditioning_scale=1.0,
  ):
    # 0. Default height and width to unet
    if width is None or height is None:
      width, height = image.size

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt=prompt,
                      image=control_image,
                      callback_steps=callback_steps,
                      negative_prompt=negative_prompt,
                      prompt_embeds=prompt_embeds,
                      negative_prompt_embeds=negative_prompt_embeds,
                      controlnet_conditioning_scale=controlnet_conditioning_scale)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
      batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
      prompt,
      device,
      num_images_per_prompt,
      do_classifier_free_guidance,
      negative_prompt,
      prompt_embeds=prompt_embeds,
      negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare image
    control_image = self.prepare_image(
      control_image,
      width,
      height,
      batch_size * num_images_per_prompt,
      num_images_per_prompt,
      device,
      self.controlnet.dtype,
    )

    if do_classifier_free_guidance:
      control_image = torch.cat([control_image] * 2)

    # 5. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 6. Prepare latent variables
    num_channels_latents = self.controlnet.config.in_channels
    latents = self.prepare_latents(
      batch_size * num_images_per_prompt,
      num_channels_latents,
      height,
      width,
      prompt_embeds.dtype,
      device,
      generator,
      latents,
    )

    # EXTRA: prepare mask latents
    mask, masked_image = prepare_mask_and_masked_image(image, mask_image)
    mask, masked_image_latents = self.prepare_mask_latents(
      mask,
      masked_image,
      batch_size * num_images_per_prompt,
      height,
      width,
      prompt_embeds.dtype,
      device,
      generator,
      do_classifier_free_guidance,
    )

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
      for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
          latent_model_input,
          t,
          encoder_hidden_states=prompt_embeds,
          controlnet_cond=control_image,
          return_dict=False,
        )

        down_block_res_samples = [
          down_block_res_sample * controlnet_conditioning_scale
          for down_block_res_sample in down_block_res_samples
        ]
        mid_block_res_sample *= controlnet_conditioning_scale

        # predict the noise residual
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
        noise_pred = self.unet(
          latent_model_input,
          t,
          encoder_hidden_states=prompt_embeds,
          cross_attention_kwargs=cross_attention_kwargs,
          down_block_additional_residuals=down_block_res_samples,
          mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
          noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
          noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
          progress_bar.update()
          if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)

    # If we do sequential model offloading, let's offload unet and controlnet
    # manually for max memory savings
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
      self.unet.to("cpu")
      self.controlnet.to("cpu")
      torch.cuda.empty_cache()

    if output_type == "latent":
      image = latents
    elif output_type == "pil":
      # 8. Post-processing
      image = self.decode_latents(latents)
      # 10. Convert to PIL
      image = self.numpy_to_pil(image)
    else:
      # 8. Post-processing
      image = self.decode_latents(latents)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
      self.final_offload_hook.offload()

    has_nsfw_concept = None
    if not return_dict:
      return (image, has_nsfw_concept)
    else:
      return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
 