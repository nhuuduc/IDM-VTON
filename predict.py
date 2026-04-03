from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModelRef
from src.unet_hacked_tryon import UNet2DConditionModel


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        base_path = "yisol/IDM-VTON"

        unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=self.dtype,
        )
        tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

        text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=self.dtype,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path,
            subfolder="image_encoder",
            torch_dtype=self.dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            base_path,
            subfolder="vae",
            torch_dtype=self.dtype,
        )
        unet_encoder = UNet2DConditionModelRef.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=self.dtype,
        )

        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
        )

        self.pipe.unet_encoder = unet_encoder
        self.pipe.to(self.device)
        self.pipe.unet_encoder.to(self.device)
        self.pipe.unet_encoder.eval()
        self.pipe.unet.eval()

    def _default_mask(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)
        left = int(width * 0.25)
        right = int(width * 0.75)
        top = int(height * 0.20)
        bottom = int(height * 0.80)
        mask[top:bottom, left:right] = 255
        return Image.fromarray(mask)

    def predict(
        self,
        human_image: Path = Input(description="Person image"),
        garment_image: Path = Input(description="Garment image"),
        garment_description: str = Input(
            description="Garment text prompt",
            default="a short-sleeve shirt",
        ),
        mask_image: Optional[Path] = Input(
            description="Optional mask image, white area is inpaint region",
            default=None,
        ),
        num_inference_steps: int = Input(default=30, ge=1, le=100),
        guidance_scale: float = Input(default=2.0, ge=0.1, le=20),
        seed: int = Input(default=42),
    ) -> Path:
        human_img = Image.open(human_image).convert("RGB").resize((768, 1024))
        garm_img = Image.open(garment_image).convert("RGB").resize((768, 1024))

        if mask_image:
            mask = Image.open(mask_image).convert("L").resize((768, 1024))
        else:
            mask = self._default_mask(human_img)

        autocast_ctx = torch.cuda.amp.autocast() if self.device == "cuda" else nullcontext()
        generator = torch.Generator(self.device).manual_seed(seed)

        with torch.no_grad(), autocast_ctx:
            prompt = f"model is wearing {garment_description}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            cloth_prompt = f"a photo of {garment_description}"
            (
                prompt_embeds_c,
                _,
                _,
                _,
            ) = self.pipe.encode_prompt(
                [cloth_prompt],
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=[negative_prompt],
            )

            garm_tensor = self.tensor_transform(garm_img).unsqueeze(0).to(self.device, self.dtype)
            # Pose condition is approximated by person image to keep API simple.
            pose_img = self.tensor_transform(human_img).unsqueeze(0).to(self.device, self.dtype)

            images = self.pipe(
                prompt_embeds=prompt_embeds.to(self.device, self.dtype),
                negative_prompt_embeds=negative_prompt_embeds.to(self.device, self.dtype),
                pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, self.dtype),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, self.dtype),
                num_inference_steps=num_inference_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img,
                text_embeds_cloth=prompt_embeds_c.to(self.device, self.dtype),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=garm_img,
                guidance_scale=guidance_scale,
            )[0]

        output = "/tmp/output.png"
        images[0].save(output)
        return Path(output)
