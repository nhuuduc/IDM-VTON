from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        import os

        from fashn_human_parser import FashnHumanParser
        from fashn_vton import TryOnPipeline
        from huggingface_hub import hf_hub_download

        self.weights_dir = "/src/weights"
        os.makedirs(self.weights_dir, exist_ok=True)
        dwpose_dir = os.path.join(self.weights_dir, "dwpose")
        os.makedirs(dwpose_dir, exist_ok=True)

        hf_hub_download(
            repo_id="fashn-ai/fashn-vton-1.5",
            filename="model.safetensors",
            local_dir=self.weights_dir,
        )
        hf_hub_download(
            repo_id="fashn-ai/DWPose",
            filename="yolox_l.onnx",
            local_dir=dwpose_dir,
        )
        hf_hub_download(
            repo_id="fashn-ai/DWPose",
            filename="dw-ll_ucoco_384.onnx",
            local_dir=dwpose_dir,
        )

        # Trigger parser weights cache at build/runtime setup once.
        _ = FashnHumanParser(device="cpu")

        self.pipeline = TryOnPipeline(weights_dir=self.weights_dir)

    def predict(
        self,
        person_image: Path = Input(description="Person image"),
        garment_image: Path = Input(description="Garment image"),
        category: str = Input(
            description='Garment category: "tops", "bottoms", or "one-pieces"',
            choices=["tops", "bottoms", "one-pieces"],
            default="tops",
        ),
        garment_photo_type: str = Input(
            description='Garment photo type: "model" or "flat-lay"',
            choices=["model", "flat-lay"],
            default="model",
        ),
        num_samples: int = Input(default=1, ge=1, le=4),
        num_timesteps: int = Input(default=30, ge=10, le=60),
        guidance_scale: float = Input(default=1.5, ge=0.1, le=5.0),
        seed: int = Input(default=42),
        segmentation_free: bool = Input(default=True),
    ) -> Path:
        from PIL import Image

        person = Image.open(person_image).convert("RGB")
        garment = Image.open(garment_image).convert("RGB")

        result = self.pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=num_samples,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
            segmentation_free=segmentation_free,
        )

        out_path = "/tmp/output.png"
        result.images[0].save(out_path)
        return Path(out_path)
