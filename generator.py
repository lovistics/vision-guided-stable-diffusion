import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationStyle(Enum):
    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    CARTOON = "cartoon"
    FANTASY = "fantasy"

@dataclass
class GenerationParams:
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    style: GenerationStyle
    seed: Optional[int] = None

class ImageGenerator:
    def __init__(self, config: Optional[Dict] = None):
        try:
            self.config = config or {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'model_id': "runwayml/stable-diffusion-v1-5",
                'controlnet_model': "lllyasviel/sd-controlnet-openpose",
                'default_style': GenerationStyle.REALISTIC,
                'default_steps': 30,  # Reduced from 50
                'default_height': 512,  # Reduced from 768
                'default_width': 512,   # Reduced from 768
                'default_guidance_scale': 7.5,
                'default_conditioning_scale': 0.8,
                'safety_checker': None  # Disabled for speed
            }

            # Use float16 for faster processing
            dtype = torch.float16 if self.config['device'] == 'cuda' else torch.float32

            # Initialize models with optimizations
            self.controlnet = ControlNetModel.from_pretrained(
                self.config['controlnet_model'],
                torch_dtype=dtype,
                use_safetensors=True
            )

            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.config['model_id'],
                controlnet=self.controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                use_safetensors=True
            ).to(self.config['device'])

            # Apply memory optimizations
            self.apply_memory_optimization()


            self.style_prompts = {
                GenerationStyle.REALISTIC: (
                    "highly detailed, realistic, professional wildlife photography, "
                    "bright natural lighting, well-exposed, vibrant, 8K UHD, "
                    "perfect exposure, daylight scene"
                ),
                GenerationStyle.ARTISTIC: (
                    "digital art, stylized, vibrant colors, artistic composition, "
                    "bright lighting, well-lit scene"
                ),
                GenerationStyle.CARTOON: (
                    "cartoon style, cel shaded, simple, clean lines, "
                    "bright and colorful, vivid lighting"
                ),
                GenerationStyle.FANTASY: (
                    "fantasy art, magical, ethereal, mystical atmosphere, "
                    "radiant lighting, luminescent details"
                )
            }

            # negative prompts to prevent dark outputs
            self.negative_prompts = {
                GenerationStyle.REALISTIC: (
                    "dark, underexposed, dim lighting, shadows, silhouette, "
                    "low contrast, muddy, murky, gloomy, low quality"
                ),
                GenerationStyle.ARTISTIC: (
                    "dark, gloomy, underexposed, low contrast, "
                    "photograph, realistic, low quality, blurry"
                ),
                GenerationStyle.CARTOON: (
                    "dark colors, shadowy, moody, realistic, photograph, "
                    "complex, detailed, 3d rendering"
                ),
                GenerationStyle.FANTASY: (
                    "dark atmosphere, gloomy, underexposed, modern, mundane, "
                    "realistic photograph, low quality"
                )
            }

        except Exception as e:
            logger.error(f"Error initializing ImageGenerator: {str(e)}")
            raise

    def generate_image(
        self,
        base_prompt: str,
        pose_map: np.ndarray,
        params: Optional[GenerationParams] = None,
        num_inference_steps: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Image.Image:
        try:
            # Use default parameters if not specified
            num_inference_steps = num_inference_steps or self.config['default_steps']
            height = height or self.config['default_height']
            width = width or self.config['default_width']

            if params is None:
                params = self._get_default_params(base_prompt)

            pose_image = self._prepare_pose_map(pose_map)
            enhanced_prompt = f"{params.prompt}, bright natural lighting, well-exposed"

            # Generate with optimizations
            with torch.inference_mode(), torch.cuda.amp.autocast():
                output = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=params.negative_prompt,
                    image=pose_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=params.guidance_scale,
                    controlnet_conditioning_scale=params.controlnet_conditioning_scale,
                    height=height,
                    width=width
                )

                # Clear CUDA cache after generation
                if self.config['device'] == 'cuda':
                    torch.cuda.empty_cache()

            return self._post_process_image(output.images[0])

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise

    def _post_process_image(self, image: Image.Image) -> Image.Image:
        """Apply post-processing to enhance brightness and contrast"""
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Apply automatic contrast enhancement
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            # Increase brightness
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            # Calculate mean brightness and adjust accordingly
            mean_v = np.mean(v)
            if mean_v < 127:
                # Increase brightness if image is too dark
                brightness_increase = int(min(127 - mean_v, 50))  # Cap the increase
                v = cv2.add(v, brightness_increase)

            hsv = cv2.merge((h, s, v))
            img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            # Final contrast adjustment
            img_array = cv2.convertScaleAbs(img_array, alpha=1.1, beta=10)

            # Clip values to valid range
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            return Image.fromarray(img_array)

        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            return image

    def _prepare_pose_map(self, pose_map: np.ndarray) -> Image.Image:
        """Prepare pose map with enhanced contrast"""
        try:
            if len(pose_map.shape) == 2:
                pose_map = cv2.cvtColor(pose_map, cv2.COLOR_GRAY2RGB)

            # Enhance pose map contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            pose_map = cv2.cvtColor(pose_map, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(pose_map)
            l = clahe.apply(l)
            pose_map = cv2.merge((l,a,b))
            pose_map = cv2.cvtColor(pose_map, cv2.COLOR_LAB2RGB)

            # Increase line visibility
            pose_map = cv2.convertScaleAbs(pose_map, alpha=1.2, beta=10)

            return Image.fromarray(pose_map)

        except Exception as e:
            logger.error(f"Error preparing pose map: {str(e)}")
            if isinstance(pose_map, np.ndarray):
                return Image.fromarray(pose_map)
            return pose_map

    def _get_default_params(self, base_prompt: str) -> GenerationParams:
        """Get default generation parameters with style-specific enhancements"""
        style = self.config['default_style']
        enhanced_prompt = f"{base_prompt}, {self.style_prompts[style]}"

        return GenerationParams(
            prompt=enhanced_prompt,
            negative_prompt=self.negative_prompts[style],
            num_inference_steps=50,  # Increased steps
            guidance_scale=8.5,
            controlnet_conditioning_scale=0.8,
            style=style
        )

    def enhance_prompt(self, base_prompt: str, style: GenerationStyle) -> str:
        """Enhance prompt with style-specific additions"""
        return f"{base_prompt}, {self.style_prompts[style]}"

    async def batch_generate(
        self,
        prompts: List[str],
        pose_maps: List[np.ndarray],
        params_list: Optional[List[GenerationParams]] = None,
        batch_size: int = 4
    ) -> List[Image.Image]:
        """Generate multiple images in batches"""
        try:
            if params_list is None:
                params_list = [self._get_default_params(prompt) for prompt in prompts]

            results = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_pose_maps = pose_maps[i:i + batch_size]
                batch_params = params_list[i:i + batch_size]

                # Process batch
                batch_results = [
                    self.generate_image(prompt, pose_map, params)
                    for prompt, pose_map, params in zip(
                        batch_prompts, batch_pose_maps, batch_params
                    )
                ]
                results.extend(batch_results)

            return results

        except Exception as e:
            logger.error(f"Error in batch generation: {str(e)}")
            raise

    def save_image(
        self,
        image: Image.Image,
        output_path: Union[str, Path],
        filename: str,
        format: str = 'PNG',
        quality: int = 95
    ) -> Path:
        """Save generated image with optional quality settings"""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            file_path = output_path / f"{filename}.{format.lower()}"

            # Apply final enhancement before saving
            enhanced_image = self._post_process_image(image)

            # Save with specific format settings
            if format.lower() in ['jpg', 'jpeg']:
                enhanced_image.save(file_path, format=format, quality=quality)
            else:
                enhanced_image.save(file_path, format=format)

            return file_path

        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise

    def apply_memory_optimization(self):
        """Apply comprehensive memory optimizations"""
        try:
            # Enable attention slicing with automatic slice size
            self.pipe.enable_attention_slicing(slice_size="auto")

            # Enable model offloading if on CUDA
            if self.config['device'] == 'cuda':
                self.pipe.enable_model_cpu_offload()
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True

            # Enable xformers memory efficient attention if available
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
                # Fall back to VAE slicing if xformers is not available
                self.pipe.enable_vae_slicing()

            # Additional memory optimizations
            self.pipe.enable_sequential_cpu_offload()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error applying memory optimization: {str(e)}")
            raise