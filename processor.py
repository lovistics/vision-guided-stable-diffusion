import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
import logging
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'input_size': (512, 512),
            'normalize_mean': [0.0, 0.0, 0.0],
            'normalize_std': [1.0, 1.0, 1.0],
            'max_image_size': 8192,
            'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp'],
            'enhance_contrast': True,
            'enhance_brightness': True,
            'clahe_clip_limit': 3.0,
            'clahe_grid_size': (8, 8)
        }

        # Modified transform pipeline to preserve brightness
        self.transform = transforms.Compose([
            transforms.Resize(self.config['input_size']),
            transforms.ToTensor()
            # Removed normalization to preserve original image characteristics
        ])

        # Initialize CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_grid_size']
        )

    def load_image(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if isinstance(image_path, str):
                image_path = Path(image_path)

            if not self._validate_image_path(image_path):
                raise ValueError(f"Invalid image path or format: {image_path}")

            bgr_image = cv2.imread(str(image_path))
            if bgr_image is None:
                raise ValueError(f"Failed to load image at {image_path}")

            if not self._check_image_size(bgr_image):
                raise ValueError(f"Image dimensions exceed maximum allowed size of {self.config['max_image_size']}")

            # Convert to RGB and enhance
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            enhanced_image = self._enhance_image_quality(rgb_image)

            # Create copies for pose and face analysis
            pose_image = enhanced_image.copy()
            face_image = enhanced_image.copy()

            # Resize while maintaining aspect ratio
            pose_image = self._resize_preserve_aspect(pose_image, self.config['input_size'])
            face_image = self._resize_preserve_aspect(face_image, self.config['input_size'])

            return pose_image, face_image

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive image enhancement"""
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            if self.config['enhance_contrast']:
                # Apply CLAHE to L channel
                l = self.clahe.apply(l)

            # Merge channels
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            if self.config['enhance_brightness']:
                # Convert to HSV for brightness adjustment
                hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)

                # Calculate mean brightness
                mean_v = np.mean(v)
                if mean_v < 127:
                    # Increase brightness if image is too dark
                    v = cv2.add(v, int(127 - mean_v))

                hsv = cv2.merge((h, s, v))
                enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            # Ensure valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image

    def _resize_preserve_aspect(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while preserving aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate aspect ratios
        aspect = min(target_w/w, target_h/h)
        new_w = int(w * aspect)
        new_h = int(h * aspect)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create canvas of target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Calculate position to paste resized image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # Paste resized image
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def prepare_for_face(self, image: np.ndarray) -> np.ndarray:
        try:
            if not self.validate_image(image):
                raise ValueError("Invalid input image for facial analysis")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Enhance contrast for better feature detection
            enhanced_gray = self.clahe.apply(gray)

            return enhanced_gray

        except Exception as e:
            logger.error(f"Error preparing image for facial analysis: {str(e)}")
            raise

    def prepare_for_controlnet(
        self,
        image: np.ndarray,
        pose_map: Optional[np.ndarray] = None,
        enhance: bool = True
    ) -> Dict[str, torch.Tensor]:
        try:
            if not self.validate_image(image):
                raise ValueError("Invalid input image for ControlNet")

            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            if enhance:
                # Convert back to numpy for enhancement
                enhanced_array = self._enhance_image_quality(np.array(pil_image))
                pil_image = Image.fromarray(enhanced_array)

            # Handle pose map
            condition_image = pose_map if pose_map is not None else image
            pil_condition = Image.fromarray(condition_image)
            if enhance:
                # Enhance pose map contrast
                enhanced_condition = self._enhance_pose_map(np.array(pil_condition))
                pil_condition = Image.fromarray(enhanced_condition)

            # Convert to tensors without normalization
            image_tensor = transforms.ToTensor()(pil_image)
            condition_tensor = transforms.ToTensor()(pil_condition)

            return {
                'image': image_tensor.unsqueeze(0),
                'condition': condition_tensor.unsqueeze(0)
            }

        except Exception as e:
            logger.error(f"Error preparing for ControlNet: {str(e)}")
            raise

    def _enhance_pose_map(self, pose_map: np.ndarray) -> np.ndarray:
        """Enhance pose map for better visibility"""
        try:
            if len(pose_map.shape) == 2:
                pose_map = cv2.cvtColor(pose_map, cv2.COLOR_GRAY2RGB)

            # Convert to LAB for enhanced contrast
            lab = cv2.cvtColor(pose_map, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            l = self.clahe.apply(l)

            # Merge channels
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            # Increase contrast
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=0)

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing pose map: {str(e)}")
            return pose_map

    def save_output(
        self,
        image: Union[np.ndarray, torch.Tensor],
        output_path: Union[str, Path],
        filename: str,
        format: str = 'png',
        quality: int = 95
    ) -> Path:
        try:
            if isinstance(output_path, str):
                output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            if isinstance(image, torch.Tensor):
                image = image.squeeze(0).permute(1, 2, 0).numpy()

            # Ensure proper scaling
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            # Enhance output image
            enhanced_image = self._enhance_image_quality(image)

            full_path = output_path / f"{filename}.{format}"

            if format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(
                    str(full_path),
                    cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                )
            else:
                cv2.imwrite(
                    str(full_path),
                    cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
                )

            return full_path

        except Exception as e:
            logger.error(f"Error saving output image: {str(e)}")
            raise

    def _validate_image_path(self, path: Path) -> bool:
        return (path.exists() and
                path.suffix.lower() in self.config['supported_formats'] and
                path.stat().st_size > 0)

    def _check_image_size(self, image: np.ndarray) -> bool:
        return all(dim <= self.config['max_image_size'] for dim in image.shape[:2])

    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        try:
            if image is None or image.size == 0:
                return False
            if len(image.shape) not in [2, 3]:
                return False
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                return False
            if np.isnan(image).any() or np.isinf(image).any():
                return False
            if image.dtype != np.uint8 and (image.max() > 1.0 or image.min() < 0.0):
                return False
            return True
        except Exception:
            return False

    def process_batch(self, image_paths: List[Union[str, Path]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        results = []
        for path in image_paths:
            try:
                result = self.load_image(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {path}: {str(e)}")
                results.append(None)
        return results