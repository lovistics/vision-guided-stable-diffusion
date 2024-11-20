import logging
from analyzer import FaceAnalyzer
from pose import PoseDetector
from generator import ImageGenerator
from processor import ImageProcessor
from gpt import GPTHandler
from typing import Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpiritAnimalGenerator:
    def __init__(self, openai_api_key: str):
        self.face_analyzer = FaceAnalyzer()
        self.gpt_handler = GPTHandler(openai_api_key)
        self.image_generator = ImageGenerator()
        self.image_processor = ImageProcessor()
        self.pose_detector = PoseDetector()

    async def process_image(self, image_path: str, output_path: str = "output") -> Dict:
        try:
            # Process input image
            pose_image, face_image = self.image_processor.load_image(image_path)
            face_analysis = self.face_analyzer.analyze_face(face_image)
            pose_data = self.pose_detector.detect_pose(pose_image)

            # Get spirit animal suggestions
            spirit_animal = await self.gpt_handler.analyze_spirit_animal(
                face_data=face_analysis,
                pose_data=pose_data
            )

            # Generate individual images for each animal
            generated_images = {}

            # Generate primary animal image
            primary_prompt = f"majestic {spirit_animal.primary_animal} in {pose_data['pose_type']} pose, "
            primary_prompt += "natural environment, professional wildlife photography, detailed lighting, photorealistic"

            primary_image = self.image_generator.generate_image(
                primary_prompt,
                pose_data['pose_map']
            )

            # Save primary animal image
            primary_path = self.image_generator.save_image(
                primary_image,
                output_path,
                f"primary_{Path(image_path).stem}"
            )
            generated_images['primary'] = str(primary_path)

            # Generate images for each feature-specific animal
            for animal in spirit_animal.animals:
                if animal['animal'] != spirit_animal.primary_animal:
                    prompt = f"majestic {animal['animal']} showing {animal['associated_feature']}, "
                    prompt += "natural environment, professional wildlife photography, detailed lighting, photorealistic"

                    feature_image = self.image_generator.generate_image(
                        prompt,
                        pose_data['pose_map']
                    )

                    feature_path = self.image_generator.save_image(
                        feature_image,
                        output_path,
                        f"{animal['animal']}_{Path(image_path).stem}"
                    )
                    generated_images[animal['associated_feature']] = str(feature_path)

            return {
                'face_analysis': face_analysis,
                'pose_data': pose_data,
                'spirit_animal': spirit_animal,
                'generated_images': generated_images,
                'prompts_used': {
                    'primary': primary_prompt,
                    'features': {animal['associated_feature']: f"majestic {animal['animal']}"
                               for animal in spirit_animal.animals
                               if animal['animal'] != spirit_animal.primary_animal}
                }
            }

        except Exception as e:
            logger.error(f"Error in spirit animal generation pipeline: {str(e)}")
            raise

def setup_environment() -> Dict:
    """Set up the environment and return configuration."""
    config = {
        'openai_api_key': '',
    }
    return config