import openai
import logging
import json
from typing import Any, Dict, List
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimalCategory(Enum):
    DOMESTIC = "domestic"
    WILD = "wild"
    MYTHICAL = "mythical"
    MARINE = "marine"
    AERIAL = "aerial"
    WILD_AERIAL = "wild/aerial"
    WILD_MARINE = "wild/marine"
    WILD_DOMESTIC = "wild/domestic"

    @classmethod
    def parse(cls, value: str) -> 'AnimalCategory':
        try:
            return cls(value.lower())
        except ValueError:
            normalized = value.lower().strip()
            for category in cls:
                if category.value == normalized:
                    return category
            return cls.WILD

@dataclass
class SpiritAnimalResult:
    animals: List[Dict[str, Any]]
    primary_animal: str
    categories: List[AnimalCategory]
    reasoning: str
    characteristics: List[str]
    pose_description: str
    confidence: float
    prompt_elements: Dict[str, str]
    feature_animal_mapping: Dict[str, str]

class GPTHandler:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.config = {
            'model': 'gpt-4',
            'max_tokens': 1000,
            'temperature': 0.7,
            'presence_penalty': 0.1,
            'frequency_penalty': 0.1
        }
        self.base_prompt = """
        You are an expert in analyzing human characteristics and matching them to spirit animals.
        Based on the detailed description of facial features, expressions, and body pose provided,
        determine multiple spirit animals that represent different aspects of the person.

        Consider matching specific features to animals:
        1. Eyes -> Animal known for its gaze/vision
        2. Mouth/Expression -> Animal known for its expressions
        3. Overall Face Shape -> Animal with similar face structure
        4. Body Posture -> Animal with similar movement patterns
        5. Overall Energy -> Animal that matches the general presence

        Format your response as JSON with the following structure:
        {
            "animals": [
                {
                    "animal": "name of animal",
                    "category": "wild/domestic/mythical/marine/aerial",
                    "associated_feature": "which feature this animal represents",
                    "reasoning": "why this animal matches"
                }
            ],
            "primary_animal": "the most dominant spirit animal",
            "overall_reasoning": "detailed explanation of the combination",
            "characteristics": ["trait1", "trait2", "trait3"],
            "pose_description": "description of pose",
            "confidence": 0.95,
            "feature_animal_mapping": {
                "eyes": "animal name",
                "expression": "animal name",
                "face_shape": "animal name",
                "posture": "animal name",
                "energy": "animal name"
            }
        }
        """

    def _build_analysis_prompt(self, face_data: Dict, pose_data: Dict) -> str:
        # text description of facial features
        emotion = face_data.get('emotion', 'neutral')
        face_shape = face_data.get('features', {}).get('face_shape', 'unknown')
        symmetry_score = face_data.get('features', {}).get('symmetry_score', 0)
        eye_ratio = face_data.get('features', {}).get('eye_aspect_ratio', 0)
        mouth_ratio = face_data.get('features', {}).get('mouth_aspect_ratio', 0)

        # pose description
        pose_type = pose_data.get('pose_type', 'unknown')
        balance = pose_data.get('metrics', {}).get('balance', 0)
        stability = pose_data.get('metrics', {}).get('stability', 0)
        pose_symmetry = pose_data.get('metrics', {}).get('symmetry', 0)

        prompt = f"""
        Analyze this detailed description and identify multiple spirit animals
        that represent different aspects of the person. For animal categories,
        use only these values: domestic, wild, mythical, marine, aerial, or
        combined categories like wild/aerial for flying wild animals.

        FACIAL CHARACTERISTICS:
        1. Eyes:
           - Eye Shape: {"wide and expressive" if eye_ratio > 0.3 else "gentle and focused" if eye_ratio > 0.2 else "narrow and intense"}
           - Eye Movement: {face_data.get('features', {}).get('eye_movement', 'steady')}

        2. Expression:
           - Primary Emotion: {emotion}
           - Mouth Expression: {"broad and open" if mouth_ratio > 0.6 else "neutral" if mouth_ratio > 0.3 else "subtle and reserved"}

        3. Face Structure:
           - Shape: {face_shape}
           - Symmetry: {symmetry_score:.2f}/1.0

        4. Overall Energy:
           - Presence: {"High energy and dynamic" if stability < 0.5 else "Balanced and controlled" if stability < 0.8 else "Grounded and steady"}
           - Movement Quality: {"Fluid and expressive" if pose_symmetry < 0.5 else "Balanced and coordinated" if pose_symmetry < 0.8 else "Precise and controlled"}

        Match each feature to a suitable spirit animal and provide an overall combination
        that represents their complete spiritual nature. Respond in the specified JSON format.
        """
        return prompt

    def _describe_score(self, score: float) -> str:
        """Convert numerical scores into descriptive text"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "moderate"
        elif score >= 0.2:
            return "fair"
        else:
            return "low"

    async def analyze_spirit_animal(
        self,
        face_data: Dict,
        pose_data: Dict
    ) -> SpiritAnimalResult:
        try:
            prompt = self._build_analysis_prompt(face_data, pose_data)

            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {
                        "role": "system",
                        "content": self.base_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config['max_tokens']
            )

            return self._parse_gpt_response(response)
        except Exception as e:
            logger.error(f"Error in spirit animal analysis: {str(e)}")
            raise

    def _parse_gpt_response(self, response) -> SpiritAnimalResult:
        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)

            # Parse categories with error handling
            categories = []
            for animal in parsed['animals']:
                try:
                    category = AnimalCategory.parse(animal['category'])
                    categories.append(category)
                except (ValueError, KeyError):
                    # Default to WILD if category parsing fails
                    categories.append(AnimalCategory.WILD)
                    logger.warning(f"Invalid category found: {animal.get('category', 'unknown')}, defaulting to WILD")

            return SpiritAnimalResult(
                animals=parsed['animals'],
                primary_animal=parsed['primary_animal'],
                categories=categories,  # Use the parsed categories
                reasoning=parsed.get('overall_reasoning', ''),
                characteristics=parsed.get('characteristics', []),
                pose_description=parsed.get('pose_description', ''),
                confidence=float(parsed.get('confidence', 0.8)),
                prompt_elements=parsed.get('prompt_elements', {}),
                feature_animal_mapping=parsed.get('feature_animal_mapping', {})
            )
        except Exception as e:
            logger.error(f"Error parsing GPT response: {str(e)}")
            raise

    def generate_image_prompt(self, result: SpiritAnimalResult, pose_data: Dict) -> str:
      """Generate a detailed prompt for image generation with multiple animals"""
      pose_type = pose_data.get('pose_type', 'standing')
      pose_metrics = pose_data.get('metrics', {})
      stability = pose_metrics.get('stability', 0.5)

      # Create a scene that incorporates multiple animals with specific positioning
      primary_animal = result.primary_animal
      other_animals = [animal for animal in result.animals
                      if animal['animal'] != primary_animal][:2]  # Limit to 2 additional animals

      # Build scene description with specific positions and interactions
      scene = f"close-up cinematic shot of a majestic {primary_animal} in {pose_type} pose in the foreground, "

      if other_animals:
          # Add secondary animals with specific positioning based on their features
          for i, animal in enumerate(other_animals):
              feature = animal['associated_feature']
              if 'eyes' in feature.lower():
                  scene += f"a {animal['animal']} watching attentively from a elevated position, "
              elif 'expression' in feature.lower():
                  scene += f"a {animal['animal']} interacting naturally in the mid-ground, "
              elif 'energy' in feature.lower():
                  scene += f"a {animal['animal']} moving dynamically in the scene, "
              else:
                  scene += f"a {animal['animal']} visible at a complementary position, "

      # scene composition elements
      scene += (
          "all animals in perfect scale proportion, natural environment, "
          "cinematic lighting, detailed textures, "
          "maintaining original background elements, atmospheric perspective, "
          "professional wildlife photography, 8K UHD, photorealistic, "
          "cohesive composition with natural depth of field"
      )

      return scene

    def _validate_response(self, response_content: str) -> bool:
        """Validate that the response contains all required fields"""
        try:
            parsed = json.loads(response_content)
            required_fields = ['animal', 'category', 'reasoning',
                             'characteristics', 'pose_description']
            return all(field in parsed for field in required_fields)
        except json.JSONDecodeError:
            return False