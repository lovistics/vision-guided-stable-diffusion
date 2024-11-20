import cv2
import dlib
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionType(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"
    THOUGHTFUL = "thoughtful"

@dataclass
class FacialFeatures:
    eye_aspect_ratio: float
    mouth_aspect_ratio: float
    eyebrow_position: float
    symmetry_score: float
    face_shape: str
    dominant_emotion: EmotionType
    confidence: float

class FaceAnalyzer:
    def __init__(self):
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
            self.FACIAL_LANDMARKS = {
                "jaw": list(range(0, 17)),
                "right_eyebrow": list(range(17, 22)),
                "left_eyebrow": list(range(22, 27)),
                "nose_bridge": list(range(27, 31)),
                "nose_tip": list(range(31, 36)),
                "right_eye": list(range(36, 42)),
                "left_eye": list(range(42, 48)),
                "outer_mouth": list(range(48, 60)),
                "inner_mouth": list(range(60, 68))
            }
        except Exception as e:
            logger.error(f"Error initializing FaceAnalyzer: {str(e)}")
            raise

    def analyze_face(self, image: np.ndarray) -> Dict:
        try:
            faces = self.detector(image)
            if len(faces) == 0:
                raise ValueError("No faces detected in the image")

            face = max(faces, key=lambda rect: rect.width() * rect.height())
            landmarks = self.predictor(image, face)
            landmarks_points = self._get_landmarks_points(landmarks)

            features = self._extract_facial_features(landmarks_points)
            emotion = self._analyze_emotion(features)
            confidence = self._calculate_confidence(face, landmarks_points)
            description = self._generate_feature_description(features, emotion)

            return {
                'features': features.__dict__,
                'landmarks': landmarks_points,
                'emotion': emotion.value,
                'confidence': confidence,
                'description': description,
                'face_rect': {
                    'x': face.left(),
                    'y': face.top(),
                    'width': face.width(),
                    'height': face.height()
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing face: {str(e)}")
            raise

    def _get_landmarks_points(self, landmarks) -> List[Tuple[int, int]]:
        return [(landmarks.part(i).x, landmarks.part(i).y)
                for i in range(68)]

    def _extract_facial_features(self, landmarks: List[Tuple[int, int]]) -> FacialFeatures:
        left_eye = [landmarks[i] for i in self.FACIAL_LANDMARKS["left_eye"]]
        right_eye = [landmarks[i] for i in self.FACIAL_LANDMARKS["right_eye"]]
        eye_ratio = (self._calculate_eye_ratio(left_eye) +
                    self._calculate_eye_ratio(right_eye)) / 2

        outer_mouth = [landmarks[i] for i in self.FACIAL_LANDMARKS["outer_mouth"]]
        mouth_ratio = self._calculate_mouth_ratio(outer_mouth)

        left_brow = [landmarks[i] for i in self.FACIAL_LANDMARKS["left_eyebrow"]]
        right_brow = [landmarks[i] for i in self.FACIAL_LANDMARKS["right_eyebrow"]]
        brow_pos = self._calculate_eyebrow_position(left_brow, right_brow)

        symmetry = self._calculate_symmetry(landmarks)
        face_shape = self._determine_face_shape(
            [landmarks[i] for i in self.FACIAL_LANDMARKS["jaw"]]
        )

        return FacialFeatures(
            eye_aspect_ratio=eye_ratio,
            mouth_aspect_ratio=mouth_ratio,
            eyebrow_position=brow_pos,
            symmetry_score=symmetry,
            face_shape=face_shape,
            dominant_emotion=EmotionType.NEUTRAL,
            confidence=0.0
        )

    def _analyze_emotion(self, features: FacialFeatures) -> EmotionType:
        if features.mouth_aspect_ratio > 0.6:
            return EmotionType.HAPPY
        elif features.eyebrow_position < -0.2:
            return EmotionType.ANGRY
        elif features.eye_aspect_ratio > 0.3:
            return EmotionType.SURPRISED
        elif features.eyebrow_position < 0 and features.mouth_aspect_ratio < 0.4:
            return EmotionType.SAD
        elif features.eyebrow_position > 0.2:
            return EmotionType.THOUGHTFUL
        return EmotionType.NEUTRAL

    @staticmethod
    def _calculate_eye_ratio(eye_points: List[Tuple[int, int]]) -> float:
        width = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        height1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        height2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        return (height1 + height2) / (2.0 * width)

    @staticmethod
    def _calculate_mouth_ratio(mouth_points: List[Tuple[int, int]]) -> float:
        width = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))
        height = np.linalg.norm(np.array(mouth_points[3]) - np.array(mouth_points[9]))
        return height / width

    @staticmethod
    def _calculate_eyebrow_position(left_brow: List[Tuple[int, int]],
                                  right_brow: List[Tuple[int, int]]) -> float:
        left_height = np.mean([p[1] for p in left_brow])
        right_height = np.mean([p[1] for p in right_brow])
        return -(left_height + right_height) / 2

    @staticmethod
    def _calculate_symmetry(landmarks: List[Tuple[int, int]]) -> float:
        nose_bridge = np.mean([landmarks[27][0], landmarks[30][0]])
        symmetry_scores = []
        for left, right in zip(range(0, 8), range(16, 8, -1)):
            left_dist = abs(landmarks[left][0] - nose_bridge)
            right_dist = abs(landmarks[right][0] - nose_bridge)
            symmetry_scores.append(1 - abs(left_dist - right_dist) / max(left_dist, right_dist))
        return np.mean(symmetry_scores)

    @staticmethod
    def _determine_face_shape(jaw_points: List[Tuple[int, int]]) -> str:
        face_width = np.linalg.norm(np.array(jaw_points[0]) - np.array(jaw_points[16]))
        face_height = np.linalg.norm(np.array(jaw_points[8]) -
                                   np.mean([jaw_points[0], jaw_points[16]], axis=0))
        jaw_width = np.linalg.norm(np.array(jaw_points[4]) - np.array(jaw_points[12]))

        ratio = face_height / face_width
        jaw_ratio = jaw_width / face_width

        if ratio > 1.2:
            return "oval" if jaw_ratio > 0.78 else "oblong"
        elif ratio < 0.85:
            return "round"
        elif jaw_ratio < 0.72:
            return "heart"
        elif jaw_ratio > 0.78:
            return "square"
        return "diamond"

    def _calculate_confidence(self, face: dlib.rectangle,
                            landmarks: List[Tuple[int, int]]) -> float:
        try:
            face_area = face.width() * face.height()
            size_score = min(face_area / (640 * 480), 1.0)
            symmetry_score = self._calculate_symmetry(landmarks)

            distances = []
            for i in range(len(landmarks) - 1):
                dist = np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[i + 1]))
                distances.append(dist)

            landmark_score = 1 - np.std(distances) / np.mean(distances)

            confidence = (0.4 * size_score +
                        0.4 * symmetry_score +
                        0.2 * landmark_score)
            return float(np.clip(confidence, 0.0, 1.0))
        except Exception:
            return 0.5

    def _generate_feature_description(self, features: FacialFeatures,
                                    emotion: EmotionType) -> str:
        description = f"The person has a {features.face_shape} face shape "
        description += f"and appears to be {emotion.value}. "

        if features.eye_aspect_ratio > 0.3:
            description += "Their eyes are wide open. "
        elif features.eye_aspect_ratio < 0.2:
            description += "Their eyes are narrowed. "

        if features.mouth_aspect_ratio > 0.6:
            description += "They have a broad smile. "
        elif features.mouth_aspect_ratio < 0.3:
            description += "Their expression is reserved. "

        if features.symmetry_score > 0.9:
            description += "Their facial features are very symmetrical. "

        return description.strip()

    def visualize_analysis(self, image: np.ndarray,
                          analysis_result: Dict) -> np.ndarray:
        try:
            vis_image = image.copy()

            # Draw landmarks
            for point in analysis_result['landmarks']:
                cv2.circle(vis_image, point, 2, (0, 255, 0), -1)

            # Draw face rectangle
            rect = analysis_result['face_rect']
            cv2.rectangle(vis_image,
                         (rect['x'], rect['y']),
                         (rect['x'] + rect['width'], rect['y'] + rect['height']),
                         (0, 255, 0), 2)

            # Add text annotations
            y_offset = 30
            cv2.putText(vis_image,
                       f"Emotion: {analysis_result['emotion']}",
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y_offset += 30
            cv2.putText(vis_image,
                       f"Face Shape: {analysis_result['features']['face_shape']}",
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y_offset += 30
            cv2.putText(vis_image,
                       f"Confidence: {analysis_result['confidence']:.2f}",
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return vis_image
        except Exception as e:
            logger.error(f"Error creating analysis visualization: {str(e)}")
            raise