import mediapipe as mp
import cv2
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseType(Enum):
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    UNKNOWN = "unknown"

@dataclass
class PoseMetrics:
    symmetry_score: float
    balance_score: float
    stability_score: float
    pose_type: PoseType
    confidence: float

class PoseDetector:
    def __init__(self):
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            logger.error(f"Error initializing PoseDetector: {str(e)}")
            raise

    def detect_pose(self, image: np.ndarray) -> Dict:
        """Detect pose in image and return pose data including landmarks and pose map."""
        try:
            # Convert the image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if not results.pose_landmarks:
                raise ValueError("No pose detected in the image")

            # Create pose visualization map for ControlNet
            pose_map = np.zeros_like(image)
            self.mp_drawing.draw_landmarks(
                pose_map,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Convert to white lines on black background
            pose_map_gray = cv2.cvtColor(pose_map, cv2.COLOR_BGR2GRAY)
            _, pose_map_binary = cv2.threshold(pose_map_gray, 1, 255, cv2.THRESH_BINARY)

            # Calculate metrics and pose type
            metrics = self._calculate_pose_metrics(results.pose_landmarks)

            return {
                'pose_map': pose_map_binary,
                'pose_type': metrics.pose_type.value,
                'metrics': {
                    'balance': metrics.balance_score,
                    'stability': metrics.stability_score,
                    'symmetry': metrics.symmetry_score
                },
                'confidence': metrics.confidence
            }

        except Exception as e:
            logger.error(f"Error in pose detection: {str(e)}")
            raise

    def _calculate_pose_metrics(self, landmarks) -> PoseMetrics:
        """Calculate pose metrics from landmarks."""
        try:
            # Extract key landmarks for calculations
            l_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

            # Convert landmarks to numpy arrays for calculations
            l_shoulder_pos = np.array([l_shoulder.x, l_shoulder.y])
            r_shoulder_pos = np.array([r_shoulder.x, r_shoulder.y])
            l_hip_pos = np.array([l_hip.x, l_hip.y])
            r_hip_pos = np.array([r_hip.x, r_hip.y])

            # Calculate body center points
            shoulder_center = (l_shoulder_pos + r_shoulder_pos) / 2
            hip_center = (l_hip_pos + r_hip_pos) / 2

            # Calculate symmetry score based on shoulders and hips alignment
            shoulder_width = np.linalg.norm(l_shoulder_pos - r_shoulder_pos)
            hip_width = np.linalg.norm(l_hip_pos - r_hip_pos)
            width_ratio = min(shoulder_width, hip_width) / max(shoulder_width, hip_width)
            symmetry_score = (width_ratio +
                            (1 - abs(l_shoulder_pos[1] - r_shoulder_pos[1])) +
                            (1 - abs(l_hip_pos[1] - r_hip_pos[1]))) / 3

            # Calculate balance score based on vertical alignment
            vertical_alignment = 1 - abs(shoulder_center[0] - hip_center[0])
            balance_score = float(np.clip(vertical_alignment, 0, 1))

            # Calculate stability score based on pose distribution
            heights = [lm.y for lm in landmarks.landmark if lm.visibility > 0.5]
            height_distribution = np.std(heights) if heights else 1.0
            stability_score = float(np.clip(1 - height_distribution, 0, 1))

            # Calculate average confidence from landmark visibilities
            confidence = float(np.mean([
                lm.visibility for lm in landmarks.landmark
                if hasattr(lm, 'visibility')
            ]))

            # Determine pose type
            pose_type = self._determine_pose_type(landmarks)

            return PoseMetrics(
                symmetry_score=float(symmetry_score),
                balance_score=float(balance_score),
                stability_score=float(stability_score),
                pose_type=pose_type,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error calculating pose metrics: {str(e)}")
            raise

    def _determine_pose_type(self, landmarks) -> PoseType:
        """Determine the type of pose based on landmark positions."""
        try:
            # Get relevant landmarks
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            l_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            l_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            r_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            l_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            r_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Calculate average heights
            hip_height = (l_hip.y + r_hip.y) / 2
            knee_height = (l_knee.y + r_knee.y) / 2
            ankle_height = (l_ankle.y + r_ankle.y) / 2

            # Get vertical distances
            hip_knee_dist = abs(hip_height - knee_height)
            knee_ankle_dist = abs(knee_height - ankle_height)

            # Determine pose type based on relative positions and angles
            if hip_height < 0.6 and knee_height > hip_height:
                return PoseType.STANDING
            elif (0.4 < hip_height < 0.8 and
                  hip_knee_dist < 0.2 and
                  knee_ankle_dist > 0.2):
                return PoseType.SITTING
            elif hip_height > 0.7 and abs(nose.y - hip_height) < 0.3:
                return PoseType.LYING
            else:
                return PoseType.UNKNOWN

        except Exception as e:
            logger.error(f"Error determining pose type: {str(e)}")
            return PoseType.UNKNOWN

    def visualize_pose(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Create a visualization of the detected pose on the input image."""
        try:
            vis_image = image.copy()
            self.mp_drawing.draw_landmarks(
                vis_image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            return vis_image
        except Exception as e:
            logger.error(f"Error visualizing pose: {str(e)}")
            raise