import torch
from torch import nn
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from typing import Union, List, Tuple, Dict, Optional
from collections import defaultdict
import os
from tqdm import tqdm
from deepFakeDataSet_checkpoints import DeepFakeDetector
from facenet_pytorch import MTCNN


class ImprovedDeepFakeVideoPredictor:
    def __init__(self,
                 model_path: str,
                 use_face_detection: bool = True,
                 device: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize the improved DeepFake video predictor with a trained model.
        """
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = DeepFakeDetector()
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size
        self.use_face_detection = use_face_detection

        # Initialize MTCNN with parameters matching training
        if self.use_face_detection:
            self.face_detector = MTCNN(
                image_size=128,  # Match training size
                margin=20,  # No margin to match training
                min_face_size=60,
                thresholds=[0.6, 0.7, 0.9],
                factor=0.709,
                post_process=True,
                device=self.device,
            )

        # Match the exact preprocessing used in training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def preprocess_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess a single frame for inference, matching training preprocessing.
        """
        if self.use_face_detection:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect and align face
            face_tensor = self.face_detector(frame_rgb)

            if face_tensor is None:
                # If no face detected, return None
                return None

            # Normalize to [0, 1] range
            face_tensor = face_tensor.float() / 255.0
            return face_tensor
        else:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to match training size
            frame_resized = cv2.resize(frame_rgb, (128, 128))
            # Convert to tensor and normalize
            frame_tensor = torch.from_numpy(
                frame_resized).float().permute(2, 0, 1) / 255.0
            return frame_tensor

    def predict_video(self,
                      video_path: str,
                      sample_rate: int = 1,
                      threshold: float = 0.5,
                      temperature: float = 2.0) -> Dict:  # Added temperature parameter
        """
        Predict whether a video is real or fake using improved processing.
        """
        cap = cv2.VideoCapture(video_path)
        real_probs = []
        fake_probs = []
        valid_frames = 0
        total_frames = 0

        with tqdm(desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if total_frames % sample_rate == 0:
                    processed_frame = self.preprocess_frame(frame)

                    if processed_frame is not None:
                        processed_frame = processed_frame.unsqueeze(
                            0).to(self.device)

                        with torch.no_grad():
                            # Apply temperature scaling
                            output = self.model(
                                processed_frame, temperature=temperature)
                            probabilities = torch.softmax(output, dim=1)

                            real_prob = probabilities[0][0].item()
                            fake_prob = probabilities[0][1].item()
                            real_probs.append(real_prob)
                            fake_probs.append(fake_prob)
                            valid_frames += 1

                total_frames += 1
                pbar.update(1)

        cap.release()

        if valid_frames == 0:
            return {
                'error': 'No valid frames processed',
                'prediction': 'UNKNOWN',
                'confidence': 0.0
            }

        # Calculate metrics
        avg_real_prob = np.mean(real_probs)
        avg_fake_prob = np.mean(fake_probs)
        std_real = np.std(real_probs)
        std_fake = np.std(fake_probs)

        # More nuanced prediction logic
        if avg_fake_prob > threshold:
            prediction = 'FAKE'
            confidence = avg_fake_prob
        else:
            prediction = 'REAL'
            confidence = avg_real_prob

        # Adjust confidence based on prediction consistency
        confidence = confidence * (1 - (std_real + std_fake) / 2)

        results = {
            'prediction': prediction,
            'confidence': confidence,
            'fake_ratio': avg_fake_prob,
            'total_frames': total_frames,
            'valid_frames': valid_frames,
            'avg_real_prob': avg_real_prob,
            'avg_fake_prob': avg_fake_prob,
            'std_real': std_real,
            'std_fake': std_fake
        }

        return results


# Example usage
if __name__ == "__main__":
    predictor = ImprovedDeepFakeVideoPredictor(
        model_path='training_output_2/checkpoints/best_model.pth',
        use_face_detection=False
    )

    # video_path = "path/to/your/video.mp4"
    # video_path = r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_alex.mp4"
    # List of video paths to process
    video_paths = [
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\1110702_human_male_looking-at-camera_import6174f695850139280054171080p12000br_1728357844.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\1110702_human_male_looking-at-camera_import6174f695850139280054171080p12000br_1728359604.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\5997642_beach_sand_sea_h22f32a0cV02300854k25000br_1728358708.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\6013339_romantic_paradise_person_h4c11f82aV020124k25000br_1728358392.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\jose.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\paz.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\paz_0.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\paz_2.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\paz_ff.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker 4_1728357437.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_aaron.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_alex.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_alex_analyzed.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_charles.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_gab.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_gab_2.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_jose.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_paz.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_paz_2.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter parker_paz_2_analyzed.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\peter_parker_paz_ff.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\protects-the-city-from-the-weather_1728355826.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\quiana_1.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\quiana_2.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\quiana_3.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\quiana_4.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\quiana_5.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\quiana_6_ff.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\quiana_7_ff.mp4",
        r"C:\Users\aaron\Documents\df\Rope\video_outputs\quiana_8_ff.mp4"
    ]
    for video_path in video_paths:
        results = predictor.predict_video(
            video_path=video_path,
            sample_rate=2,
            threshold=0.5
        )

        print(f"Video: {video_path}")
        print(f"\nVideo Analysis Results:")
        print(f"Prediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"Fake ratio: {results['fake_ratio']:.2%}")
        print(
            f"Valid frames: {results['valid_frames']} / {results['total_frames']}")
        print('Average real probability:', results['avg_real_prob'])
        print('Average fake probability:', results['avg_fake_prob'])
        print('Real probability standard deviation:', results['std_real'])
        print('Fake probability standard deviation:', results['std_fake'])
