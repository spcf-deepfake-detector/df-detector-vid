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
from deepfake_data import DeepFakeDetector
from facenet_pytorch import MTCNN

import matplotlib.pyplot as plt


class ImprovedDeepFakeVideoPredictor:
    def __init__(self,
                 model_path: str,
                 use_face_detection: bool = True,
                 use_face_only: bool = True,
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 visualize_frames_bool: bool = False):
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
        self.use_face_only = use_face_only

        # For visualization
        self.visualize_frames_bool = visualize_frames_bool

        # Initialize MTCNN with parameters matching training
        if self.use_face_detection:
            self.face_detector = MTCNN(
                image_size=128,  # Match training size
                margin=20,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                keep_all=False,
                device=self.device,
            )

        # Match the exact preprocessing used in training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def preprocess_frame(self, frame: np.ndarray) -> Optional[Tuple[Optional[torch.Tensor], np.ndarray]]:
        """
        Preprocessing with reduced redundant operations
        """
        # Convert BGR to RGB only once
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_face_detection:
            try:
                # Detect faces
                boxes, _ = self.face_detector.detect(frame_rgb)  # type: ignore
            except Exception:
                return None, frame_rgb

            if boxes is not None and len(boxes) > 0:
                # Extract the first detected face
                x1, y1, x2, y2 = map(int, boxes[0])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(
                    x2, frame_rgb.shape[1]), min(y2, frame_rgb.shape[0])

                try:
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    face_resized = cv2.resize(
                        face_crop, (128, 128), interpolation=cv2.INTER_AREA)
                    face_tensor = torch.from_numpy(
                        face_resized).permute(2, 0, 1).float() / 255.0

                    if self.use_face_only:
                        return face_tensor, frame_rgb
                    else:
                        frame_tensor = torch.from_numpy(
                            frame_rgb).permute(2, 0, 1).float() / 255.0
                        return frame_tensor, frame_rgb
                except Exception:
                    return None, frame_rgb

            return None, frame_rgb
        else:
            # Non-face detection path
            frame_resized = cv2.resize(
                frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)
            frame_tensor = torch.from_numpy(
                frame_resized).permute(2, 0, 1).float() / 255.0
            return frame_tensor, frame_rgb

    def predict_video(self,
                      video_path: str,
                      sample_rate: int = 1,
                      threshold: float = 0.5,
                      temperature: float = 2.0) -> Dict:
        """
        Predict whether a video is real or fake using improved processing.
        """
        cap = cv2.VideoCapture(video_path)
        real_probs = []
        fake_probs = []
        valid_frames = 0
        total_frames = 0
        processed_tensors = []  # Collect the tensors

        with tqdm(desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if total_frames % sample_rate == 0:
                    result = self.preprocess_frame(frame)
                    if result is not None:
                        processed_tensor, frame_rgb = result
                    else:
                        processed_tensor, frame_rgb = None, None

                    if processed_tensor is not None:
                        processed_tensor = processed_tensor.unsqueeze(
                            0).to(self.device)

                        with torch.no_grad():
                            # Apply temperature scaling
                            output = self.model(
                                processed_tensor, temperature=temperature)
                            probabilities = torch.softmax(output, dim=1)

                            real_prob = probabilities[0][0].item()
                            fake_prob = probabilities[0][1].item()
                            real_probs.append(real_prob)
                            fake_probs.append(fake_prob)
                            valid_frames += 1

                        # Collect the processed tensor
                        processed_tensors.append(processed_tensor)

                total_frames += 1
                pbar.update(1)

        cap.release()

        if valid_frames == 0:
            return {
                'error': 'No valid frames processed',
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'message': 'No faces detected in the video.'
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
            'std_fake': std_fake,
        }

        # Visualize the face tensors
        if self.visualize_frames_bool:
            self.visualize_face_tensors(processed_tensors)

        return results

    def visualize_face_tensors(self, face_tensors: List[Union[torch.Tensor, np.ndarray]], grid_size: int = 4):
        """
        Visualize the processed face tensors using Matplotlib.
        """
        num_faces = len(face_tensors)
        num_plots = grid_size * grid_size

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 8))
        axes = axes.flatten()

        for ax in axes:
            ax.axis('off')

        for i, face_tensor in enumerate(face_tensors):
            ax = axes[i % num_plots]
            if isinstance(face_tensor, torch.Tensor):
                # Ensure the tensor has the correct number of dimensions
                face_rgb = face_tensor.squeeze(
                    0).permute(1, 2, 0).cpu().numpy()
                # Convert from [0, 1] range to [0, 255] range for visualization
                face_rgb = (face_rgb * 255).astype(np.uint8)
            else:
                face_rgb = face_tensor
            ax.imshow(face_rgb)
            ax.axis('off')

            if (i + 1) % num_plots == 0 or i == num_faces - 1:
                plt.tight_layout()
                # Use pause instead of show to keep the figure open
                plt.pause(0.001)
                if i != num_faces - 1:
                    fig, axes = plt.subplots(
                        grid_size, grid_size, figsize=(12, 8))
                    axes = axes.flatten()
                    for ax in axes:
                        ax.axis('off')

    def visualize_frames(self, frames: List[np.ndarray], grid_size: int = 4):
        """
        Visualize the processed frames using Matplotlib.
        """
        num_frames = len(frames)
        num_plots = grid_size * grid_size

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 8))
        axes = axes.flatten()

        for ax in axes:
            ax.axis('off')

        for i, frame in enumerate(frames):
            ax = axes[i % num_plots]
            ax.imshow(frame)
            ax.axis('off')

            if (i + 1) % num_plots == 0 or i == num_frames - 1:
                plt.tight_layout()
                # Use pause instead of show to keep the figure open
                plt.pause(0.001)
                if i != num_frames - 1:
                    fig, axes = plt.subplots(
                        grid_size, grid_size, figsize=(12, 8))
                    axes = axes.flatten()
                    for ax in axes:
                        ax.axis('off')


# Example usage
if __name__ == "__main__":
    predictor = ImprovedDeepFakeVideoPredictor(
        model_path='models/best_model.pth',
        use_face_detection=True,
        use_face_only=False,
        visualize_frames_bool=False,
    )

    # video_path = r"path/to/your/video.mp4"
    # List of video paths to process
    video_paths = [
        r"path/to/your/video.mp4"
    ]
    for video_path in video_paths:
        results = predictor.predict_video(
            video_path=video_path,
            sample_rate=2,
            threshold=0.5,
        )

        print(f"Video: {video_path}")
        if 'message' in results:
            print(results['message'])
        else:
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
