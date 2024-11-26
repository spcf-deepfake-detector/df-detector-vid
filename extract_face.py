import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import os
from typing import Optional


class SmoothFaceTracker:
    def __init__(
        self,
        device: str = None,
        min_face_size: int = 50,
        confidence_threshold: float = 0.7,
        smooth_factor: float = 0.7
    ):
        """
        Initialize face tracker with smoothing capabilities
        
        Args:
            device (str, optional): Compute device 
            min_face_size (int, optional): Minimum detectable face size
            confidence_threshold (float, optional): Detection confidence threshold
            smooth_factor (float, optional): Smoothing factor for face tracking (0-1)
        """
        # Set default device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize MTCNN detector
        self.mtcnn = MTCNN(
            keep_all=False,  # Track only the most prominent face
            device=device,
            min_face_size=min_face_size,
            thresholds=[confidence_threshold] * 3
        )

        # Tracking parameters
        self.smooth_factor = smooth_factor
        self.prev_bbox = None
        self.frame_count = 0
        self.lost_track_count = 0
        self.MAX_LOST_FRAMES = 10

    def smooth_bbox(self, bbox):
        """
        Apply exponential smoothing to bounding box
        
        Args:
            bbox (np.ndarray): Current detected bounding box
        
        Returns:
            np.ndarray: Smoothed bounding box
        """
        if self.prev_bbox is None:
            self.prev_bbox = bbox
            return bbox

        # Exponential smoothing
        smoothed = (
            self.smooth_factor * bbox +
            (1 - self.smooth_factor) * self.prev_bbox
        )

        self.prev_bbox = smoothed
        return smoothed.astype(np.int32)

    def extract_stabilized_face(
        self,
        frame,
        padding: float = 0.2,
        target_size: tuple = (224, 224)
    ):
        """
        Extract and stabilize face from frame
        
        Args:
            frame (np.ndarray): Input frame
            padding (float, optional): Additional padding around face
            target_size (tuple, optional): Resize output face to this size
        
        Returns:
            np.ndarray or None: Stabilized face image
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face
        try:
            detection_result = self.mtcnn.detect(frame_rgb)
            if len(detection_result) == 2:
                boxes, probs = detection_result
            else:
                boxes, probs, _ = detection_result
        except Exception:
            return None

        # No face detected
        if boxes is None or len(boxes) == 0:
            self.lost_track_count += 1
            if self.lost_track_count > self.MAX_LOST_FRAMES:
                # Reset tracking if face lost for too long
                self.prev_bbox = None
                self.lost_track_count = 0
            return None

        # Get the most confident face
        best_face_idx = np.argmax(probs)
        bbox = boxes[best_face_idx]

        # Smooth the bounding box
        bbox = self.smooth_bbox(bbox)

        # Add padding
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Calculate padded coordinates
        x1 = max(0, int(x1 - width * padding))
        y1 = max(0, int(y1 - height * padding))
        x2 = min(frame.shape[1], int(x2 + width * padding))
        y2 = min(frame.shape[0], int(y2 + height * padding))

        # Extract face
        face = frame[y1:y2, x1:x2]

        # Resize if needed
        if target_size:
            face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)

        # Reset lost track count
        self.lost_track_count = 0

        return face


def extract_smooth_faces_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    smooth_factor: float = 0.7,
    target_size: tuple = (224, 224)
):
    """
    Extract smooth, stabilized faces from video
    
    Args:
        video_path (str): Input video path
        output_path (str, optional): Output video path
        smooth_factor (float, optional): Smoothing intensity
        target_size (tuple, optional): Resize faces to this size
    """
    # Validate input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Prepare output
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_face{ext}"

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        target_size
    )

    # Initialize face tracker
    face_tracker = SmoothFaceTracker(smooth_factor=smooth_factor)

    # Process video
    face_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract stabilized face
            face = face_tracker.extract_stabilized_face(
                frame,
                target_size=target_size
            )

            if face is not None:
                out.write(face)
                face_count += 1

    except Exception as e:
        print(f"Error processing video: {e}")

    finally:
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Smooth face extraction completed:")
        print(f"  Input Video: {video_path}")
        print(f"  Output Video: {output_path}")
        print(f"  Faces Extracted: {face_count}")


if __name__ == "__main__":
    # Example usage
    video_path = r"C:\Users\aaron\Documents\df\Rope\Videos\1110919_asian_looking-at-camera_outside_import617a3753ce4662386673051080p12000br.mp4"
    output_path = None  # Automatically generate output path

    extract_smooth_faces_from_video(
        video_path,
        output_path,
        smooth_factor=0.7,  # Adjust smoothness (0-1)
        target_size=(224, 224)  # Resize faces
    )
