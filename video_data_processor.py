import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
import json
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Any, Optional
from PIL import Image


class VideoDataProcessor:
    def __init__(self,
                 base_path: str,
                 output_path: str,
                 use_face_detection: bool = True,
                 frame_sampling_rate: int = 10,
                 num_workers: Optional[int] = None,
                 output_format: str = 'npy'):
        """
        Initialize video data processor with configurable parameters

        Args:
            base_path (str): Root directory of video dataset
            output_path (str): Directory to save processed data
            use_face_detection (bool): Whether to detect and crop faces
            frame_sampling_rate (int): Process every nth frame
            num_workers (int): Number of parallel processing workers
            output_format (str): Output format for frames ('npy' or 'image')
        """
        self.base_path = base_path
        self.output_path = output_path
        self.use_face_detection = use_face_detection
        self.frame_sampling_rate = frame_sampling_rate
        self.num_workers = num_workers or (os.cpu_count() or 1)
        self.output_format = output_format

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize face detector
        self.face_detector = self._setup_face_detector() if use_face_detection else None

        # Create output directories
        self._create_output_directories()

    def _setup_face_detector(self):
        """Setup face detection with GPU acceleration if available"""
        try:
            return MTCNN(
                margin=20,
                select_largest=True,
                post_process=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        except Exception as e:
            self.logger.error(f"Face detector setup failed: {e}")
            return None

    def _create_output_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path,
                    'processed_frames'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path,
                    'checkpoints'), exist_ok=True)

        # Checkpoint file path
        self.checkpoint_path = os.path.join(
            self.output_path, 'checkpoints', 'processing_checkpoint.json')

    def save_checkpoint(self, processed_videos, current_dir=None):
        """Save processing checkpoint"""
        checkpoint_data = {
            'processed_videos': processed_videos,
            'current_dir': current_dir
        }

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

    def load_checkpoint(self):
        """Load processing checkpoint if exists"""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data
        return None

    def load_test_list(self):
        """Load testing videos list"""
        test_list_path = os.path.join(
            self.base_path, 'List_of_testing_videos.txt')
        with open(test_list_path, 'r') as f:
            test_videos = f.read().splitlines()
        return test_videos

    def process_video(self, video_path: str, label: int) -> List[Dict[str, Any]]:
        """
        Extract frames from video with optional face detection

        Args:
            video_path (str): Path to video file
            label (int): Label for the video

        Returns:
            List of processed frame dictionaries
        """
        frames_data = []

        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames
                if frame_count % self.frame_sampling_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame = self._extract_frame(
                        frame_rgb, video_path, frame_count, label)

                    if processed_frame:
                        frames_data.append(processed_frame)

                frame_count += 1

            cap.release()
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")

        return frames_data

    def _extract_frame(self, frame_rgb, video_path, frame_count, label):
        """Extract and process a single frame"""
        if self.use_face_detection and self.face_detector:
            try:
                detection_result = self.face_detector.detect(frame_rgb)

                # Handle different MTCNN detection output formats
                boxes = detection_result[0] if len(
                    detection_result) == 3 else detection_result[0]

                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    face = frame_rgb[y1:y2, x1:x2]
                    face = cv2.resize(face, (128, 128))
                else:
                    return None
            except Exception as e:
                self.logger.warning(
                    f"Face detection failed for frame {frame_count}: {e}")
                return None
        else:
            # Resize entire frame if no face detection
            face = cv2.resize(frame_rgb, (128, 128))

        return {
            'video_path': video_path,
            'frame_number': frame_count,
            'label': label,
            'face_array': face
        }

    def process_dataset(self, video_dirs: Dict[str, int], resume: bool = False) -> pd.DataFrame:
        """
        Process entire dataset with checkpoint support

        Args:
            video_dirs (dict): Directories with label mapping
            resume (bool): Whether to resume from previous checkpoint

        Returns:
            Metadata DataFrame containing all processed frames
        """
        processed_videos = set()
        current_dir = None

        # Load checkpoint if resuming
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                processed_videos = set(checkpoint['processed_videos'])
                current_dir = checkpoint['current_dir']
                self.logger.info(
                    f"Resuming from checkpoint. Processed {len(processed_videos)} videos.")

        # Load or create metadata DataFrame
        metadata_path = os.path.join(self.output_path, 'partial_metadata.csv')
        if os.path.exists(metadata_path):
            metadata_df = pd.read_csv(metadata_path)
        else:
            metadata_df = pd.DataFrame(columns=[
                'frame_path', 'video_path', 'frame_number', 'label'
            ])

        # Process videos
        for dir_name, label in video_dirs.items():
            # Skip directories until we reach the one we were processing
            if current_dir and dir_name != current_dir:
                continue
            current_dir = None  # Reset after finding the resume point

            dir_path = os.path.join(self.base_path, dir_name)
            videos = [f for f in os.listdir(
                dir_path) if f.endswith(('.mp4', '.avi'))]

            for video_file in videos:
                video_path = os.path.join(dir_path, video_file)

                # Skip if already processed
                if video_path in processed_videos:
                    continue

                # Process video
                frames_data = self.process_video(video_path, label)

                # Save processed frames and update metadata
                for frame_data in frames_data:
                    frame_path = self._save_frame(frame_data)
                    metadata_df = pd.concat([metadata_df, pd.DataFrame([{
                        'frame_path': frame_path,
                        'video_path': frame_data['video_path'],
                        'frame_number': frame_data['frame_number'],
                        'label': frame_data['label']
                    }])], ignore_index=True)

                # Mark video as processed
                processed_videos.add(video_path)

                # Save checkpoint after each video
                self.save_checkpoint(list(processed_videos), dir_name)

                # Save metadata periodically
                if len(metadata_df) % 1000 == 0:
                    metadata_df.to_csv(metadata_path, index=False)

        # Save final metadata
        metadata_df.to_csv(metadata_path, index=False)
        return metadata_df

    def _save_frame(self, frame_data: Dict[str, Any]) -> str:
        """
        Save a single frame to disk and return its file path

        Args:
            frame_data (dict): Dictionary containing frame data

        Returns:
            str: Path to the saved frame
        """
        frame_index = len(os.listdir(os.path.join(
            self.output_path, 'processed_frames')))
        if self.output_format == 'npy':
            frame_path = os.path.join(
                self.output_path, 'processed_frames', f'frame_{frame_index}.npy')
            np.save(frame_path, frame_data['face_array'])
        else:
            frame_path = os.path.join(
                self.output_path, 'processed_frames', f'frame_{frame_index}.jpg')
            face_image = Image.fromarray(frame_data['face_array'])
            face_image.save(frame_path)
        return frame_path

    def save_processed_data(self, processed_data, metadata_filename='metadata.csv'):
        """Save processed data to disk"""
        metadata = []
        for i, data in enumerate(processed_data):
            if self.output_format == 'npy':
                frame_path = os.path.join(
                    self.output_path, 'processed_frames', f'frame_{i}.npy')
                # Save face array as .npy file
                np.save(frame_path, data['face_array'])
            else:
                frame_path = os.path.join(
                    self.output_path, 'processed_frames', f'frame_{i}.jpg')
                # Save face array as image
                face_image = Image.fromarray(data['face_array'])
                face_image.save(frame_path)

            # Add to metadata
            metadata.append({
                'frame_path': frame_path,
                'video_path': data['video_path'],
                'frame_number': data['frame_number'],
                'label': data['label']
            })

        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(
            self.output_path, metadata_filename), index=False)

        return metadata_df

    def create_train_val_split(self, metadata_df, val_size=0.2):
        """Create train/validation split"""
        unique_videos = metadata_df['video_path'].unique()

        train_videos, val_videos = train_test_split(
            unique_videos, test_size=val_size, random_state=42)

        train_df = metadata_df[metadata_df['video_path'].isin(train_videos)]
        val_df = metadata_df[metadata_df['video_path'].isin(val_videos)]

        return train_df, val_df

# Example usage:


def main(base_path: str, output_path: str, resume: bool = False, output_format: str = 'npy'):
    """Main processing pipeline"""
    video_dirs = {
        'Celeb-real': 1,
        'Celeb-synthesis': 0,
        'YouTube-real': 1
    }

    processor = VideoDataProcessor(
        base_path=base_path,  # Root directory of video dataset
        output_path=output_path,  # Output directory
        use_face_detection=True,  # Enable face detection
        frame_sampling_rate=15,  # Configurable sampling
        num_workers=None,  # Auto-detect cores
        output_format=output_format  # Output format ('npy' or 'image')
    )

    # Process dataset with resume option
    # metadata_df is already in the correct format, no need to process it again
    metadata_df = processor.process_dataset(video_dirs, resume=resume)

    # Create and save train/val splits directly from metadata_df
    train_df, val_df = processor.create_train_val_split(metadata_df)

    # Save final splits
    train_df.to_csv(os.path.join(
        output_path, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val_metadata.csv'), index=False)


if __name__ == "__main__":
    main('Celeb-DF-v2', 'output_test_1_image',
         resume=True, output_format='image')
