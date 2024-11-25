import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
import random
import json
from sklearn.model_selection import train_test_split


class CelebDFProcessor:
    def __init__(self, base_path, output_path):
        self.base_path = base_path
        self.output_path = output_path
        self.face_detector = MTCNN(
            margin=20,
            select_largest=True,
            post_process=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(
            output_path, 'processed_frames'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'checkpoints'), exist_ok=True)

        # Checkpoint file path
        self.checkpoint_path = os.path.join(
            output_path, 'checkpoints', 'processing_checkpoint.json')

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

    def process_video(self, video_path, label, frame_interval=10):
        """Process single video and extract faces"""
        frames_data = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    detection_result = self.face_detector.detect(frame_rgb)
                    if len(detection_result) == 3:
                        boxes, _, _ = detection_result
                    else:
                        boxes, _ = detection_result

                    if boxes is not None and len(boxes) > 0:
                        x1, y1, x2, y2 = boxes[0]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        face = frame_rgb[y1:y2, x1:x2]
                        face = cv2.resize(face, (128, 128))

                        frame_data = {
                            'video_path': video_path,
                            'frame_number': frame_count,
                            'label': label,
                            'face_array': face
                        }
                        frames_data.append(frame_data)

                except Exception as e:
                    print(
                        f"Error processing frame {frame_count} from {video_path}: {str(e)}")

            frame_count += 1

        cap.release()
        return frames_data

    def process_dataset(self, resume=False):
        """Process entire CelebDF dataset with checkpoint support"""
        test_videos = self.load_test_list()

        video_dirs = {
            'Celeb-real': 1,
            'Celeb-synthesis': 0,
            'YouTube-real': 1
        }

        all_data = []
        processed_videos = set()
        current_dir = None

        # Load checkpoint if resuming
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                processed_videos = set(checkpoint['processed_videos'])
                current_dir = checkpoint['current_dir']
                print(
                    f"Resuming from checkpoint. Already processed {len(processed_videos)} videos.")

                # Load previously processed data
                metadata_path = os.path.join(
                    self.output_path, 'partial_metadata.csv')
                if os.path.exists(metadata_path):
                    partial_df = pd.read_csv(metadata_path)
                    for _, row in partial_df.iterrows():
                        face_array = np.load(row['frame_path'])
                        frame_data = {
                            'video_path': row['video_path'],
                            'frame_number': row['frame_number'],
                            'label': row['label'],
                            'face_array': face_array
                        }
                        all_data.append(frame_data)

        # Process each directory
        for dir_name, label in video_dirs.items():
            # Skip directories until we reach the one we were processing
            if current_dir and dir_name != current_dir:
                continue
            current_dir = None  # Reset after finding the resume point

            dir_path = os.path.join(self.base_path, dir_name)
            videos = [f for f in os.listdir(
                dir_path) if f.endswith(('.mp4', '.avi'))]

            print(f"Processing {dir_name}...")
            for video in tqdm(videos):
                video_path = os.path.join(dir_path, video)

                # Skip if video is in test set or already processed
                if video in test_videos or video_path in processed_videos:
                    continue

                # Process video
                frames_data = self.process_video(video_path, label)
                all_data.extend(frames_data)
                processed_videos.add(video_path)

                # Save checkpoint periodically (e.g., after each video)
                self.save_checkpoint(list(processed_videos), dir_name)

                # Save partial results periodically
                if len(all_data) % 1000 == 0:  # Adjust this number as needed
                    partial_metadata = self.save_processed_data(
                        all_data,
                        metadata_filename='partial_metadata.csv'
                    )

        return all_data

    def save_processed_data(self, processed_data, metadata_filename='metadata.csv'):
        """Save processed data to disk"""
        metadata = []
        for i, data in enumerate(processed_data):
            frame_path = os.path.join(
                self.output_path, 'processed_frames', f'frame_{i}.npy')

            # Save face array
            np.save(frame_path, data['face_array'])

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
            unique_videos,
            test_size=val_size,
            random_state=42
        )

        train_df = metadata_df[metadata_df['video_path'].isin(train_videos)]
        val_df = metadata_df[metadata_df['video_path'].isin(val_videos)]

        return train_df, val_df


# Example usage:
def main(base_path, output_path, resume=False):
    # Initialize processor
    processor = CelebDFProcessor(
        base_path=base_path,
        output_path=output_path
    )

    # Process dataset with resume capability
    processed_data = processor.process_dataset(resume=resume)

    # Save final processed data
    metadata_df = processor.save_processed_data(processed_data)

    # Create train/val split
    train_df, val_df = processor.create_train_val_split(metadata_df)

    # Save splits
    train_df.to_csv(os.path.join(
        output_path, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val_metadata.csv'), index=False)


if __name__ == "__main__":
    base_path = 'C:\\Users\\aaron\\Documents\\df\\Detector\\df-detector-vid\\Celeb-DF-v2-copy'
    output_path = 'C:\\Users\\aaron\\Documents\\df\\Detector\\df-detector-vid\\output-2'
    # Set resume=True to continue from checkpoint
    main(base_path, output_path, resume=True)
