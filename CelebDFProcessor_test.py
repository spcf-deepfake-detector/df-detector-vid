import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch


class CelebDFProcessor:
    def __init__(self, base_path, output_path):
        self.base_path = base_path
        self.output_path = output_path
        self.processed_frames_dir = os.path.join(
            output_path, 'processed_frames')
        os.makedirs(self.processed_frames_dir, exist_ok=True)

        # Initialize MTCNN for face detection
        self.face_detector = MTCNN(
            margin=20,
            select_largest=True,
            post_process=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def load_test_list(self):
        """Load testing videos list with enhanced error handling"""
        test_list_path = os.path.join(
            self.base_path, 'List_of_testing_videos.txt')
        if not os.path.exists(test_list_path):
            raise FileNotFoundError(
                f"Test list file not found: {test_list_path}")

        test_videos = []
        with open(test_list_path, 'r') as f:
            for line in f:
                label, video_path = line.strip().split(' ', 1)
                test_videos.append((int(label), video_path))

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

    def save_processed_data(self, processed_data, metadata_filename='test_metadata.csv'):
        """Save processed data to disk"""
        metadata = []
        for i, data in enumerate(processed_data):
            frame_path = os.path.join(
                self.processed_frames_dir, f'frame_{i}.npy')
            np.save(frame_path, data['face_array'])
            metadata.append({
                'frame_path': frame_path,
                'video_path': data['video_path'],
                'frame_number': data['frame_number'],
                'label': data['label']
            })

        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(
            self.output_path, metadata_filename), index=False)

    def process_test_videos(self):
        """Process test videos listed in List_of_testing_videos.txt"""
        test_videos = self.load_test_list()

        test_data = []

        for label, video_path in tqdm(test_videos, desc="Processing test videos"):
            full_video_path = os.path.join(self.base_path, video_path)
            frames_data = self.process_video(full_video_path, label)
            test_data.extend(frames_data)

        self.save_processed_data(test_data)


if __name__ == "__main__":
    base_path = 'Celeb-DF-v2'
    output_path = 'output_test'
    processor = CelebDFProcessor(base_path, output_path)
    processor.process_test_videos()
