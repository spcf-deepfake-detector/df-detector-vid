import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
import cv2
import torch


class ImageDataProcessor:
    def __init__(self, base_path, output_path, use_face_detection=True):
        self.base_path = base_path
        self.output_path = output_path
        self.use_face_detection = use_face_detection

        if self.use_face_detection:
            self.face_detector = MTCNN(
                margin=20,
                select_largest=True,
                post_process=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(
            output_path, 'processed_images'), exist_ok=True)

    def process_image(self, image_path, label):
        """Process a single image and extract faces if needed"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_face_detection:
            try:
                boxes, _ = self.face_detector.detect(image_rgb)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    face = image_rgb[y1:y2, x1:x2]
                    face_resized = cv2.resize(face, (128, 128))
                    face_array = np.array(face_resized)
                else:
                    face_array = np.array(cv2.resize(image_rgb, (128, 128)))
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                face_array = np.array(cv2.resize(image_rgb, (128, 128)))
        else:
            face_array = np.array(cv2.resize(image_rgb, (128, 128)))

        return {
            'image_path': image_path,
            'label': label,
            'face_array': face_array
        }

    def process_dataset(self, image_list):
        """Process the entire image dataset"""
        processed_data = []
        for image_info in tqdm(image_list, desc="Processing images"):
            image_path, label = image_info
            image_path = os.path.join(self.base_path, image_path)
            if os.path.exists(image_path):
                data = self.process_image(image_path, label)
                processed_data.append(data)
            else:
                print(f"Warning: {image_path} does not exist.")

        return processed_data

    def save_processed_data(self, processed_data, metadata_filename='metadata.csv'):
        """Save processed data to disk"""
        metadata = []
        for i, data in enumerate(processed_data):
            image_path = os.path.join(
                self.output_path, 'processed_images', f'image_{i}.npy')

            # Save face array
            np.save(image_path, data['face_array'])

            # Add to metadata
            metadata.append({
                'image_path': image_path,
                'original_image_path': data['image_path'],
                'label': data['label']
            })

        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(
            self.output_path, metadata_filename), index=False)

        return metadata_df


# Example usage
if __name__ == "__main__":
    base_path = 'path/to/image_dataset'
    output_path = 'output_images'
    image_list = [
        ('image1.jpg', 0),
        ('image2.jpg', 1),
        # Add more images and labels
    ]

    processor = ImageDataProcessor(
        base_path, output_path, use_face_detection=True)
    processed_data = processor.process_dataset(image_list)
    processor.save_processed_data(processed_data)
