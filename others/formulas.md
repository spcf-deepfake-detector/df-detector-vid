# Combating Digital Deception: Development of a Robust Deepfake Detection System

## Abstract
In today's rapidly advancing technological landscape, the spread of misinformation has become a significant challenge. The rise of digital manipulation techniques, such as deepfake technology, has led to the proliferation of highly convincing yet fraudulent media content, deceiving countless individuals. Previous research in deepfake detection often relies on outdated datasets, limiting the effectiveness of these methods against evolving techniques.

To address this limitation, this study implements a Convolutional Neural Network (CNN) using the Celeb-DF (v2) dataset, alongside synthetic data generated using popular deepfake tools like DeepFaceLab and FaceSwap. By incorporating the latest advancements in deepfake creation, we developed a robust detection model capable of identifying new and sophisticated forgery methods. The findings underscore the importance of continuously updating detection systems with recent data to keep pace with rapidly evolving manipulation techniques. This approach is critical in the ongoing effort to combat digital misinformation effectively.

---

## Introduction
Deepfakes are synthetic media in which a person's likeness—such as their face, voice, or actions—is manipulated and replaced with someone else’s, using artificial intelligence and machine learning techniques. This technology poses significant risks, particularly in spreading misinformation, as it becomes increasingly sophisticated and accessible.

---

## Datasets Used for Detection Training

### Celeb-DF (v2): A New Dataset for DeepFake Forensics
- **Overview:** A large-scale, publicly available dataset designed specifically for deepfake forensics. It features high-quality fake videos that pose a significant challenge to detection models.  

### DeepFaceLab
- **Purpose:** A synthetic data generation tool widely used for creating face-swapped videos.  

### FaceSwap
- **Role:** Another prominent tool for generating synthetic data, contributing to the diversity and complexity of training datasets.

---

## CNN Architecture in the `DeepFakeDetector` Class

The `DeepFakeDetector` class consists of several convolutional layers followed by fully connected layers. Here is the architecture:

### 1. Convolutional Layers:
1. **Conv2d(3, 64, kernel_size=3, padding=1)**  
   - **ReLU Activation**  
   - **BatchNorm2d(64)**  
   - **Dropout2d(0.2)**  
   - **MaxPool2d(2, 2)**  

2. **Conv2d(64, 128, kernel_size=3, padding=1)**  
   - **ReLU Activation**  
   - **BatchNorm2d(128)**  
   - **Dropout2d(0.2)**  
   - **MaxPool2d(2, 2)**  

3. **Conv2d(128, 256, kernel_size=3, padding=1)**  
   - **ReLU Activation**  
   - **BatchNorm2d(256)**  
   - **Dropout2d(0.2)**  
   - **MaxPool2d(2, 2)**  

4. **Conv2d(256, 512, kernel_size=3, padding=1)**  
   - **ReLU Activation**  
   - **BatchNorm2d(512)**  
   - **Dropout2d(0.2)**  
   - **MaxPool2d(2, 2)**  

### 2. Fully Connected Layers:
1. **Linear(512 * 8 * 8, 1024)**  
   - **ReLU Activation**  
   - **BatchNorm1d(1024)**  
   - **Dropout(0.5)**  

2. **Linear(1024, 512)**  
   - **ReLU Activation**  
   - **BatchNorm1d(512)**  
   - **Dropout(0.5)**  

3. **Linear(512, 256)**  
   - **ReLU Activation**  
   - **BatchNorm1d(256)**  
   - **Dropout(0.5)**  

4. **Linear(256, 2)**  

---

## Formulas

### Convolutional Layer
The output size of a convolutional layer can be calculated using the formula:

\[
\text{Output Size} = \left( \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} \right) + 1
\]

**Example:** For the first convolutional layer:
- **Input Size:** \(128 \times 128\)  
- **Kernel Size:** \(3 \times 3\)  
- **Padding:** \(1\)  
- **Stride:** \(1\)  

\[
\text{Output Size} = \left( \frac{128 - 3 + 2 \times 1}{1} \right) + 1 = 128
\]

---

### Max Pooling Layer
The output size of a max pooling layer can be calculated using the formula:

\[
\text{Output Size} = \left( \frac{\text{Input Size}}{\text{Stride}} \right)
\]

**Example:** For the first max pooling layer:
- **Input Size:** \(128 \times 128\)  
- **Pool Size:** \(2 \times 2\)  
- **Stride:** \(2\)  

\[
\text{Output Size} = \left( \frac{128}{2} \right) = 64
\]

---

### Fully Connected Layer
The input to the first fully connected layer is the flattened output of the last convolutional layer. The size can be calculated as:

\[
\text{Flattened Size} = \text{Number of Filters} \times \text{Output Height} \times \text{Output Width}
\]

**Example:** For the last convolutional layer:
- **Number of Filters:** \(512\)  
- **Output Height:** \(8\)  
- **Output Width:** \(8\)  

\[
\text{Flattened Size} = 512 \times 8 \times 8 = 32,768
\]

---

## Summary
The `DeepFakeDetector` class uses a combination of convolutional and fully connected layers to extract features from input images and classify them as real or fake. The architecture includes dropout and batch normalization layers to improve generalization and prevent overfitting.
