import tensorflow as tf
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(
    0) if torch.cuda.is_available() else "No GPU")

print("TensorFlow GPU available:", tf.config.list_physical_devices('GPU'))
