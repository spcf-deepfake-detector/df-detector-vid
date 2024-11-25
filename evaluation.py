import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example tensors for predictions and ground truth
# Replace these with your actual data
predictions = torch.tensor([1, 0, 1, 1, 0])  # Example predictions
ground_truth = torch.tensor([1, 0, 1, 0, 0])  # Example true labels

# Convert probabilities to class labels if needed (assuming binary classification)
# For example, if probs are [[0.99, 0.01], [0.02, 0.98]], argmax gives class labels [0, 1]
# predicted_classes = torch.argmax(probabilities, dim=1)

# Calculate metrics
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optional: Identify misclassified frames
misclassified_indices = (predictions != ground_truth).nonzero(as_tuple=True)[0]
print(f"Misclassified frames: {misclassified_indices}")
