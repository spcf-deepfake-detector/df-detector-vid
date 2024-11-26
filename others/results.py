import matplotlib.pyplot as plt

# Performance metrics
metrics = {
    'Accuracy': 97.31,
    'Precision': 96.02,
    'Recall': 84.90,
    'F1 Score': 90.12
}

# Extract metric names and values
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(metric_names, metric_values, color=[
               '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval,
             f'{yval:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add titles and labels
plt.title('Performance Metrics of CNN-based Deepfake Detection System',
          fontsize=16, fontweight='bold')
plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Percentage', fontsize=14)

# Set y-axis limit to 100%
plt.ylim(0, 100)

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Use a more professional font style
plt.rcParams.update({'font.family': 'serif'})

# Show the chart
plt.tight_layout()
plt.show()
