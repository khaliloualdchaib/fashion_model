import cv2
import os
import numpy as np

def compute_mean_std(dataset_path):
    channel_sums = np.zeros(3)
    channel_sums_squared = np.zeros(3)
    total_pixels = 0

    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

            # Accumulate sums and squared sums
            channel_sums += np.sum(image, axis=(0, 1))
            channel_sums_squared += np.sum(image ** 2, axis=(0, 1))
            total_pixels += image.shape[0] * image.shape[1]

    mean = channel_sums / total_pixels
    std = np.sqrt((channel_sums_squared / total_pixels) - mean ** 2)

    return mean, std

# Example usage
dataset_path = "images"
mean, std = compute_mean_std(dataset_path)
print("Channel-wise Mean:", mean)
print("Channel-wise Std Dev:", std)
