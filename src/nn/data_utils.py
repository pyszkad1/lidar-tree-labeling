from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class TrunkDataset(Dataset):
    def __init__(self, data_dir, exclude_oldest=0):
        self.data_dir = data_dir
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy') and not f.endswith('.bin.npy')]

        # Exclude a specified number of the oldest files
        all_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
        if exclude_oldest > 0 and exclude_oldest < len(all_files):
            all_files = all_files[exclude_oldest:]

        self.data_files = all_files
        self.mask_files = [f.replace('.npy', '.bin.npy') for f in self.data_files]

    def __len__(self):
        return len(self.data_files)

    def get_batch_size(self):
        return len(self.data_files) // 4

    def __getitem__(self, idx):
        # Load input data
        input_path = os.path.join(self.data_dir, self.data_files[idx])
        input_data = np.load(input_path)

        # Load mask data
        mask_path = os.path.join(self.data_dir, self.mask_files[idx])
        target_mask = np.load(mask_path)

        # Convert arrays to tensors
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)  # Add channel dimension
        target_tensor = torch.from_numpy(target_mask).float().unsqueeze(0)  # Add channel dimension

        return input_tensor, target_tensor


def save_plot(csv_file, loss_plot_file, accuracy_plot_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Plot for loss
    plt.figure(figsize=(10, 6))
    plt.plot(data['Epoch'], data['Training Loss'], label='Training Loss')
    plt.plot(data['Epoch'], data['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(loss_plot_file)
    plt.close()

    # Plot for accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(data['Epoch'], data['Training Accuracy'], label='Training Accuracy')
    plt.plot(data['Epoch'], data['Validation Accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(accuracy_plot_file)
    plt.close()


def rename_files(starting_number, directory='data'):
    # List all files in the specified directory

    files = os.listdir(directory)

    # Sort files to maintain order if needed
    files.sort()

    # Rename each file
    for i, filename in enumerate(files):
        # Define new filename, prepending the starting number
        number = str(starting_number + i).zfill(2)
        new_filename = f"{number}_{filename}"
        # Define the full old and new file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_filename}'")


def distribute_files(source_dir, target_dirs):
    # Create a dictionary to store groups of files
    file_groups = defaultdict(list)

    # List all files in the source directory
    for filename in os.listdir(source_dir):
        # Extract the number prefix assuming the format "number.extension"
        prefix = filename.split('.')[0]
        # Group files by their number prefix
        file_groups[prefix].append(filename)

    # Prepare the target directories by ensuring they exist
    for dir_path in target_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Flatten the list of groups into a single list to evenly distribute them
    grouped_files = list(file_groups.values())

    # Distribute the groups evenly across the target directories
    for index, group in enumerate(grouped_files):
        # Determine which directory to use based on the index
        target_dir = target_dirs[index % len(target_dirs)]
        # Copy all files in the group to the selected directory
        for filename in group:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
            print(f"Copied {filename} to {target_dir}")

def jaccard_index(true_mask, pred_mask):
    true_mask = np.asarray(true_mask, dtype=bool)
    pred_mask = np.asarray(pred_mask, dtype=bool)
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    if union.sum() == 0:
        return 1.0 if intersection.sum() == 0 else 0.0
    return intersection.sum() / union.sum()

def precision_score(true_mask, pred_mask):
    true_mask = np.asarray(true_mask, dtype=bool)
    pred_mask = np.asarray(pred_mask, dtype=bool)
    true_positive = np.logical_and(true_mask, pred_mask).sum()
    predicted_positive = pred_mask.sum()
    if predicted_positive == 0:
        return 1.0 if true_positive == 0 else 0.0
    return true_positive / predicted_positive

def accuracy_score(true_mask, pred_mask):
    true_mask = np.asarray(true_mask, dtype=bool)
    pred_mask = np.asarray(pred_mask, dtype=bool)
    correct_predictions = np.equal(true_mask, pred_mask).sum()
    total_elements = true_mask.size
    return correct_predictions / total_elements


def compute_metrics_for_directory(pred_dir, true_dir):
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.bin.npy')]
    jaccard_scores = {}
    precision_scores = {}
    accuracy_scores = {}

    for pred_file in pred_files:
        pred_mask_path = os.path.join(pred_dir, pred_file)
        true_mask_path = os.path.join(true_dir, pred_file)

        if os.path.exists(true_mask_path):
            pred_mask = np.load(pred_mask_path)
            true_mask = np.load(true_mask_path)

            jaccard = jaccard_index(true_mask, pred_mask)
            precision = precision_score(true_mask, pred_mask)
            accuracy = accuracy_score(true_mask, pred_mask)

            jaccard_scores[pred_file] = jaccard
            precision_scores[pred_file] = precision
            accuracy_scores[pred_file] = accuracy
        else:
            print(f"No corresponding true mask found for {pred_file}")

    # Calculate averages
    avg_jaccard = np.mean(list(jaccard_scores.values())) if jaccard_scores else 0
    avg_precision = np.mean(list(precision_scores.values())) if precision_scores else 0
    avg_accuracy = np.mean(list(accuracy_scores.values())) if accuracy_scores else 0

    return jaccard_scores, precision_scores, accuracy_scores, avg_jaccard, avg_precision, avg_accuracy


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    pred_directory = os.path.join(project_dir, 'data', 'pred4_25e')
    true_directory = os.path.join(project_dir, 'data', 'true_labels')

    jaccard_scores, precision_scores, accuracy_scores, avg_jaccard, avg_precision, avg_accuracy = compute_metrics_for_directory(pred_directory, true_directory)

    # Determine the file with the best Jaccard score
    best_jaccard_file = max(jaccard_scores, key=jaccard_scores.get)
    best_jaccard_score = jaccard_scores[best_jaccard_file]

    # Prepare the results text
    results_text = f"Jaccard Score: {avg_jaccard}\n"
    results_text += f"Best Jaccard Score: {best_jaccard_score} in file {best_jaccard_file}\n"
    results_text += f"Precision Score: {avg_precision}\n"
    results_text += f"Accuracy Score: {avg_accuracy}\n"

    # Write the results to a file in the prediction directory
    results_file_path = os.path.join(pred_directory, 'results.txt')
    with open(results_file_path, 'w') as results_file:
        results_file.write(results_text)

    print(f"Results written to {results_file_path}")



