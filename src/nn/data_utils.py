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



if __name__ == '__main__':
    # Rename files starting from 100 in the 'data' directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    target_directory = os.path.join(project_dir, 'data', 'true_labels')

    source_dir = target_directory
    target_dirs = [os.path.join(project_dir, 'data', 'train1'),
                   os.path.join(project_dir, 'data', 'train2'),
                   os.path.join(project_dir, 'data', 'train3'),
                   os.path.join(project_dir, 'data', 'train4'),
                   os.path.join(project_dir, 'data', 'test')]

    distribute_files(source_dir, target_dirs)