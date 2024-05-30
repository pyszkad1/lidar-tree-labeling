from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


class TrunkDataset(Dataset):
    def __init__(self, data_dir=r"C:\Users\Adam\Desktop\school\Bc_projekt\labeling\data\labeled", exclude_oldest=0):
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
        return len(self.data_files) // 5

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
