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
        input_path = os.path.join(self.data_dir, self.data_files[idx])
        input_data = np.load(input_path)

        mask_path = os.path.join(self.data_dir, self.mask_files[idx])
        target_mask = np.load(mask_path)

        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_mask).float().unsqueeze(0)

        return input_tensor, target_tensor


def save_plot(csv_file, loss_plot_file, accuracy_plot_file):
    data = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))
    plt.plot(data['Epoch'], data['Training Loss'], label='Training Loss')
    plt.plot(data['Epoch'], data['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(loss_plot_file)
    plt.close()

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

    files = os.listdir(directory)

    files.sort()

    for i, filename in enumerate(files):
        number = str(starting_number + i).zfill(2)
        new_filename = f"{number}_{filename}"
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_filename}'")


def distribute_files(source_dir, target_dirs):
    file_groups = defaultdict(list)

    for filename in os.listdir(source_dir):
        prefix = filename.split('.')[0]
        file_groups[prefix].append(filename)

    for dir_path in target_dirs:
        os.makedirs(dir_path, exist_ok=True)

    grouped_files = list(file_groups.values())

    for index, group in enumerate(grouped_files):
        target_dir = target_dirs[index % len(target_dirs)]
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

    avg_jaccard = np.mean(list(jaccard_scores.values())) if jaccard_scores else 0
    avg_precision = np.mean(list(precision_scores.values())) if precision_scores else 0
    avg_accuracy = np.mean(list(accuracy_scores.values())) if accuracy_scores else 0

    return jaccard_scores, precision_scores, accuracy_scores, avg_jaccard, avg_precision, avg_accuracy


def run_compute_metrics_and_save_them():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    pred_directory = os.path.join(project_dir, 'data', 'with_nn')
    true_directory = os.path.join(project_dir, 'data', 'true_labels')

    jaccard_scores, precision_scores, accuracy_scores, avg_jaccard, avg_precision, avg_accuracy = compute_metrics_for_directory(
        pred_directory, true_directory)

    std_jaccard = np.std(list(jaccard_scores.values()))
    std_precision = np.std(list(precision_scores.values()))
    std_accuracy = np.std(list(accuracy_scores.values()))

    best_jaccard_file = max(jaccard_scores, key=jaccard_scores.get)
    best_jaccard_score = jaccard_scores[best_jaccard_file]

    results_text = (
        f"Jaccard Score: {avg_jaccard}, Standard Deviation: {std_jaccard}\n"
        f"Best Jaccard Score: {best_jaccard_score} in file {best_jaccard_file}\n"
        f"Precision Score: {avg_precision}, Standard Deviation: {std_precision}\n"
        f"Accuracy Score: {avg_accuracy}, Standard Deviation: {std_accuracy}\n"
    )

    results_file_path = os.path.join(pred_directory, 'results.txt')
    with open(results_file_path, 'w') as results_file:
        results_file.write(results_text)

    print(f"Results written to {results_file_path}")

def convert_time_to_minutes(time_list):
    minutes = []
    for time_str in time_list:
        min_part, sec_part = time_str.split('m')
        minutes.append(int(min_part) + int(sec_part.rstrip('s')) / 60)
    return minutes


def create_graph_of_time():
    images = list(range(1, 11))
    time_with_nn = ['1m 24s', '3m 30s', '2m 00s', '2m 51s', '3m 08s', '1m 42s', '1m 05s', '2m 00s', '2m 47s', '0m 53s']
    time_without_nn = ['6m 58s', '5m 04s', '3m 38s', '5m 13s', '5m 07s', '3m 06s', '3m 32s', '4m 34s', '5m 20s',
                       '2m 53s']

    time_with_nn_min = convert_time_to_minutes(time_with_nn)
    time_without_nn_min = convert_time_to_minutes(time_without_nn)

    # Plotting times
    plt.figure(figsize=(10, 6))
    plt.plot(images, time_with_nn_min, marker='o', color='tab:orange', label='With NN Assistance')
    plt.plot(images, time_without_nn_min, marker='x', color='tab:blue', label='Without NN Assistance')
    plt.xlabel('Image Number')
    plt.ylabel('Time Spent (minutes)')
    plt.title('Time Spent Labeling With vs. Without NN Assistance')
    plt.legend()
    plt.grid(True)
    plt.xticks(images)
    plt.savefig('Time_Comparison.png')
    plt.show()


    # Plotting speedup
    speedup = [without / with_nn for without, with_nn in zip(time_without_nn_min, time_with_nn_min)]

    average_speedup = np.mean(speedup)

    plt.figure(figsize=(10, 6))
    plt.bar(images, speedup, color='tab:blue')
    plt.axhline(y=average_speedup, color='r', linestyle='-', label=f'Average Speedup: {average_speedup:.2f}x')
    plt.xlabel('Image Number')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup from NN Assistance')
    plt.xticks(images)
    plt.legend()
    plt.savefig('Speedup_Factor.png')
    plt.show()


if __name__ == '__main__':
    create_graph_of_time()






