import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.nn.UNetParts import *
import os
import matplotlib.pyplot as plt

from src.nn.data_utils import *

import csv

_SIZE = (128, 1024)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



def train_UNet(num_epochs=5):
    model = UNet(n_channels=1, n_classes=1)
    train_dataset = TrunkDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True) #TODO change batch size
    validation_dataset = TrunkDataset()
    validation_dataloader = DataLoader(validation_dataset, batch_size=1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    # Open a file to write the losses and accuracy
    with open('training_validation_losses_accuracy.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'])

        for epoch in range(num_epochs):
            # Training phase with accuracy calculation
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            i = 0
            for inputs, targets in train_dataloader:

                print(f"in learning for cycle: iteration {i+1} / {len(train_dataloader)}")
                i += 1
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Calculate training accuracy
                predicted = outputs > 0.5  # Assuming binary classification
                correct_train += (predicted == targets).sum().item()
                total_train += targets.numel()

            train_accuracy = 100 * correct_train / total_train

            # # Validation phase with accuracy calculation
            # model.eval()
            # validation_loss = 0.0
            # correct_val = 0
            # total_val = 0
            # with torch.no_grad():
            #     for inputs, targets in validation_dataloader:
            #         outputs = model(inputs)
            #         loss = criterion(outputs, targets)
            #         validation_loss += loss.item()
            #
            #         # Calculate validation accuracy
            #         predicted = outputs > 0.5
            #         correct_val += (predicted == targets).sum().item()
            #         total_val += targets.numel()
            #
            # validation_accuracy = 100 * correct_val / total_val
            #
            # # Write the current epoch's losses and accuracy to the file
            # writer.writerow(
            #     [epoch + 1, train_loss / len(train_dataloader), validation_loss / len(validation_dataloader),
            #      train_accuracy, validation_accuracy])

            validation_loss = 0
            validation_accuracy = 0

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss / len(train_dataloader)}, Validation Loss: {validation_loss / len(validation_dataloader)}, Training Accuracy: {train_accuracy}, Validation Accuracy: {validation_accuracy}")

    print("Training complete")
    torch.save(model.state_dict(), 'model_state_dict.pth')

    return model


def test_model(model, test_dataloader, plot_dir):
    model.eval()
    total_accuracy = 0.0
    total_images = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dataloader):
            outputs = model(inputs)
            predicted_mask = torch.sigmoid(outputs) > 0.5  # Convert to binary mask

            # Calculate accuracy
            correct_predictions = (predicted_mask == targets).sum().item()
            total_pixels = targets.numel()
            accuracy = correct_predictions / total_pixels
            total_accuracy += accuracy
            total_images += 1

            # Visualize the input, true mask, predicted mask, and accuracy
            input_image = inputs[0].squeeze().numpy()
            true_mask = targets[0].squeeze().numpy()
            pred_mask = predicted_mask[0].squeeze().numpy()

            plt.figure(figsize=(15, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(input_image, cmap='gray')
            plt.title('Input Image')
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(true_mask, cmap='gray')
            plt.title('True Mask')
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.text(0.5, 0.5, f'Accuracy: {accuracy:.2%}', horizontalalignment='center', verticalalignment='center')
            plt.axis('off')

            # Save the plot
            plt.savefig(os.path.join(plot_dir, f'test_plot_{i}.png'))
            plt.close()

            print(f"Processed and saved test case {i}, Accuracy: {accuracy:.2%}")

    overall_accuracy = total_accuracy / total_images
    print(f"Overall Accuracy: {overall_accuracy:.2%}")


