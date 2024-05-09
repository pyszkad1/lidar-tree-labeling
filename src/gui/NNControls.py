from src.nn.UNet import *
from src.nn.data_utils import *



class NNControls:
    def __init__(self):
        self.is_learning = False
        self.current_predictions = None
        self.model = None
        self.last_amount_of_pictures = 0

    def learn_UNet(self):
        print('Learning UNet')
        if os.path.exists('model_state_dict.pth'):
            print("Model found. Loading model...")
            model = UNet(1, 1)
            model.load_state_dict(torch.load('model_state_dict.pth'))

            script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
            project_dir = os.path.dirname(os.path.dirname(script_dir))  # Move up two levels to get to the project root
            target_directory = os.path.join(project_dir, 'data', 'true_labels')

            new_dataset = TrunkDataset(target_directory, exclude_oldest=self.last_amount_of_pictures)
            self.last_amount_of_pictures = len(new_dataset) + self.last_amount_of_pictures

            print("Retraining model...")
            retrain_UNet(model, new_dataset, 10)
            self.model = model
            print("Model retrained")

        else:
            print("Model file not found. Training new model...")
            train_dataset = TrunkDataset()
            self.last_amount_of_pictures = len(train_dataset)
            model = UNet(1, 1)
            train_UNet(model, train_dataset)
            self.model = model
            print("Model trained")


    def test_UNet(self, current_picture):
        if os.path.exists('model_state_dict.pth'):
            print("Model found. Loading model...")
            model = UNet(1, 1)
            model.load_state_dict(torch.load('model_state_dict.pth'))
        else:
            print("Model file not found. A new model needs to be trained or the correct path provided.")
            return

        model.eval()

        with torch.no_grad():
            input = torch.tensor(current_picture, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            outputs = model(input)
            predicted_mask = torch.sigmoid(outputs) > 0.5
            numpy_array = predicted_mask.squeeze().numpy()

            self.current_predictions = numpy_array
            print("Predictions made")


if __name__ == '__main__':
    nn_controls = NNControls()
    nn_controls.learn_UNet()

    print("Done")


