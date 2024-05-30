from src.nn.UNet import *
from src.nn.data_utils import *



class NNControls:
    def __init__(self):
        self.current_predictions = None

    def learn_UNet(self):
        print('Learning UNet')
        model = train_UNet(8)

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
            print(numpy_array.shape)

            self.current_predictions = numpy_array
            print("Predictions made")





