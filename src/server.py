from flask import Flask, request, jsonify
from flask_cors import CORS
import textfunc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import string
import numpy as np
import cv2

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": ["http://localhost:5173"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

dir_path = "temp/letters_output"
letters_size = (128, 128)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 36)
        self.relu = nn.ReLU()

    def forward(self, x, drop_prob=0.3):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten for FC layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=drop_prob, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=drop_prob, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=drop_prob, training=self.training)
        x = self.fc4(x)
        return x

model = CNN()

# Load the model weights
model.load_state_dict(torch.load("src/model/model_weights.pth", map_location=torch.device('cpu')))
model.eval()

# Set up the server
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the image file from the client
        image_file = request.files['image']
        
        # Read the image directly with OpenCV
        nparr = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            return jsonify({"error": "Invalid image received"}), 400

        textfunc.clean_dir(dir_path)
        textfunc.generate_letters(dir_path, image)

        # Character mapping
        CHARACTERS = list(string.digits + string.ascii_uppercase)

        predictions = []
        dir = Path(dir_path)
        for letter in dir.iterdir():
            image = Image.open(letter)
            input_tensor = transform(image)
            input_tensor = input_tensor.unsqueeze(0)
            output = model(input_tensor)
            
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_char = CHARACTERS[predicted_idx]
            predictions.append(predicted_char)

        textfunc.clean_dir(dir_path)

        # Create the response
        response = {
            "prediction": "".join(predictions)
        }

        # Send the JSON response back to the client
        return jsonify(response)

    except Exception as e:
        # Error handling
        error_response = {
            "error": str(e)
        }
        return jsonify(error_response), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
