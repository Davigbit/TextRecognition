import socket
import textfunc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

dir_path = "test/letters_output"
letters_size = (28, 28)

# The transformer from the training process
class InvertIfMajority:
    def __call__(self, image):
        image = image.convert("L")
        tensor_image = transforms.ToTensor()(image)
        if tensor_image.mean() > 0.5:
            tensor_image = 1 - tensor_image
        return tensor_image

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    InvertIfMajority()
])

# Define the CNN class (same as the one used for training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()

# Load the model weights
model.load_state_dict(torch.load("src/model/model_weights.pth"))
model.eval()

# Set up the server
HOST = '127.0.0.1'
PORT = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Server is listening")
print("Do CTRL+C to stop the server")

try:
    while True:
        textfunc.clean_dir(dir_path)

        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")

        # Receive the image path from the client
        image_path = conn.recv(1024).decode()

        predictions = []

        # Process the image and make a prediction
        try:
            textfunc.generate_letters(dir_path, image_path, letters_size)

            dir = Path(dir_path)
            for letter in dir.iterdir():
                image = Image.open(letter)
                input_tensor = transform(image)
                input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, 1, 28, 28]
                output = model(input_tensor)
                predicted_class = str(torch.argmax(output, dim=1).item())
                predictions.append(predicted_class)

            # Send the predicted class back to the client
            predictions_str = ",".join(predictions)
            conn.sendall(predictions_str.encode())

        except Exception as e:
            conn.sendall(f"Error: {e}".encode())

        conn.close()

except KeyboardInterrupt:
    print("\nServer is shutting down")

finally:
    server_socket.close()