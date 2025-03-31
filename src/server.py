import socket
import textfunc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import string

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
        self.fc4 = nn.Linear(128, 62)
        self.relu = nn.ReLU()

    def forward(self, x, drop_prob=0.5):
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

        # Process the image and make a prediction
        try:
            textfunc.generate_letters(dir_path, image_path)

            # Define the character order (must match training)
            CHARACTERS = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

            predictions = []
            dir = Path(dir_path)
            for letter in dir.iterdir():
                image = Image.open(letter)
                input_tensor = transform(image)
                input_tensor = input_tensor.unsqueeze(0)
                output = model(input_tensor)
                
                predicted_idx = torch.argmax(output, dim=1).item()
                predicted_char = CHARACTERS[predicted_idx]  # Direct mapping
                predictions.append(predicted_char)

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
