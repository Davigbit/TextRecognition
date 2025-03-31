import socket
from tkinter import filedialog, Tk

# Set up the client
HOST = '127.0.0.1'
PORT = 5000

# Function to open the file dialog and choose an image
def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("PNG Files", "*.png"), ("JPG Files", "*.jpg"), ("All Files", "*.*"))
    )
    return file_path

# Create the main window for the GUI
root = Tk()
root.withdraw()  # Hide the root window

# Ask the user to select an image
image_path = select_image()

if image_path:
    # Connect to the server
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))

        # Send the selected image path to the server
        client_socket.sendall(image_path.encode())
    

        # Receive the predictions from the server
        predictions_str = client_socket.recv(1024).decode()

        # Split the received string into individual predictions and print them
        predictions = predictions_str.split(",")
        for prediction in predictions:
            print(prediction, end="")

        print()

    except:
        print("Connection error, verify server connection")

    client_socket.close()
else:
    print("No image selected.")
