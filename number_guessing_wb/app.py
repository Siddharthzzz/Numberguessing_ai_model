import base64
from io import BytesIO
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# --- PYTORCH MODEL CLASS DEFINITION ---
# This MUST be the exact same architecture as the model you trained
class MNISTmodel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app)  # This enables cross-origin requests

# --- MODEL AND TRANSFORM SETUP ---
MODEL = None
MODEL_FILE_NAME = 'model.pth' # Use the corrected model filename
TRANSFORM = None

def load_model():
    """Load the PyTorch model and define transformations."""
    global MODEL, TRANSFORM
    try:
        device = torch.device('cpu')
        MODEL = MNISTmodel(input_shape=1, output_shape=10, hidden_units=128)
        # CRUCIAL: Use weights_only=False to load the model correctly
        MODEL.load_state_dict(torch.load(MODEL_FILE_NAME, map_location=device, weights_only=False))
        MODEL.eval()
        
        TRANSFORM = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        print(f"✨ PyTorch Model '{MODEL_FILE_NAME}' loaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️ ERROR: An unexpected error occurred while loading the model: {e}")
        return False

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives image data, processes it, and returns a prediction."""
    if not MODEL or not TRANSFORM:
        return jsonify({'error': 'Model is not loaded.'}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data found in request.'}), 400

    try:
        # Decode the Base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        # Preprocess the image for the model
        with torch.no_grad():
            transformed_image = TRANSFORM(image).unsqueeze(0)
            
            # Make a prediction
            output = MODEL(transformed_image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            response = {
                'prediction': predicted_class.item(),
                'confidence': confidence.item()
            }
            return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Failed to process the image.'}), 500

if __name__ == '__main__':
    if load_model():
        app.run(host='0.0.0.0', port=5000)

