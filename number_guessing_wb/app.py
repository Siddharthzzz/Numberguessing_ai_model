import base64
from io import BytesIO
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app)

# --- PYTORCH MODEL CLASS DEFINITION ---
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

# --- MODEL AND TRANSFORM SETUP ---
MODEL_FILE_NAME = 'model.pth'
MODEL_LOADED = False
device = torch.device('cpu')

try:
    MODEL = MNISTmodel(input_shape=1, output_shape=10, hidden_units=128)
    MODEL.load_state_dict(torch.load(MODEL_FILE_NAME, map_location=device))
    MODEL.eval()
    print(f"✨ PyTorch Model '{MODEL_FILE_NAME}' loaded successfully!")
    MODEL_LOADED = True
except FileNotFoundError:
    print(f"⚠️ ERROR: Model file '{MODEL_FILE_NAME}' not found.")
except Exception as e:
    print(f"⚠️ ERROR: An unexpected error occurred while loading the model: {e}")

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    print("\n--- Received a new prediction request ---")
    if not MODEL_LOADED:
        print("❌ ERROR: Model is not loaded.")
        return jsonify({'error': 'Model is not loaded, check server logs.'}), 500

    print("1. Getting image data from request...")
    data = request.get_json()
    if not data or 'image' not in data:
        print("❌ ERROR: No image data in request.")
        return jsonify({'error': 'No image data found in request.'}), 400

    print("2. Decoding Base64 image...")
    try:
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))
        # Convert image to grayscale to handle potential alpha channel from canvas
        image = image.convert('L')
        print("   ✅ Image decoded and converted to grayscale successfully.")
    except Exception as e:
        print(f"❌ ERROR: Invalid image data: {e}")
        return jsonify({'error': f'Invalid image data: {e}'}), 400

    print("3. Preprocessing image for PyTorch...")
    with torch.no_grad():
        try:
            transformed_image = TRANSFORM(image).unsqueeze(0)
            print(f"   ✅ Image transformed. Tensor shape: {transformed_image.shape}")

            print("4. Making prediction...")
            output = MODEL(transformed_image)
            print("   ✅ Prediction output received from model.")

            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            print(f"   ✅ Prediction: {predicted_class.item()}, Confidence: {confidence.item():.4f}")

        except Exception as e:
            print(f"❌ ERROR during model prediction: {e}")
            return jsonify({'error': 'Failed during model inference.'}), 500

    print("5. Sending JSON response back to browser.")
    return jsonify({
        'prediction': predicted_class.item(),
        'confidence': confidence.item()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

