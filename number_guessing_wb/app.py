from flask import Flask, render_template, request, jsonify
from io import BytesIO
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import base64

app = Flask(__name__)

# ----------------------------
# Load Model
# ----------------------------
class MNISTmodel(nn.Module):
    def __init__(self, input_shape:int, output_shape:int, hidden_units:int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_shape)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MNISTmodel(1, 10, 128)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    image_data = data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(image)
        predicted_class = preds.argmax(dim=1).item()

    return jsonify({"prediction": int(predicted_class)})

if __name__ == "__main__":
    app.run(debug=True)
