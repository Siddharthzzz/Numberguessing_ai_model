# 8. CRUCIAL STEP: Save the trained model's state dictionary
import torch
from MINIST import Numberguessing_model
MODEL_SAVE_PATH = "model.pth"
print(f"Training complete. Saving model to '{MODEL_SAVE_PATH}'...")
torch.save(obj=Numberguessing_model.state_dict(), f=MODEL_SAVE_PATH)
print("Model saved successfully!")