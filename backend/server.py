from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Dict, List
from .main import run_pipeline
from .config import Config
from .data_loader import DataLoader
from .cnn import CustomCNN

app = FastAPI(title="DeepVision ML Suite API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for training
training_status = {
    "is_training": False,
    "last_run": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "logs": []
}

class ModelManager:
    _models = {}

    @classmethod
    def get_model(cls, model_name: str):
        if model_name not in cls._models:
            # Load model from disk if it exists, otherwise build it
            model_path = os.path.join(Config.MODEL_SAVE_PATH, f"best_{model_name}.h5")
            if os.path.exists(model_path):
                print(f"Loading model {model_name} from {model_path}")
                cls._models[model_name] = tf.keras.models.load_model(model_path)
            else:
                print(f"Model {model_name} not found, building new instance...")
                if model_name == "CustomCNN":
                    cls._models[model_name] = CustomCNN().build_model()
                # Add other models as needed
        return cls._models[model_name]

@app.get("/")
async def root():
    return {"message": "Welcome to DeepVision ML Suite API", "status": "online"}

@app.post("/train")
async def start_training(background_tasks: BackgroundTasks, smoke_test: bool = True):
    if training_status["is_training"]:
        return {"message": "Training is already in progress"}
    
    training_status["is_training"] = True
    background_tasks.add_task(run_training_job, smoke_test)
    return {"message": "Training started in background"}

@app.get("/status")
async def get_status():
    return training_status

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    # Load and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L') # Fashion MNIST is grayscale
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1)) # Add batch and channel dims

    # Get model and predict
    model = ModelManager.get_model(model_name)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    dl = DataLoader()
    class_name = dl.class_names[predicted_class]
    confidence = float(predictions[0][predicted_class])

    return {
        "model": model_name,
        "class_id": int(predicted_class),
        "class_name": class_name,
        "confidence": confidence
    }

def run_training_job(smoke_test: bool):
    try:
        run_pipeline(smoke_test=smoke_test)
        training_status["last_run"] = "Success"
    except Exception as e:
        print(f"Training error: {e}")
        training_status["last_run"] = f"Failed: {str(e)}"
    finally:
        training_status["is_training"] = False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
