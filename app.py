from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os
import uvicorn
from pydantic import BaseModel
from typing import Dict
import tensorflow as tf
from google.cloud import storage
from google.oauth2 import service_account

bucket_name = os.environ['BUCKET_NAME']
model_blob_name = os.environ['MODEL_BLOB_NAME']
local_model_path = os.environ['LOCAL_MODEL_PATH']
service_account_path = os.environ['SERVICE_ACCOUNT_PATH']

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using a TensorFlow CNN model",
    version="1.0.0"
)

def download_model_from_gcs(
    bucket_name: str,
    model_blob_name: str,
    local_model_path: str,
    service_account_path: str
):
    """Download model from GCS using service account credentials"""
    
    # Initialize credentials
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # Create storage client with service account
    storage_client = storage.Client(credentials=credentials)
    
    # Get bucket and download model
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_blob_name)
    blob.download_to_filename(local_model_path)
    
    return local_model_path


# Global variables
MODEL_PATH = "model_transfer_downsyndrome.keras"
LABELS = {0: "Syndrome", 1: "Healthy"}



if not os.path.exists("model_transfer_downsyndrome.keras"):
    print("Downloading model from GCS...")
    download_model_from_gcs(
    bucket_name,
    model_blob_name,
    local_model_path,
    service_account_path
)

# Load model at startup
print(f"Loading model from {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    input_shape = model.input_shape[1:3]
    print(f"Model loaded successfully. Input shape: {input_shape}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class PredictionResponse(BaseModel):
    filename: str
    predicted_label: str
    probability: float
    error: str = None

@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {"message": "Image Classification API is running", 
            "model_input_shape": input_shape}

def process_image(image_data: bytes) -> np.ndarray:
    """Process uploaded image data into model input format"""
    # Read image from bytes
    img = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize to expected input shape
    img = img.resize(input_shape)
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict class for uploaded image"""
    try:
        # Read and validate file
        if not file.content_type.startswith('image/'):
            return PredictionResponse(
                filename=file.filename,
                predicted_label="",
                probability=0.0,
                error="Uploaded file must be an image"
            )
        
        # Read file contents
        contents = await file.read()
        
        # Process image
        img_array = process_image(contents)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Process prediction based on model output shape
        if prediction.shape[-1] == 1:  # Binary classification
            predicted_class = int(prediction[0][0] > 0.5)
            probability = float(prediction[0][0])
        else:  # Multi-class classification
            predicted_class = np.argmax(prediction[0])
            probability = float(prediction[0][predicted_class])
        
        # Get label
        label = LABELS[predicted_class]
        
        return PredictionResponse(
            filename=file.filename,
            predicted_label=label,
            probability=probability
        )
        
    except Exception as e:
        return PredictionResponse(
            filename=file.filename,
            predicted_label="",
            probability=0.0,
            error=str(e)
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)