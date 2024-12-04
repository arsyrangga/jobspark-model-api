from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
import numpy as np
import os
import uvicorn

import config
from models.schemas import PredictionResponse, RecommendationRequest, RecommendationResponse
from services.gcs_service import GCSService
from services.image_service import ImageService
from services.recommendation_service import RecommendationService
from utils.data_processing import parse_list_input

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using a TensorFlow CNN model",
    version="1.0.0"
)

# Initialize services
gcs_service = None
image_service = None
recommendation_service = None
model = None
sisrek_model = None

@app.on_event("startup")
async def startup_event():
    global gcs_service, image_service, recommendation_service, model, sisrek_model
    
    # Initialize GCS service
    gcs_service = GCSService(config.SERVICE_ACCOUNT_PATH)
    
    # Download required files
    file_mappings = {
        config.MODEL_BLOB_NAME: config.LOCAL_MODEL_PATH,
        config.DATASET_BLOB_NAME: config.LOCAL_DATASET_PATH,
        config.SISREK_MODEL_BLOB_NAME: config.LOCAL_SISREK_MODEL_PATH
    }
    
    if not all(os.path.exists(path) for path in file_mappings.values()):
        print("Downloading required files from GCS...")
        gcs_service.download_files(config.BUCKET_NAME, file_mappings)
    
    # Load models
    try:
        model = load_model(config.MODEL_PATH)
        input_shape = model.input_shape[1:3]
        image_service = ImageService(input_shape)
        
        sisrek_model = load_model(config.LOCAL_SISREK_MODEL_PATH)
        recommendation_service = RecommendationService(sisrek_model, config.LOCAL_DATASET_PATH)
        
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {
        "message": "Image Classification API is running",
        "model_input_shape": model.input_shape[1:3] if model else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict class for uploaded image"""
    try:
        if not file.content_type.startswith('image/'):
            return PredictionResponse(
                filename=file.filename,
                predicted_label="",
                probability=0.0,
                error="Uploaded file must be an image"
            )
        
        contents = await file.read()
        img_array = image_service.process_image(contents)
        prediction = model.predict(img_array)
        
        if prediction.shape[-1] == 1:
            predicted_class = int(prediction[0][0] > 0.5)
            probability = float(prediction[0][0])
        else:
            predicted_class = np.argmax(prediction[0])
            probability = float(prediction[0][predicted_class])
        
        return PredictionResponse(
            filename=file.filename,
            predicted_label=config.LABELS[predicted_class],
            probability=probability
        )
        
    except Exception as e:
        return PredictionResponse(
            filename=file.filename,
            predicted_label="",
            probability=0.0,
            error=str(e)
        )

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    try:
        minat_list = parse_list_input(request.minat)
        kemampuan_list = parse_list_input(request.kemampuan)
        kondisi_list = parse_list_input(request.kondisi)

        recommendations = recommendation_service.get_recommendations(
            minat_list,
            kemampuan_list,
            kondisi_list
        )

        return RecommendationResponse(
            success=True,
            message="Recommendations generated successfully",
            data=recommendations
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", config.DEFAULT_PORT))
    uvicorn.run(app, host="0.0.0.0", port=port)