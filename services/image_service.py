from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
import io

class ImageService:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def process_image(self, image_data: bytes) -> np.ndarray:
        """Process uploaded image data into model input format"""
        # Read image from bytes
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize to expected input shape
        img = img.resize(self.input_shape)
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array