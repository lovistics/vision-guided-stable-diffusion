import os
import urllib.request
import bz2
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_extract_model():
    """Download and extract the facial landmark model."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        model_path = "models/shape_predictor_68_face_landmarks.dat"
        if os.path.exists(model_path):
            logger.info("Model file already exists.")
            return
        
        # Download the compressed model file
        logger.info("Downloading facial landmark model...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = model_path + ".bz2"
        
        urllib.request.urlretrieve(url, compressed_path)
        
        # Extract the model file
        logger.info("Extracting model file...")
        with bz2.BZ2File(compressed_path) as fr, open(model_path, 'wb') as fw:
            shutil.copyfileobj(fr, fw)
        
        # Clean up the compressed file
        os.remove(compressed_path)
        logger.info("Model setup completed successfully!")

    except Exception as e:
        logger.error(f"Error setting up model: {str(e)}")
        raise

if __name__ == "__main__":
    download_and_extract_model()