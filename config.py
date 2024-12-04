import os

# Environment variables
BUCKET_NAME = os.environ['BUCKET_NAME']
MODEL_BLOB_NAME = os.environ['MODEL_BLOB_NAME']
LOCAL_MODEL_PATH = os.environ['LOCAL_MODEL_PATH']
SERVICE_ACCOUNT_PATH = os.environ['SERVICE_ACCOUNT_PATH']
DATASET_BLOB_NAME = os.environ['DATASET_PATH']
SISREK_MODEL_BLOB_NAME = os.environ['SISREK_MODEL_PATH']

# Local file paths
LOCAL_DATASET_PATH = DATASET_BLOB_NAME
LOCAL_SISREK_MODEL_PATH = SISREK_MODEL_BLOB_NAME

# Model configuration
MODEL_PATH = "model_transfer_downsyndrome.keras"
LABELS = {0: "Syndrome", 1: "Healthy"}

# Feature weights
MINAT_WEIGHT = 0.6
SKILLS_WEIGHT = 0.3
CONDITIONS_WEIGHT = 0.1

# API Settings
DEFAULT_PORT = 8000