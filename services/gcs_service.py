from google.cloud import storage
from google.oauth2 import service_account
from typing import Dict
import os

class GCSService:
    def __init__(self, service_account_path: str):
        self.credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.storage_client = storage.Client(credentials=self.credentials)

    def download_files(self, bucket_name: str, file_mappings: Dict[str, str]):
        """
        Download multiple files from GCS using service account credentials
        
        Args:
            bucket_name: Name of the GCS bucket
            file_mappings: Dictionary mapping GCS blob names to local file paths
        """
        bucket = self.storage_client.bucket(bucket_name)
        
        for blob_name, local_path in file_mappings.items():
            try:
                print(f"Downloading {blob_name} to {local_path}...")
                blob = bucket.blob(blob_name)
                blob.download_to_filename(local_path)
                print(f"Successfully downloaded {blob_name}")
            except Exception as e:
                print(f"Error downloading {blob_name}: {str(e)}")
                raise