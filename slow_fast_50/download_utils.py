# download_utils.py
import os
import gdown
import zipfile

def download_file(file_id, output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

def extract_zip(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)