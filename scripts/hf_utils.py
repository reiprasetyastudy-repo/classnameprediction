#!/usr/bin/env python
"""
HuggingFace Hub Utility Functions
"""
import os
import time
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, create_repo, hf_hub_download
from tqdm import tqdm


def get_hf_token():
    """Get HuggingFace token from environment or file"""
    # Try environment variable first
    token = os.getenv('HF_TOKEN')
    if token:
        return token

    # Try .env file
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith('HF_TOKEN='):
                    return line.split('=', 1)[1].strip()

    return None


def create_repo_if_not_exists(repo_id: str, private: bool = False, token: Optional[str] = None):
    """Create HuggingFace repo if it doesn't exist"""
    try:
        create_repo(repo_id, exist_ok=True, private=private, token=token)
        print(f"✅ Repo ready: {repo_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        return False


def upload_file_safe(
    file_path: str,
    repo_id: str,
    path_in_repo: str,
    token: Optional[str] = None,
    max_retries: int = 3
):
    """Upload file with retry logic"""
    api = HfApi(token=token)

    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  ✅ Uploaded: {path_in_repo}")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  ⚠️ Upload failed, retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Upload failed after {max_retries} attempts: {e}")
                return False


def upload_folder_safe(
    folder_path: str,
    repo_id: str,
    path_in_repo: Optional[str] = None,
    token: Optional[str] = None,
    max_retries: int = 3
):
    """Upload folder with retry logic"""
    api = HfApi(token=token)

    for attempt in range(max_retries):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=path_in_repo,
            )
            print(f"  ✅ Uploaded folder: {folder_path}")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⚠️ Upload failed, retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Upload failed after {max_retries} attempts: {e}")
                return False


def list_repo_files(repo_id: str, token: Optional[str] = None):
    """List all files in a HuggingFace repo"""
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
        return files
    except Exception as e:
        print(f"❌ Failed to list files: {e}")
        return []


def download_file_from_hub(
    repo_id: str,
    filename: str,
    local_dir: Optional[str] = None,
    token: Optional[str] = None
):
    """Download a single file from HuggingFace Hub"""
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            token=token,
            repo_type="model"
        )
        print(f"✅ Downloaded: {filename}")
        return file_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def get_repo_url(repo_id: str):
    """Get HuggingFace repo URL"""
    return f"https://huggingface.co/{repo_id}"
