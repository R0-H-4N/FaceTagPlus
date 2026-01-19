import os
import tarfile
import requests
import gdown

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = "datasets"
VAL_DIR = os.path.join(BASE_DIR, "val_256")
CUSTOM_DIR = os.path.join(BASE_DIR, "custom_dataset")

VAL_TAR_URL = "https://data.csail.mit.edu/places/places365/val_256.tar"
VAL_TXT_URL = "https://data.csail.mit.edu/places/places365/places365_val.txt"

GOOGLE_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "124IIMN2q_gvPENKZVHf7jNS3EpCDzvP0"
)

# -----------------------------
# Utilities
# -----------------------------
def mkdir_if_not_exists(path):
    os.makedirs(path, exist_ok=True)

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"[SKIP] {dest} already exists")
        return
    print(f"[DOWNLOAD] {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def extract_tar(tar_path, extract_to):
    print(f"[EXTRACT] {tar_path}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)

# -----------------------------
# Main setup
# -----------------------------
def setup_datasets():
    mkdir_if_not_exists(BASE_DIR)

    # ---- Places365 val_256 ----
    mkdir_if_not_exists(VAL_DIR)

    tar_path = os.path.join(BASE_DIR, "val_256.tar")

    download_file(VAL_TAR_URL, tar_path)
    extract_tar(tar_path, BASE_DIR)

    txt_path = os.path.join(VAL_DIR, "places365_val.txt")
    download_file(VAL_TXT_URL, txt_path)

    # ---- Custom Dataset (Google Drive) ----
    mkdir_if_not_exists(CUSTOM_DIR)

    print("[DOWNLOAD] Google Drive custom_dataset folder")
    gdown.download_folder(
        GOOGLE_DRIVE_FOLDER_URL,
        output=BASE_DIR,
        quiet=False,
        use_cookies=False
    )

    print("\n✅ Dataset setup complete!")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    setup_datasets()
