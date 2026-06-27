"""
Upload preprocessor.joblib ke Google Drive
Otomatis overwrite jika file sudah ada di folder tujuan.

Cara pakai:
    export GDRIVE_CREDENTIALS='<isi JSON service account>'
    export GDRIVE_FOLDER_ID='<ID folder Google Drive>'
    python Pre-processing/upload_to_gdrive.py
"""

import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ── 1. Load credentials dari environment variable
creds_json = json.loads(os.environ["GDRIVE_CREDENTIALS"])
credentials = Credentials.from_service_account_info(
    creds_json,
    scopes=["https://www.googleapis.com/auth/drive"]
)

# ── 2. Build Google Drive API client
service = build('drive', 'v3', credentials=credentials)

# ── 3. Konfigurasi file & folder tujuan
FOLDER_ID    = os.environ["GDRIVE_FOLDER_ID"]
LOCAL_FILE   = "Pre-processing/preprocessor.joblib"
FILE_NAME    = "preprocessor.joblib"


def cari_file_existing(folder_id, file_name):
    """Cari file dengan nama tertentu di dalam folder Google Drive."""
    query = (
        f"name = '{file_name}' "
        f"and '{folder_id}' in parents "
        f"and trashed = false"
    )
    response = service.files().list(
        q=query,
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    files = response.get("files", [])
    return files[0] if files else None


def upload_file(local_path, file_name, folder_id):
    """Upload file ke Google Drive. Overwrite jika sudah ada."""
    if not os.path.exists(local_path):
        print(f"❌ File tidak ditemukan: {local_path}")
        raise FileNotFoundError(f"{local_path} tidak ditemukan")

    file_size = os.path.getsize(local_path) / (1024 * 1024)
    print(f"📦 File: {file_name} ({file_size:.1f} MB)")

    media = MediaFileUpload(local_path, resumable=True)
    existing = cari_file_existing(folder_id, file_name)

    if existing:
        # ── File sudah ada → UPDATE (overwrite)
        print(f"🔄 File sudah ada (ID: {existing['id']}) → Overwrite...")
        updated = service.files().update(
            fileId=existing["id"],
            media_body=media,
            fields="id, name",
            supportsAllDrives=True
        ).execute()
        print(f"✅ Berhasil diupdate: {updated['name']} (ID: {updated['id']})")
    else:
        # ── File belum ada → CREATE baru
        print("🆕 File belum ada → Upload baru...")
        file_meta = {
            "name": file_name,
            "parents": [folder_id]
        }
        created = service.files().create(
            body=file_meta,
            media_body=media,
            fields="id, name",
            supportsAllDrives=True
        ).execute()
        print(f"✅ Berhasil diupload: {created['name']} (ID: {created['id']})")


if __name__ == "__main__":
    print("☁️  Memulai upload ke Google Drive...")
    print(f"📂 Folder tujuan ID: {FOLDER_ID}")
    upload_file(LOCAL_FILE, FILE_NAME, FOLDER_ID)
    print("🎉 Upload selesai!")
