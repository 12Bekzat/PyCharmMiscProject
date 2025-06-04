import os
import subprocess
import importlib.util

def install_models_if_missing():
    try:
        import face_recognition_models
    except ImportError:
        print("face_recognition_models not found. Installing...")
        subprocess.check_call([
            "pip", "install",
            "git+https://github.com/ageitgey/face_recognition_models"
        ])
        print("Installation complete. Reload the server to continue.")
        exit()

install_models_if_missing()