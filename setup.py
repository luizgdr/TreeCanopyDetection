from pathlib import Path
import sys
import os
import subprocess

VENV_NAME = ".venv"
REQUIREMENTS_FILE = "requirements.txt"


def main():
    if sys.platform == "win32":
        python_exe = "python"
        venv_python = os.path.join(VENV_NAME, "Scripts", "python.exe")
        venv_pip = os.path.join(VENV_NAME, "Scripts", "pip.exe")
    elif sys.platform in ["linux", "darwin"]:
        python_exe = "python3"
        venv_python = os.path.join(VENV_NAME, "bin", "python")
        venv_pip = os.path.join(VENV_NAME, "bin", "pip")
    else:
        print(f"Unsupported OS: {sys.platform}. Please set up manually.")
        sys.exit(1)

    if not os.path.exists(VENV_NAME):
        print(f"Creating virtual environment in {VENV_NAME}...")
        subprocess.run([python_exe, "-m", "venv", VENV_NAME], check=True)
    else:
        print(f"Virtual environment {VENV_NAME} already exists.")

    if os.path.exists(REQUIREMENTS_FILE):
        print(f"Installing requirements from {REQUIREMENTS_FILE}...")
        subprocess.run([venv_pip, "install", "-r", REQUIREMENTS_FILE], check=True)
        print("Requirements installed successfully.")
    else:
        print(f"No {REQUIREMENTS_FILE} found. Please add it to the repository root.")
        sys.exit(1)

    print(f"Setup complete! Activate the environment with:")
    if sys.platform == "win32":
        print(f"    {VENV_NAME}\\Scripts\\activate")
    else:
        print(f"    source {VENV_NAME}/bin/activate")

    print("Generating initial folders")
    Path("data/images").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
