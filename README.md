# privproject — One-Shot Prototype FL (CIFAR-10)

## Local Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data/download_cifar10.py

## Colab Setup
!git clone <REPO_URL>
%cd privproject
!pip install -r requirements.txt
!python src/data/download_cifar10.py
