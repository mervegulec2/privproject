# Federated Learning Baseline - CIFAR-10

This project implements a baseline for Federated Learning where a central server initializes a model and "distributes" it to clients. Each client then trains the model on their own non-IID subset of the CIFAR-10 dataset separately.

## 📁 Project Structure
- `run_baseline.py`: Main entry point for the simulation.
- `download_data.py`: Script to download the CIFAR-10 dataset.
- `src/`: Core logic modules (Models, Data Utils, Training Utils).
- `data/`: Raw dataset storage.
- `outputs/`: Metrics and data splits.

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**:
   ```bash
   python download_data.py
   ```

3. **Run Baseline Simulation**:
   ```bash
   python run_baseline.py
   ```

## 📊 Project Logic
- **Non-IID Partitioning**: Splitting CIFAR-10 among 15 clients using **Dirichlet ($\alpha=0.1$)**.
- **Model**: ResNet-18 (plain baseline).
- **Process**: Independent local training (5 epochs) for each client.
- **Goal**: Measure performance across 3 random seeds.
