# SatVision LULC

A **Satellite Land Use & Land Cover (LULC)** web application built with **PyTorch**, **Flask**, and **Grad-CAM** on the **EuroSAT** (Sentinel-2) dataset across 10 land cover classes (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`).

## Project Structure

```text
satellite-lulc-app/
├── data/                    # EuroSAT dataset (auto-downloaded or local)
├── models/                  # Saved weights (e.g., eurosat_resnet18.pth)
├── templates/
│   └── index.html           # Web UI for tile upload, probabilities & Grad-CAM
├── app.py                   # Flask server and /predict endpoint
├── model_utils.py           # ResNet-18 builder, transforms & Grad-CAM hooks
├── train.py                 # Training script for EuroSAT
├── requirements.txt         # Python dependencies
├── README.md                # Local setup and usage guide
└── .gitignore
```

## Local Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd satellite-lulc-app
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

* **Windows (Command Prompt):**
  ```bat
  venv\Scripts\activate
  ```
* **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```
* **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

Run `train.py` to download or load the EuroSAT dataset (27,000 images with an 80/20 stratified train/validation split) and save the best checkpoint to `models/eurosat_resnet18.pth`:

```bash
python -u train.py
```

### CPU Training Tips
* Set `num_workers=0` in `DataLoader` on Windows to avoid multiprocessing hangs.
* Freezing the pretrained `ResNet-18` backbone and training only the classification head (`model.fc`) significantly reduces CPU training time.

## Run the Web Application

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```
3. Upload any satellite image tile (such as sample `.jpg` images inside `data/eurosat/2750/`) and click **Analyze Tile** to view:
   * Primary land cover classification and confidence percentage
   * Top-3 class probability distribution
   * Input satellite patch alongside the **Grad-CAM** spatial heatmap overlay