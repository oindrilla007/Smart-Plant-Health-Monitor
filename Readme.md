# Smart Plant Health Monitor

A machine learning project for plant disease detection and watering prediction using image and sensor data.

## Project Objective

This project monitors plant health by:

- Identifying plant diseases from leaf images
- Detecting infected regions on leaves
- Predicting whether a plant needs watering from sensor readings

## Models Used

| Task | Model |
| --- | --- |
| Disease classification | ResNet50 |
| Disease detection | YOLOv8 |
| Watering prediction | LSTM |

## Dataset

### Plant Disease Dataset

- Source: Kaggle - New Plant Diseases Dataset
- 38 disease and healthy classes
- RGB leaf images

### Sensor Dataset

- Includes temperature, humidity, soil moisture, light intensity, NPK values, and health score
- Sensor data is synthetically generated for time-series analysis

## Project Structure

```text
SPH/
|-- dataset_subset/      # Image dataset for training
|-- dataset_yolo_mini/   # YOLO formatted dataset
|-- data/timeseries/     # Sensor data
|-- models/              # Saved models
|-- train_resnet.py      # ResNet training
|-- train_yolo.py        # YOLO training
|-- train_lstm.py        # LSTM training
|-- pipeline.py          # Combined prediction pipeline
|-- main.py              # FastAPI backend
`-- streamlit_app.py     # Web interface
```

## Installation

### Requirements

- Python 3.8+
- CUDA GPU (optional but recommended)

### Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Execution Guide

### Model Training

```bash
# Train ResNet
python train_resnet.py

# Train YOLO
python train_yolo.py

# Train LSTM
python generate_sensor_data.py
python train_lstm.py
```

### Running the Application

1. Start the backend:

```bash
python main.py
```

2. Start the frontend:

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`.

## How It Works

1. User uploads a leaf image.
2. ResNet classifies the disease.
3. YOLO detects infected regions.
4. Sensor data is processed by LSTM.
5. The app returns disease and watering predictions.

## Results

- Disease classification accuracy: ~89%
- Disease detection mAP: ~85%
- Watering prediction accuracy: ~92%

## Tools and Technologies

- Python
- PyTorch
- YOLOv8
- FastAPI
- Streamlit
- OpenCV
- NumPy
- Pandas

## Conclusion

This project combines multiple machine learning models to support early disease detection and precision irrigation decisions.

## Contributor

- [Oindrilla](https://github.com/oindrilla007)
