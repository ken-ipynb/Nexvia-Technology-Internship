# AI-Assisted Medical Imaging and Diagnosis System

## About the Project

This project is an AI-based medical imaging system developed as part of my Nexvia Technology internship.

The main goal of this project is to use deep learning models to analyze medical images. It includes a chest X-ray classification model and a 3D U-Net model for CT scan lung segmentation.

I also integrated a FastAPI backend, React frontend, SQLite database, and Orthanc DICOM server to make the system more complete.

## Technologies Used

- **Python** – Main programming language
- **PyTorch** – Building and using deep learning models
- **MONAI** – Medical imaging deep learning
- **DenseNet121** – Chest X-ray classification
- **3D U-Net** – CT scan segmentation
- **FastAPI** – Backend and API development
- **React + Vite** – Frontend dashboard
- **Grad-CAM** – Visualizing model predictions
- **SQLite** – Storing audit logs
- **Orthanc** – DICOM server
- **Docker** – Running the application in containers
- **DVC** – Dataset and model version tracking
- **3D Slicer** – Viewing CT scans and segmentation results
- **NumPy, Pandas** – Data handling and preprocessing

## Project Features

1. **Chest X-ray Classification:** Uses DenseNet121 to classify chest X-ray images into 14 categories.
2. **CT Scan Segmentation:** Uses a 3D U-Net model to generate lung segmentation masks from CT scans.
3. **Grad-CAM:** Generates heatmaps to show which parts of an X-ray influenced the model's prediction.
4. **DICOM Integration:** Uses Orthanc to manage and access DICOM medical images.
5. **FastAPI Backend:** Provides API endpoints for predictions, segmentation and other operations.
6. **React Dashboard:** Provides an interface to upload medical images and view results.
7. **Audit Logging:** Stores application activity records using SQLite.
8. **Clinical Review:** Allows review feedback to be recorded through the backend.
9. **Docker Support:** Runs the frontend and backend using Docker Compose.

## How the System Works

The system has different parts that work together:

1. The user uploads a medical image through the frontend or uses the API.
2. The backend receives the uploaded image.
3. The selected deep learning model processes the image.
4. The system generates a classification result or segmentation mask.
5. Grad-CAM can generate a heatmap for X-ray predictions.
6. The result is returned to the frontend, where it can be viewed.
7. Relevant application activities are stored in the audit logs.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | Checks whether the API is running |
| `/orthanc/status` | Checks Orthanc connection |
| `/orthanc/studies` | Gets available DICOM studies |
| `/orthanc/studies/{study_id}/volume` | Gets volume information |
| `/predict/xray` | Runs chest X-ray classification |
| `/predict/ct` | Runs CT scan segmentation |
| `/review` | Records review feedback |
| `/audit-logs` | Gets stored audit records |

## Project Setup

### 1. Clone or Open the Project

Open the project folder in VS Code.

### 2. Start the Backend

Activate the Conda environment:

```bash
conda activate medical_ai
```

Run the backend:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Backend URL: `http://127.0.0.1:8000`

API documentation: `http://127.0.0.1:8000/docs`

### 3. Start the Frontend

Open another terminal:

```bash
cd Frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:5173`

## Running with Docker

I also configured Docker to run the frontend and backend separately.

Make sure Docker Desktop is running and execute these commands from the project root folder.

Build Docker images:

```bash
docker compose build
```

Start the containers:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

Application URLs:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

To stop the containers:

```bash
docker compose down
```

## Dataset and Model Versioning (DVC)

This project uses DVC to track large datasets and trained model files. The actual data is stored in Google Drive, while DVC tracking files are included in the project.

After configuring Google Drive access and setting up DVC, run:

```bash
dvc pull
```

This downloads the tracked datasets and model files from remote storage.

Google Drive access and DVC remote authentication must be configured separately. OAuth credentials should not be committed to the repository.

## Testing and Validation

I tested the application using both the frontend and backend.

The following features were tested successfully:

- Docker images were built successfully.
- Both frontend and backend containers started successfully.
- X-ray classification returned predictions.
- Grad-CAM generated heatmaps.
- CT scan segmentation worked successfully.
- Clinical review endpoint worked.
- Audit logs were stored and retrieved successfully.
- Frontend and backend were accessible through Docker.

These tests confirm that the application works technically, but they do not prove that the models are medically accurate.

## Project Outputs

- Chest X-ray classification results
- Grad-CAM heatmaps
- CT segmentation masks in NIfTI format
- Segmentation statistics
- DICOM study information through Orthanc
- Audit logs and review records
- Web dashboard for interacting with the system

## Future Improvements

- Improve model performance with better training data and tuning.
- Improve CT segmentation results.
- Add user authentication.
- Add prediction history.
- Improve the dashboard with more visualization options.
- Improve application stability and deployment.

## Note

This project is developed for learning and research purposes. It is not clinically validated and its predictions should not be used for real medical diagnosis or treatment.

---

**Developed as part of my Nexvia Technology Internship.**