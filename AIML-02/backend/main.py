
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel
from typing import Literal

from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import os
import sqlite3
import base64

import requests
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

from PIL import Image
from torchvision import models, transforms

from monai.networks.nets import UNet
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing
)
from monai.inferers import sliding_window_inference


app = FastAPI(title="Medical AI Diagnosis API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Positive-Voxels",
        "X-Total-Voxels",
        "X-Segmentation-Percentage"
    ],
)


ORTHANC_URL = os.getenv("ORTHANC_URL", "http://host.docker.internal:8042")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DENSENET_PATH = (
    PROJECT_ROOT / "outputs" / "models" / "densenet121_finetuned.pth"
)

UNET_PATH = (
    PROJECT_ROOT / "outputs" / "models" / "3d_unet_lung.pth"
)

DATABASE_PATH = PROJECT_ROOT / "backend" / "audit_log.db"

DEVICE = torch.device("cpu")

DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

TARGET_SPACING = (1.0, 1.0, 1.25)
PATCH_SIZE = (64, 64, 64)


def init_audit_database():

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(DATABASE_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            comment TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_audit_database()


class ReviewRequest(BaseModel):

    study_id: str
    decision: Literal["Approve", "Reject"]
    reviewer: str
    comment: str = ""


xray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_densenet():

    checkpoint = torch.load(
        DENSENET_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 14)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


def load_unet():

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="INSTANCE"
    )

    checkpoint = torch.load(
        UNET_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


densenet_model = load_densenet()
unet_model = load_unet()


def generate_gradcam(image, input_tensor, target_idx):

    activation_holder = {}

    target_layer = densenet_model.features.denseblock4

    def forward_hook(module, inputs, output):
        activation_holder["activation"] = output

    handle = target_layer.register_forward_hook(forward_hook)

    try:

        densenet_model.zero_grad(set_to_none=True)

        with torch.enable_grad():

            outputs = densenet_model(input_tensor)

            target_score = outputs[0, target_idx]

            activations = activation_holder["activation"]

            gradients = torch.autograd.grad(
                outputs=target_score,
                inputs=activations,
                retain_graph=False,
                create_graph=False
            )[0]

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            cam = torch.sum(
                weights * activations,
                dim=1,
                keepdim=True
            )

            cam = torch.relu(cam)

            cam = F.interpolate(
                cam,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )

            cam = cam[0, 0]

            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            heatmap = cam.detach().cpu().numpy()

    finally:

        handle.remove()
        densenet_model.zero_grad(set_to_none=True)

    r = np.clip(
        1.5 - np.abs(4 * heatmap - 3),
        0,
        1
    )

    g = np.clip(
        1.5 - np.abs(4 * heatmap - 2),
        0,
        1
    )

    b = np.clip(
        1.5 - np.abs(4 * heatmap - 1),
        0,
        1
    )

    colored_heatmap = np.stack(
        [r, g, b],
        axis=-1
    )

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(
        colored_heatmap,
        mode="RGB"
    )

    original = image.resize((224, 224))

    overlay = Image.blend(
        original,
        heatmap_image,
        alpha=0.4
    )

    heatmap_buffer = BytesIO()
    overlay_buffer = BytesIO()

    heatmap_image.save(
        heatmap_buffer,
        format="PNG"
    )

    overlay.save(
        overlay_buffer,
        format="PNG"
    )

    return {
        "target_disease": DISEASE_CLASSES[target_idx],
        "heatmap_base64": base64.b64encode(
            heatmap_buffer.getvalue()
        ).decode("utf-8"),
        "overlay_base64": base64.b64encode(
            overlay_buffer.getvalue()
        ).decode("utf-8")
    }


@app.get("/")
def home():

    return {
        "message": "Medical AI Diagnosis API is running",
        "models": ["DenseNet121", "3D U-Net"],
        "gradcam": "Enabled",
        "audit_logging": "Enabled"
    }


@app.get("/orthanc/status")
def orthanc_status():

    try:

        response = requests.get(
            f"{ORTHANC_URL}/system",
            timeout=5
        )

        response.raise_for_status()

        return {
            "orthanc_status": "Connected",
            "server": response.json().get("Name", "Orthanc")
        }

    except requests.RequestException as e:

        return {
            "orthanc_status": "Disconnected",
            "error": str(e)
        }


@app.get("/orthanc/studies")
def get_studies():

    try:

        response = requests.get(
            f"{ORTHANC_URL}/studies",
            timeout=10
        )

        response.raise_for_status()

        return {
            "studies": response.json()
        }

    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=f"Orthanc request failed: {str(e)}"
        )


@app.get("/orthanc/studies/{study_id}/volume")
def get_ct_volume(study_id: str):

    try:

        series_response = requests.get(
            f"{ORTHANC_URL}/studies/{study_id}/series",
            timeout=15
        )

        series_response.raise_for_status()

        series_ids = series_response.json()

        if not series_ids:

            raise HTTPException(
                status_code=404,
                detail="No series found"
            )

        slices = []

        for series_id in series_ids:

            instances_response = requests.get(
                f"{ORTHANC_URL}/series/{series_id}/instances",
                timeout=15
            )

            instances_response.raise_for_status()

            for instance in instances_response.json():

                instance_id = instance["ID"]

                file_response = requests.get(
                    f"{ORTHANC_URL}/instances/{instance_id}/file",
                    timeout=30
                )

                file_response.raise_for_status()

                ds = pydicom.dcmread(
                    BytesIO(file_response.content)
                )

                if (
                    ds.get("Modality") == "CT"
                    and hasattr(ds, "PixelData")
                    and hasattr(ds, "ImagePositionPatient")
                ):

                    slices.append(ds)

        if not slices:

            raise HTTPException(
                status_code=404,
                detail="No valid CT slices found"
            )

        slices.sort(
            key=lambda ds: float(ds.ImagePositionPatient[2])
        )

        volume = np.stack([
            ds.pixel_array.astype(np.float32)
            * float(ds.get("RescaleSlope", 1))
            + float(ds.get("RescaleIntercept", 0))
            for ds in slices
        ])

        pixel_spacing = [
            float(x) for x in slices[0].PixelSpacing
        ]

        if len(slices) > 1:

            z_positions = [
                float(ds.ImagePositionPatient[2])
                for ds in slices
            ]

            z_spacing = float(
                np.median(np.abs(np.diff(z_positions)))
            )

        else:

            z_spacing = float(
                slices[0].get("SliceThickness", 1)
            )

        return {
            "study_id": study_id,
            "modality": "CT",
            "number_of_slices": len(slices),
            "volume_shape": list(volume.shape),
            "voxel_spacing_mm": [
                z_spacing,
                pixel_spacing[0],
                pixel_spacing[1]
            ],
            "data_type": str(volume.dtype),
            "hu_min": float(volume.min()),
            "hu_max": float(volume.max()),
            "message": "3D CT volume reconstructed successfully"
        }

    except HTTPException:
        raise

    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=f"Orthanc request failed: {str(e)}"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Volume reconstruction failed: {str(e)}"
        )


@app.post("/predict/xray")
async def predict_xray(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:

        raise HTTPException(
            status_code=400,
            detail="Upload a JPG or PNG chest X-ray image"
        )

    try:

        image_bytes = await file.read()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        input_tensor = xray_transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():

            outputs = densenet_model(input_tensor)

            probabilities = torch.sigmoid(outputs)[0]

        results = {
            disease: round(float(probability), 4)
            for disease, probability in zip(
                DISEASE_CLASSES,
                probabilities
            )
        }

        predicted_idx = int(
            torch.argmax(probabilities).item()
        )

        predicted_probability = float(
            probabilities[predicted_idx]
        )

        gradcam_result = generate_gradcam(
            image,
            input_tensor,
            predicted_idx
        )

        return {
            "filename": file.filename,
            "model": "DenseNet121",
            "image_size": "224x224",
            "predictions": results,
            "gradcam": {
                "target_disease": gradcam_result["target_disease"],
                "target_score": round(predicted_probability, 4),
                "heatmap_base64": gradcam_result["heatmap_base64"],
                "overlay_base64": gradcam_result["overlay_base64"]
            },
            "message": (
                "Experimental AI predictions and Grad-CAM "
                "visualizations only. Not intended for "
                "clinical diagnosis."
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"X-ray prediction failed: {str(e)}"
        )


@app.post("/predict/ct")
async def predict_ct(file: UploadFile = File(...)):

    filename = file.filename or ""

    if not (
        filename.lower().endswith(".nii")
        or filename.lower().endswith(".nii.gz")
    ):

        raise HTTPException(
            status_code=400,
            detail="Upload a NIfTI CT volume (.nii or .nii.gz)"
        )

    temp_input = None
    temp_output = None

    try:

        suffix = (
            ".nii.gz"
            if filename.lower().endswith(".nii.gz")
            else ".nii"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp.write(await file.read())
            temp_input = temp.name

        loader = LoadImage(
            image_only=True,
            dtype=np.float32
        )

        volume = loader(temp_input)

        volume = EnsureChannelFirst(
            channel_dim="no_channel"
        )(volume)

        volume = Orientation(
            axcodes="RAS"
        )(volume)

        volume = Spacing(
            pixdim=TARGET_SPACING,
            mode="bilinear"
        )(volume)

        volume = torch.clamp(
            volume,
            min=-1000,
            max=1000
        )

        volume = (volume + 1000.0) / 2000.0

        input_tensor = volume.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():

            prediction = sliding_window_inference(
                inputs=input_tensor,
                roi_size=PATCH_SIZE,
                sw_batch_size=1,
                predictor=unet_model,
                overlap=0.25
            )

            prediction = torch.sigmoid(prediction)

            mask = (
                prediction[0, 0] > 0.5
            ).cpu().numpy().astype(np.uint8)

        affine = volume.affine.cpu().numpy()

        nifti_mask = nib.Nifti1Image(
            mask,
            affine
        )

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix="_segmentation.nii.gz"
        )

        temp_output = output_file.name
        output_file.close()

        nib.save(
            nifti_mask,
            temp_output
        )

        positive_voxels = int(mask.sum())
        total_voxels = int(mask.size)

        return FileResponse(
            path=temp_output,
            media_type="application/gzip",
            filename="lung_segmentation.nii.gz",
            background=BackgroundTask(
                lambda: os.remove(temp_output)
            ),
            headers={
                "X-Positive-Voxels": str(positive_voxels),
                "X-Total-Voxels": str(total_voxels),
                "X-Segmentation-Percentage": str(
                    round(
                        positive_voxels / total_voxels * 100,
                        4
                    )
                )
            }
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"CT segmentation failed: {str(e)}"
        )

    finally:

        if temp_input and os.path.exists(temp_input):

            os.remove(temp_input)


@app.post("/review")
def save_review(review: ReviewRequest):

    study_id = review.study_id.strip()
    reviewer = review.reviewer.strip()

    if not study_id or not reviewer:

        raise HTTPException(
            status_code=400,
            detail="Study ID and reviewer are required"
        )

    timestamp = datetime.now(timezone.utc).isoformat()

    try:

        conn = sqlite3.connect(DATABASE_PATH)

        cursor = conn.execute(
            """
            INSERT INTO audit_logs
            (study_id, decision, reviewer, comment, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                study_id,
                review.decision,
                reviewer,
                review.comment,
                timestamp
            )
        )

        review_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return {
            "message": "Review decision saved successfully",
            "review_id": review_id,
            "study_id": study_id,
            "decision": review.decision,
            "reviewer": reviewer,
            "timestamp": timestamp
        }

    except sqlite3.Error as e:

        raise HTTPException(
            status_code=500,
            detail=f"Audit logging failed: {str(e)}"
        )


@app.get("/audit-logs")
def get_audit_logs(study_id: str = None):

    try:

        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row

        if study_id:

            rows = conn.execute(
                """
                SELECT * FROM audit_logs
                WHERE study_id = ?
                ORDER BY id DESC
                """,
                (study_id,)
            ).fetchall()

        else:

            rows = conn.execute(
                """
                SELECT * FROM audit_logs
                ORDER BY id DESC
                """
            ).fetchall()

        conn.close()

        return {
            "total_records": len(rows),
            "audit_logs": [
                dict(row) for row in rows
            ]
        }

    except sqlite3.Error as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve audit logs: {str(e)}"
        )


print(
    "API ROUTES:",
    [
        getattr(route, "path", "Unknown")
        for route in app.routes
    ]
)
