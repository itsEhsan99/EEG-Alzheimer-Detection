"""Web API for the EEG Alzheimer's detection tool."""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException

from src.inference import load_model, predict_recording

app = FastAPI(title="EEG Alzheimer's Detection")

# Load the model once at startup, not on every request.
model = load_model()


@app.get("/")
def read_root():
    """A simple endpoint to check the API is alive."""
    return {"message": "EEG Alzheimer's Detection API is running"}


@app.get("/health")
def health_check():
    """Report basic service health."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an uploaded .set recording and return a prediction."""
    if not file.filename.endswith(".set"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an EEGLAB .set file.",
        )

    # Save the uploaded file to a temporary location on disk,
    # because MNE reads from a file path, not from memory.
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_recording(tmp_path, model=model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary files no matter what.
        shutil.rmtree(tmp_dir, ignore_errors=True)