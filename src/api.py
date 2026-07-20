"""Web API for the EEG Alzheimer's detection tool."""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from src.visualization import (plot_raw_signal, plot_band_powers, plot_plv_heatmap, plot_plv_topomap)
from src.data_loader import load_recording
from src.inference import load_model, predict_recording

app = FastAPI(title="EEG Alzheimer's Detection")

# Serve the static frontend files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model once at startup, not on every request.
model = load_model()


@app.get("/app")
def serve_frontend():
    """Serve the web interface."""
    return FileResponse("static/index.html")


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
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/signal")
async def signal(
    file: UploadFile = File(...),
    start_time: float = Form(0.0),
    n_seconds: float = Form(10.0),
    channels: str = Form(""),
):
    """Return a PNG plot of a chosen window of chosen EEG channels."""
    if not file.filename.endswith(".set"):
        raise HTTPException(status_code=400, detail="Please upload a .set file.")

    # channels arrives as a comma-separated string; split into a list
    channel_list = [c.strip() for c in channels.split(",") if c.strip()]

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw = load_recording(tmp_path)
        png_bytes = plot_raw_signal(
            raw, start_time=start_time, n_seconds=n_seconds,
            channels=channel_list,
        )
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/psd")
async def psd(file: UploadFile = File(...)):
    """Return a PNG bar chart of band powers from an uploaded file."""
    if not file.filename.endswith(".set"):
        raise HTTPException(status_code=400, detail="Please upload a .set file.")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw = load_recording(tmp_path)
        png_bytes = plot_band_powers(raw)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.post("/plv-heatmap")
async def plv_heatmap(file: UploadFile = File(...), band: str = Form("alpha")):
    """Return a heatmap of the mean PLV connectivity matrix for a band."""
    if not file.filename.endswith(".set"):
        raise HTTPException(status_code=400, detail="Please upload a .set file.")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        raw = load_recording(tmp_path)
        png_bytes = plot_plv_heatmap(raw, band=band)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/plv-topomap")
async def plv_topomap(file: UploadFile = File(...), band: str = Form("alpha")):
    """Return the strongest PLV connections on a head layout for a band."""
    if not file.filename.endswith(".set"):
        raise HTTPException(status_code=400, detail="Please upload a .set file.")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        raw = load_recording(tmp_path)
        png_bytes = plot_plv_topomap(raw, band=band)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/metadata")
async def metadata(file: UploadFile = File(...)):
    """Return metadata about an uploaded recording."""
    if not file.filename.endswith(".set"):
        raise HTTPException(status_code=400, detail="Please upload a .set file.")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw = load_recording(tmp_path)
        info = raw.info
        duration = raw.n_times / info["sfreq"]

        return {
            "filename": file.filename,
            "sampling_frequency": float(info["sfreq"]),
            "n_channels": int(info["nchan"]),
            "channel_names": list(raw.ch_names),
            "duration_seconds": round(float(duration), 1),
            "duration_minutes": round(float(duration) / 60, 1),
            "n_samples": int(raw.n_times),
            "highpass": float(info["highpass"]),
            "lowpass": float(info["lowpass"]),
            "bad_channels": list(info["bads"]) if info["bads"] else [],
            "n_bad_channels": int(len(info["bads"])),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
