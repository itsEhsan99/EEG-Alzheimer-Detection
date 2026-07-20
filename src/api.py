"""Web API for the EEG Alzheimer's detection tool."""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from src.visualization import (plot_raw_signal, plot_band_powers, plot_plv_heatmap, plot_plv_topomap)
from src.data_loader import load_recording
from src.preprocessing import apply_preprocessing
from src.gnn_inference import load_gnn, predict_recording_gnn

app = FastAPI(title="EEG Alzheimer's Detection")


# Serve the static frontend files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model once at startup, not on every request.
model, mu, sd = load_gnn()

def load_and_preprocess(file, resample_freq, low_freq, high_freq,
                        notch_freq, drop_channels):
    """Save the uploaded file, load it, and apply preprocessing.

    Returns (raw, tmp_dir) — the caller is responsible for cleaning up
    tmp_dir afterwards.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    raw = load_recording(tmp_path)

    drop_list = [c.strip() for c in drop_channels.split(",") if c.strip()] \
        if drop_channels else None

    raw = apply_preprocessing(
        raw,
        resample_freq=resample_freq if resample_freq and resample_freq > 0 else None,
        low_freq=low_freq if low_freq and low_freq > 0 else None,
        high_freq=high_freq if high_freq and high_freq > 0 else None,
        notch_freq=notch_freq if notch_freq and notch_freq > 0 else None,
        drop_channels=drop_list,
    )
    return raw, tmp_dir

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

        result = predict_recording_gnn(tmp_path, model=model, mu=mu, sd=sd)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Common preprocessing form fields, repeated per endpoint
@app.post("/signal")
async def signal(
    file: UploadFile = File(...),
    start_time: float = Form(0.0),
    n_seconds: float = Form(10.0),
    channels: str = Form(""),
    resample_freq: float = Form(0),
    low_freq: float = Form(0),
    high_freq: float = Form(0),
    notch_freq: float = Form(0),
    drop_channels: str = Form(""),
):
    if not file.filename.endswith(".set"):
        raise HTTPException(status_code=400, detail="Please upload a .set file.")

    channel_list = [c.strip() for c in channels.split(",") if c.strip()]
    try:
        raw, tmp_dir = load_and_preprocess(
            file, resample_freq, low_freq, high_freq, notch_freq, drop_channels)
        png_bytes = plot_raw_signal(
            raw, start_time=start_time, n_seconds=n_seconds, channels=channel_list)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/psd")
async def psd(
    file: UploadFile = File(...),
    resample_freq: float = Form(0),
    low_freq: float = Form(0),
    high_freq: float = Form(0),
    notch_freq: float = Form(0),
    drop_channels: str = Form(""),
):
    if not file.filename.endswith(".set"):
        raise HTTPException(status_code=400, detail="Please upload a .set file.")
    try:
        raw, tmp_dir = load_and_preprocess(
            file, resample_freq, low_freq, high_freq, notch_freq, drop_channels)
        png_bytes = plot_band_powers(raw)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/plv-heatmap")
async def plv_heatmap(
    file: UploadFile = File(...),
    band: str = Form("alpha"),
    resample_freq: float = Form(0),
    low_freq: float = Form(0),
    high_freq: float = Form(0),
    notch_freq: float = Form(0),
    drop_channels: str = Form(""),
):
    if not file.filename.endswith(".set"):
        raise HTTPException(status_code=400, detail="Please upload a .set file.")
    try:
        raw, tmp_dir = load_and_preprocess(
            file, resample_freq, low_freq, high_freq, notch_freq, drop_channels)
        png_bytes = plot_plv_heatmap(raw, band=band)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/plv-topomap")
async def plv_topomap(
    file: UploadFile = File(...),
    band: str = Form("alpha"),
    resample_freq: float = Form(0),
    low_freq: float = Form(0),
    high_freq: float = Form(0),
    notch_freq: float = Form(0),
    drop_channels: str = Form(""),
):
    if not file.filename.endswith(".set"):
        raise HTTPException(status_code=400, detail="Please upload a .set file.")
    try:
        raw, tmp_dir = load_and_preprocess(
            file, resample_freq, low_freq, high_freq, notch_freq, drop_channels)
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
