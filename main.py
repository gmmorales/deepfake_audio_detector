from pathlib import Path

import audioread
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from classifier import ModelNotAvailableError, classifier


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac"}

app = FastAPI(
    title="Detector de Audios: Natural vs. Digital",
    description=(
        "API local para estimar si una grabación proviene de una fuente "
        "física o fue generada electrónicamente."
    ),
    version="0.1.0",
)


@app.get("/")
def home() -> dict[str, str]:
    return {"status": "ok", "message": "API de clasificación de audio activa."}


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)) -> dict[str, str | float]:
    filename = file.filename
    if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Formato no soportado. Se aceptan archivos WAV, MP3, OGG o FLAC.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="El archivo de audio está vacío.")

    try:
        result = await run_in_threadpool(classifier.predict, content)
    except ModelNotAvailableError as error:
        raise HTTPException(
            status_code=503,
            detail="El modelo de clasificación todavía no está disponible.",
        ) from error
    except (sf.LibsndfileError, audioread.NoBackendError) as error:
        raise HTTPException(
            status_code=400,
            detail="No se pudo decodificar el archivo de audio.",
        ) from error

    return {
        "filename": filename,
        "classification": result["label"],
        "confidence_percentage": result["confidence"],
        "raw_score": result["raw_score"],
    }
