"""FastAPI app: serves the frontend, runs the CNN on an uploaded fridge photo, and ranks recipes.

    uvicorn main:app --reload
Env: DATABASE_URL (default postgresql://localhost/fridge_recipes), MODEL_PATH (default best_model.pth)
"""
import io
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from psycopg2.pool import ThreadedConnectionPool

from db import DEFAULT_URL, top_recipes
from detector import FoodDetector

BASE = Path(__file__).parent
MAX_UPLOAD = 10 * 1024 * 1024
UNITS = r"(?:cups?|c\.|tbsp\.?|tablespoons?|tsp\.?|teaspoons?|oz\.?|ounces?|lbs?\.?|pounds?|g|grams?|kg|ml|l|cans?|packages?|pinch|dash|cloves?|slices?|sticks?|bunch|heads?)"


def tidy_ingredient(text):
    """'1 pound tagliatelle, cooked' -> 'tagliatelle'. Returns None for junk lines."""
    t = re.sub(r"\([^)]*\)", " ", text).split(",")[0]
    t = re.sub(rf"^[\s\d½¼¾⅓⅔/.\-–]*(?:{UNITS}\b)?\s*(?:of\s+)?", "", t, flags=re.I).strip()
    return t if len(t) >= 3 else None


def display_name(name):
    return name.replace("_", " ")


@asynccontextmanager
async def lifespan(app):
    app.state.pool = ThreadedConnectionPool(1, 5, os.environ.get("DATABASE_URL", DEFAULT_URL))
    model_path = Path(os.environ.get("MODEL_PATH", BASE / "best_model.pth"))
    app.state.detector = FoodDetector(model_path) if model_path.exists() else None
    if app.state.detector is None:
        print(f"WARNING: {model_path} not found - train first (python train.py). /api/analyze-fridge will return 503.")
    yield
    app.state.pool.closeall()


app = FastAPI(title="Fridge to Recipe", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"model_loaded": app.state.detector is not None}


@app.post("/api/analyze-fridge")
def analyze_fridge(file: UploadFile = File(...)):
    detector = app.state.detector
    if detector is None:
        raise HTTPException(503, "Model not loaded: best_model.pth is missing. Run train.py first.")
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image too large (max 10 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file isn't a readable image.")

    detected = detector.detect(img)
    names = [d["name"] for d in detected]

    recipes = []
    if names:
        pool = app.state.pool
        conn = pool.getconn()
        try:
            rows = top_recipes(names, conn)
        finally:
            pool.putconn(conn)
        for r in rows:
            other = [tidy_ingredient(m) for m in r["missing_ingredients"]]
            recipes.append({
                "id": r["id"],
                "title": r["title"],
                "instructions": r["instructions"],
                "matched_count": r["matched_count"],
                "missing_count": r["missing_count"],
                "matched_ingredients": [display_name(m) for m in r["matched_ingredients"]],
                "missing_ingredients": [display_name(m) for m in other if m],
            })
    return {
        "detected": [{"name": display_name(d["name"]), "confidence": d["confidence"]} for d in detected],
        "recipes": recipes,
    }


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
