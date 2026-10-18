# FastAPI app: serves the frontend, runs the CNN on an optional fridge photo, merges it with typed ingredients
# returns ranked results for recipes.
import io
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from psycopg2.pool import ThreadedConnectionPool
from pydantic import BaseModel, Field

from db import DEFAULT_URL, PAGE_SIZE, class_names, ranked_recipes
from detector import FoodDetector
from ingredient_map import classify

BASE = Path(__file__).parent
FRONTEND = BASE.parent / "frontend"
MAX_UPLOAD = 10 * 1024 * 1024
MAX_TYPED = 40


def label(name):
    return name.replace("_", " ")


def parse_typed(text, classes):
    # Split a comma/newline separated string into (recognized class names, unrecognized strings).
    known, unknown = [], []
    for raw in re.split(r"[,;\n]", text or ""):
        item = re.sub(r"\s+", " ", raw).strip()
        if not item or len(item) > 60:
            continue
        key = item.lower().replace(" ", "_")
        if key in classes:                       
            cls = key
        else:
            kind, cls = classify(item, classes)  
            if kind != "class":
                cls = None
        if cls:
            if cls not in known:
                known.append(cls)
        elif item.lower() not in (u.lower() for u in unknown):
            unknown.append(item)
    return known, unknown


@asynccontextmanager
async def lifespan(app):
    app.state.pool = ThreadedConnectionPool(1, 5, os.environ.get("DATABASE_URL", DEFAULT_URL))
    conn = app.state.pool.getconn()
    try:
        app.state.classes = class_names(conn)
    finally:
        app.state.pool.putconn(conn)
    model_path = Path(os.environ.get("MODEL_PATH", BASE / "artifacts" / "best_model.pth"))
    app.state.detector = FoodDetector(model_path) if model_path.exists() else None
    if app.state.detector is None:
        print(f"WARNING: {model_path} not found - photo search disabled until you run train.py.")
    yield
    app.state.pool.closeall()


app = FastAPI(title="Fridge to Recipe", lifespan=lifespan)


def page_of_recipes(have, offset):
    pool = app.state.pool
    conn = pool.getconn()
    try:
        return ranked_recipes(have, offset=offset, conn=conn)
    finally:
        pool.putconn(conn)


@app.get("/api/health")
def health():
    return {"model_loaded": app.state.detector is not None}


@app.post("/api/analyze-fridge")
def analyze_fridge(file: Optional[UploadFile] = File(None), ingredients: str = Form("")):
    classes = app.state.classes
    typed, unrecognized = parse_typed(ingredients, classes)
    if len(typed) + len(unrecognized) > MAX_TYPED:
        raise HTTPException(400, f"Please enter at most {MAX_TYPED} ingredients.")

    detected = []
    if file is not None and file.filename:
        detector = app.state.detector
        if detector is None:
            raise HTTPException(503, "Photo search is unavailable: the model file best_model.pth is missing. "
                                     "Run train.py first, or search by typing ingredients.")
        data = file.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "Image too large (max 10 MB).")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "That file is not a readable image.")
        detected = detector.detect(img)
    elif not typed and not unrecognized:
        raise HTTPException(400, "Upload a photo, type some ingredients, or both.")

    # merge + dedupe: CNN detections first (best first), then typed ones not already present
    active = [d["name"] for d in detected]
    active += [t for t in typed if t not in active]

    recipes, has_more = page_of_recipes(active, 0) if active else ([], False)
    return {
        "detected": [{"name": d["name"], "label": label(d["name"]), "confidence": d["confidence"]} for d in detected],
        "typed": [{"name": t, "label": label(t)} for t in typed],
        "unrecognized": unrecognized,
        "active_ingredients": active,
        "recipes": recipes,
        "has_more": has_more,
        "page_size": PAGE_SIZE,
    }


class MoreRequest(BaseModel):
    ingredients: List[str] = Field(..., min_length=1, max_length=100)
    offset: int = Field(..., ge=0, le=100_000)


@app.post("/api/recipes")
def more_recipes(req: MoreRequest):
    have = [i for i in dict.fromkeys(req.ingredients) if i in app.state.classes]
    if not have:
        raise HTTPException(400, "No valid ingredients supplied.")
    recipes, has_more = page_of_recipes(have, req.offset)
    return {"recipes": recipes, "has_more": has_more}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
