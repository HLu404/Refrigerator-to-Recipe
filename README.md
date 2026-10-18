# Fridge to Recipe

A full-stack web app that turns a photo of your refrigerator into dinner ideas. A custom PyTorch
convolutional neural network detects the foods in the photo, you can add more ingredients by typing,
and a PostgreSQL query ranks recipes from the closest ingredient match to the loosest.

Each recipe card shows a photo of the finished dish, every ingredient with its exact amount
(highlighting what you have and what you need), and numbered cooking steps.

## Sample Images

Main Page:
<img width="1470" height="807" alt="image" src="https://github.com/user-attachments/assets/ae337bee-a60b-4c6c-a747-d073f7a187fa" />

Uploading a Recipe:
<img width="672" height="746" alt="image" src="https://github.com/user-attachments/assets/9bee0e68-1eb0-4813-80a7-dc96961f9e70" />
<img width="497" height="712" alt="image" src="https://github.com/user-attachments/assets/01417290-4ede-44b9-b6fe-cf9190f2712e" />


## Features

- **Three ways to search:** upload a photo, type ingredients, or combine both. Detected and typed
  ingredients are merged and deduplicated before the database is queried.
- **Multi-item detection:** one fridge photo contains many foods, so the CNN is a multi-label
  classifier (one sigmoid output per food) with per-class confidence thresholds.
- **Ranked results:** recipes are ordered by most matched ingredients, then fewest missing ones.
  The top 5 are shown first and "See more" loads the next 5 from the database without re-running
  the image model.
- **Measured ingredients and numbered steps** for every recipe, with a fallback image if a dish
  photo fails to load.

## How it works

1. **Input.** The browser sends the photo and/or the typed ingredient list to the FastAPI backend.
2. **Detection.** `detector.py` runs the CNN on the full image, its mirror image, and a 2x2 grid of
   overlapping crops (to catch small items), then keeps every food above its tuned threshold.
3. **Normalization.** Typed ingredients are mapped to the same 57 food classes the CNN predicts.
4. **Ranking.** PostgreSQL counts how many of the recipe's ingredients you have and how many you lack:

   ```
   ORDER BY matched DESC, missing ASC, title, id    LIMIT 5 OFFSET n
   ```

## The CNN

- **Data:** about 9,000 labeled images. They were aggregated from four public fridge and food datasets,
  and near-duplicates were removed with perceptual hashing. 87 overlapping labels were merged into
  57 standardized foods (see `dataset/data_cleaning.ipynb`).
- **Architecture:** a small ResNet-style network trained from scratch (about 7.8M parameters): a
  strided stem, four stages of residual blocks, global average pooling, dropout and a linear head.
- **Training:** binary cross-entropy with class weighting (the rarest class has about 30 examples,
  the most common about 2,800), AdamW with a one-cycle learning-rate schedule, augmentation (random
  crop, flip, color jitter), early stopping on validation macro-F1, and per-class thresholds tuned on
  the validation set.
- **Known limits:** the model can only recognize its 57 classes. There is no pork, bacon, fish or lamb
  and no sauces or condiments, because the source datasets did not label them. Rare classes
  (for example sausage, cauliflower, pear) are detected less reliably. Typed ingredients outside the
  57 classes are reported as "not recognized" and ignored.

## The recipe data

Recipes come from [TheMealDB](https://www.themealdb.com) (about 780 usable recipes), fetched by
`seed_db.py`. Each ingredient line is stored in full ("1.5kg Beef Brisket") and also mapped to a food
class by rules in `ingredient_map.py`. Instructions are split into numbered steps.
TheMealDB's free API key is intended for development and education.

## Project structure

```
backend/                 Python: API, model, database
  main.py                FastAPI app (serves the frontend, /api/analyze-fridge, /api/recipes)
  detector.py            loads the trained CNN and detects foods in a photo
  model.py               CNN architecture (FoodCNN)
  train.py               training script
  db.py                  ranking query with pagination
  seed_db.py             downloads TheMealDB recipes and loads PostgreSQL
  schema.sql             database schema
  ingredient_map.py      maps ingredient text to the CNN's food classes
  artifacts/             trained model (best_model.pth) and class_mapping.json
frontend/                index.html, styles.css, app.js (plain HTML, CSS and JavaScript)
dataset/                 training labels (Clean_Data.csv) and the data-cleaning notebook
requirements.txt
```

## Setup

Requirements: Python 3.10+ and PostgreSQL 14+ (the images for training are not included in this repo).

```
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

createdb fridge_recipes
export DATABASE_URL=postgresql://localhost/fridge_recipes

cd backend
../.venv/bin/python seed_db.py --reset                    # download recipes and load the database
../.venv/bin/python train.py --images /path/to/images     # train the CNN (writes artifacts/best_model.pth)
```

`train.py` expects a flat folder of images matching the filenames in `dataset/Clean_Data.csv`.
It uses an NVIDIA GPU or Apple Silicon (MPS) when available, otherwise the CPU.

## Run

```
cd backend
export DATABASE_URL=postgresql://localhost/fridge_recipes
../.venv/bin/python -m uvicorn main:app
```

Open http://localhost:8000. Restart the server after retraining so it loads the new model.
If `best_model.pth` is missing, typed-ingredient search still works and photo search shows an error.

## API

| Endpoint | Description |
| --- | --- |
| `POST /api/analyze-fridge` | Form fields: optional `file` (image, max 10 MB) and optional `ingredients` (comma-separated). Returns detections, the merged ingredient list and the first 5 recipes. |
| `POST /api/recipes` | JSON `{"ingredients": [...], "offset": 5}`. Returns the next 5 recipes and whether more remain. Does not run the CNN. |
| `GET /api/health` | Whether the model is loaded. |

## Tech stack

Python, FastAPI, PyTorch, torchvision, PostgreSQL (psycopg2), HTML, CSS and vanilla JavaScript.
Fonts: Fraunces and Plus Jakarta Sans.
