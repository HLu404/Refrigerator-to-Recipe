"""Download a public recipe dataset and load it into PostgreSQL, normalized to the CNN classes.

Dataset: "Food Ingredients and Recipes" (Epicurious, ~13.5k recipes)
         https://huggingface.co/datasets/Hieu-Pham/kaggle_food_recipes  (CC BY-SA 3.0)

Usage:
    createdb fridge_recipes
    export DATABASE_URL=postgresql://localhost/fridge_recipes
    python seed_db.py --csv Clean_Data.csv            # add --reset to wipe and reload
"""
import argparse
import ast
import csv
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

from db import get_connection
from ingredient_map import classify

DATA_URL = ("https://huggingface.co/datasets/Hieu-Pham/kaggle_food_recipes/resolve/main/"
            "Food%20Ingredients%20and%20Recipe%20Dataset%20with%20Image%20Name%20Mapping.csv")
SOURCE = "Epicurious via Hieu-Pham/kaggle_food_recipes (CC BY-SA 3.0)"


def clean_name(col):
    # identical to train.py so names match the CNN's class names
    return col.strip().lower().replace(" ", "_")


def load_classes(csv_path):
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    return [clean_name(h) for h in header if h not in ("", "filename")]


def download(dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DATA_URL}")
    try:
        import certifi  # python.org builds on macOS ship without CA certs
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(DATA_URL, context=ctx) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


def parse_recipes(path, classes):
    """Yield (title, instructions, {class: original_text}, [other ingredient texts])."""
    allowed = set(classes)
    csv.field_size_limit(sys.maxsize)
    seen = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = (row.get("Title") or "").strip()
            instructions = (row.get("Instructions") or "").strip()
            if not title or not instructions or title.lower() in seen:
                continue
            try:
                lines = ast.literal_eval(row["Cleaned_Ingredients"])
            except (ValueError, SyntaxError, KeyError):
                continue
            matched, others = {}, []
            for line in lines:
                kind, name = classify(line, allowed)
                if kind == "class":
                    matched.setdefault(name, line)
                elif kind == "other":
                    others.append(line.strip())
            if not matched:  # nothing the CNN could ever match
                continue
            seen.add(title.lower())
            yield title, instructions, matched, others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Clean_Data.csv", help="CNN label CSV (defines the class list)")
    ap.add_argument("--data", default="data/recipes.csv", help="where to cache the downloaded dataset")
    ap.add_argument("--database-url", default=None, help="default: $DATABASE_URL")
    ap.add_argument("--reset", action="store_true", help="drop and recreate all tables first")
    args = ap.parse_args()

    classes = load_classes(args.csv)
    print(f"{len(classes)} CNN classes")
    recipes_path = download(args.data)

    conn = get_connection(args.database_url)
    try:
        with conn, conn.cursor() as cur:
            if args.reset:
                cur.execute("DROP TABLE IF EXISTS recipe_ingredients, recipes, ingredients CASCADE")
            cur.execute(Path(__file__).with_name("schema.sql").read_text())

            execute_values(cur, "INSERT INTO ingredients (name) VALUES %s ON CONFLICT (name) DO NOTHING",
                           [(c,) for c in classes])
            cur.execute("SELECT name, id FROM ingredients")
            ing_id = dict(cur.fetchall())

            cur.execute("SELECT COUNT(*) FROM recipes")
            if cur.fetchone()[0]:
                sys.exit("recipes table is not empty; rerun with --reset to reload")

            n = 0
            for title, instr, matched, others in parse_recipes(recipes_path, classes):
                cur.execute(
                    "INSERT INTO recipes (title, instructions, source, n_class_ingredients, n_other, other_ingredients) "
                    "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (title, instr, SOURCE, len(matched), len(others), others))
                rid = cur.fetchone()[0]
                execute_values(cur, "INSERT INTO recipe_ingredients (recipe_id, ingredient_id, original_text) VALUES %s",
                               [(rid, ing_id[c], t) for c, t in matched.items()])
                n += 1
        print(f"inserted {n} recipes")
    except psycopg2.Error as e:
        sys.exit(f"database error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
