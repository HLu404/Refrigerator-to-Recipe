# Load TheMealDB recipes (dish image, measured ingredients, instructions) into PostgreSQL,
# with each ingredient also normalized to the CNN's food classes.
import argparse
import csv
import json
import re
import ssl
import string
import sys
import urllib.request
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

from db import get_connection
from ingredient_map import classify

ROOT = Path(__file__).resolve().parent.parent
API = "https://www.themealdb.com/api/json/v1/1/search.php?f={}"
EMOJI = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]") 


def clean_name(col):
    return col.strip().lower().replace(" ", "_")


def load_classes(csv_path):
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    return [clean_name(h) for h in header if h not in ("", "filename")]


def clean_text(s):
    return re.sub(r"\s+", " ", EMOJI.sub("", s or "")).strip()


def fetch_all(cache):
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        import certifi 
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    meals = {}
    for letter in string.ascii_lowercase:
        print(f"fetching meals starting with '{letter}'", end="\r")
        with urllib.request.urlopen(API.format(letter), context=ctx, timeout=30) as r:
            for m in json.load(r).get("meals") or []:
                meals[m["idMeal"]] = m
    print(f"fetched {len(meals)} meals          ")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(list(meals.values())))
    return list(meals.values())


# instructions 
_HEADER = re.compile(r"^(step\s*\d+|\d+)\s*[.:)\-]*$", re.I)
_PREFIX = re.compile(r"^([*\u2022\-]+\s*)?(step\s*\d+\s*[.:)\-]*|\d+\s*[.)]|\d+\s+-)\s*", re.I)
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_ABBREV = re.compile(r"(?:\b[A-Za-z]|approx|min|mins|oz|tbsp|tsp|lb|lbs|no|vs|etc|e\.g|i\.e)\.$", re.I)


def sentences(text):
    out, buf = [], ""
    for part in _SENT_END.split(text):
        buf = f"{buf} {part}".strip() if buf else part
        if _ABBREV.search(buf):
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def split_steps(raw, target=110, max_len=420): # split steps into numbers
    lines = [clean_text(l) for l in (raw or "").replace("\r", "\n").split("\n")]
    lines = [l for l in lines if l and not _HEADER.match(l)]
    lines = [_PREFIX.sub("", l).strip() for l in lines]
    lines = [l for l in lines if l]

    merged, carry = [], ""
    for l in lines:
        if l.endswith(":") and len(l) < 45:
            carry = f"{carry} {l}".strip()
            continue
        merged.append(f"{carry} {l}".strip())
        carry = ""
    if carry:
        merged.append(carry)

    steps = []
    wall_of_text = len(merged) < 3
    for block in merged:
        if not wall_of_text and len(block) <= max_len:
            steps.append(block)
            continue
        cur = ""
        for s in sentences(block):
            cur = f"{cur} {s}".strip()
            if len(cur) >= target:
                steps.append(cur)
                cur = ""
        if cur:
            if steps and len(cur) < 40 and wall_of_text:
                steps[-1] = f"{steps[-1]} {cur}"
            else:
                steps.append(cur)
    return steps


# recipe parsing
def parse_meal(m, allowed):
    # Return a dict ready to insert, or None if the recipe is unusable.
    title = clean_text(m.get("strMeal"))
    image = (m.get("strMealThumb") or "").strip()
    steps = split_steps(m.get("strInstructions"))
    if not (title and image and steps):
        return None

    lines, classes = [], set()
    for i in range(1, 21):
        name = clean_text(m.get(f"strIngredient{i}"))
        if not name:
            continue
        measure = clean_text(m.get(f"strMeasure{i}"))
        kind, cls = classify(name, allowed)
        if kind == "class":
            classes.add(cls)
        lines.append({"text": f"{measure} {name}".strip(), "kind": kind, "class": cls})
    if not classes: 
        return None
    return {
        "external_id": m["idMeal"], "title": title, "category": clean_text(m.get("strCategory")) or None,
        "area": clean_text(m.get("strArea")) or None, "image_url": image,
        "source_url": (m.get("strSource") or "").strip() or None, "steps": steps,
        "n_class": len(classes), "n_other": sum(1 for l in lines if l["kind"] != "class"),
        "lines": lines,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "dataset" / "Clean_Data.csv"), help="CNN label CSV (defines the class list)")
    ap.add_argument("--data", default=str(Path(__file__).parent / "data_cache" / "themealdb.json"),
                    help="where to cache the downloaded dataset")
    ap.add_argument("--database-url", default=None, help="default: $DATABASE_URL")
    ap.add_argument("--reset", action="store_true", help="drop and recreate all tables first")
    args = ap.parse_args()

    classes = load_classes(args.csv)
    allowed = set(classes)
    print(f"{len(classes)} CNN classes")
    meals = fetch_all(args.data)

    parsed, seen = [], set()
    for m in meals:
        r = parse_meal(m, allowed)
        if r and r["title"].lower() not in seen:
            seen.add(r["title"].lower())
            parsed.append(r)
    print(f"{len(parsed)} of {len(meals)} recipes usable")

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

            for r in parsed:
                cur.execute(
                    "INSERT INTO recipes (external_id, title, category, area, image_url, source_url, steps, "
                    "n_class_ingredients, n_other) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                    (r["external_id"], r["title"], r["category"], r["area"], r["image_url"], r["source_url"],
                     r["steps"], r["n_class"], r["n_other"]))
                rid = cur.fetchone()[0]
                execute_values(
                    cur, "INSERT INTO recipe_ingredients (recipe_id, position, display_text, ingredient_id, kind) VALUES %s",
                    [(rid, pos, l["text"], ing_id[l["class"]] if l["class"] else None, l["kind"])
                     for pos, l in enumerate(r["lines"], 1)])
        print(f"inserted {len(parsed)} recipes")
    except psycopg2.Error as e:
        sys.exit(f"database error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
