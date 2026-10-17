"""PostgreSQL access: connection helper and the top-5 recipe ranking query."""
import os

import psycopg2
import psycopg2.extras

DEFAULT_URL = "postgresql://localhost/fridge_recipes"

# Rank: most detected ingredients matched first, then fewest missing ones.
#   matched_count = recipe ingredients (CNN classes) the user has
#   missing_count = recipe CNN-class ingredients the user lacks + non-pantry "other" ingredients
# Tie-breaks (title, id) keep results deterministic.
TOP_RECIPES_SQL = """
WITH top AS (
    SELECT r.id,
           COUNT(*)::int AS matched_count,
           (r.n_class_ingredients - COUNT(*) + r.n_other)::int AS missing_count
    FROM ingredients i
    JOIN recipe_ingredients ri ON ri.ingredient_id = i.id
    JOIN recipes r             ON r.id = ri.recipe_id
    WHERE i.name = ANY(%(detected)s)
    GROUP BY r.id
    ORDER BY matched_count DESC, missing_count ASC, r.title ASC, r.id ASC
    LIMIT 5
)
SELECT r.id, r.title, r.instructions, t.matched_count, t.missing_count,
       ARRAY(SELECT i.name FROM recipe_ingredients ri JOIN ingredients i ON i.id = ri.ingredient_id
             WHERE ri.recipe_id = r.id AND i.name = ANY(%(detected)s) ORDER BY i.name) AS matched_ingredients,
       ARRAY(SELECT i.name FROM recipe_ingredients ri JOIN ingredients i ON i.id = ri.ingredient_id
             WHERE ri.recipe_id = r.id AND i.name <> ALL(%(detected)s) ORDER BY i.name)
         || r.other_ingredients AS missing_ingredients
FROM top t
JOIN recipes r ON r.id = t.id
ORDER BY t.matched_count DESC, t.missing_count ASC, r.title ASC, r.id ASC;
"""


def get_connection(url=None):
    return psycopg2.connect(url or os.environ.get("DATABASE_URL", DEFAULT_URL))


def top_recipes(detected, conn=None):
    """Return up to 5 recipes (dicts) for a list of detected class names, best match first."""
    detected = sorted(set(detected))
    if not detected:
        return []
    own = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(TOP_RECIPES_SQL, {"detected": detected})
            return [dict(r) for r in cur.fetchall()]
    finally:
        if own:
            conn.close()


if __name__ == "__main__":
    import json
    import sys

    for r in top_recipes(sys.argv[1:] or ["tomato", "cheese", "sausage"]):
        r["instructions"] = r["instructions"][:80] + "..."
        print(json.dumps(r, indent=2))
