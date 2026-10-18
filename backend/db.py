"""PostgreSQL access: connection helper and the paginated recipe ranking query."""
import os

import psycopg2
import psycopg2.extras

DEFAULT_URL = "postgresql://localhost/fridge_recipes"
PAGE_SIZE = 5

# Rank: most matched ingredients first, then fewest missing ones.
#   matched = distinct recipe ingredients (CNN classes) the user has
#   missing = recipe CNN-class ingredients the user lacks + every other ingredient (n_other),
#             including salt, oil and spices, since the card shows those as NEED too
# Tie-breaks (title, id) make the order total, so OFFSET pages never repeat or skip a recipe.
RANKED_SQL = """
WITH ranked AS (
    SELECT r.id,
           COUNT(DISTINCT ri.ingredient_id)::int AS matched_count,
           (r.n_class_ingredients - COUNT(DISTINCT ri.ingredient_id) + r.n_other)::int AS missing_count
    FROM ingredients i
    JOIN recipe_ingredients ri ON ri.ingredient_id = i.id
    JOIN recipes r             ON r.id = ri.recipe_id
    WHERE i.name = ANY(%(have)s)
    GROUP BY r.id
    ORDER BY matched_count DESC, missing_count ASC, r.title ASC, r.id ASC
    LIMIT %(limit)s OFFSET %(offset)s
)
SELECT r.id, r.title, r.category, r.area, r.image_url, r.source_url, r.steps,
       k.matched_count, k.missing_count,
       (SELECT json_agg(json_build_object('text', ri.display_text, 'kind', ri.kind, 'name', i.name)
                        ORDER BY ri.position)
          FROM recipe_ingredients ri
          LEFT JOIN ingredients i ON i.id = ri.ingredient_id
         WHERE ri.recipe_id = r.id) AS ingredients
FROM ranked k
JOIN recipes r ON r.id = k.id
ORDER BY k.matched_count DESC, k.missing_count ASC, r.title ASC, r.id ASC;
"""


def get_connection(url=None):
    return psycopg2.connect(url or os.environ.get("DATABASE_URL", DEFAULT_URL))


def class_names(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM ingredients")
        return {r[0] for r in cur.fetchall()}


def ranked_recipes(have, offset=0, limit=PAGE_SIZE, conn=None):
    """Return (recipes, has_more) for one page of ranked results.

    Each recipe's `ingredients` is a list of {text, status} where status is
    'have' (user has it) or 'need' (missing).
    """
    have = sorted(set(have))
    if not have:
        return [], False
    own = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # fetch one extra row to learn whether another page exists
            cur.execute(RANKED_SQL, {"have": have, "limit": limit + 1, "offset": offset})
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        if own:
            conn.close()

    has_more = len(rows) > limit
    rows = rows[:limit]
    owned = set(have)
    for r in rows:
        for ing in r["ingredients"]:
            kind = ing.pop("kind")
            name = ing.pop("name")
            ing["status"] = "have" if kind == "class" and name in owned else "need"
    return rows, has_more


if __name__ == "__main__":
    import sys

    recipes, more = ranked_recipes(sys.argv[1:] or ["tomato", "cheese", "sausage"])
    for r in recipes:
        print(r["matched_count"], r["missing_count"], r["title"], f"({len(r['steps'])} steps)")
    print("has_more:", more)
