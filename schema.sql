-- ingredients: one row per CNN class (names match classes in class_mapping.json exactly)
CREATE TABLE IF NOT EXISTS ingredients (
    id   SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- recipes: n_class_ingredients / n_other let the ranking query count missing ingredients
-- without storing every non-CNN ingredient as a row.
--   n_class_ingredients : distinct CNN-class ingredients in the recipe
--   n_other             : other non-pantry ingredients (celery, bacon, ...) = always "missing"
--   other_ingredients   : the raw text of those, for display
-- Pantry staples (salt, oil, spices, water...) are ignored entirely.
CREATE TABLE IF NOT EXISTS recipes (
    id                  SERIAL PRIMARY KEY,
    title               TEXT NOT NULL,
    instructions        TEXT NOT NULL,
    source              TEXT,
    n_class_ingredients INT  NOT NULL,
    n_other             INT  NOT NULL DEFAULT 0,
    other_ingredients   TEXT[] NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS recipe_ingredients (
    recipe_id     INT NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    ingredient_id INT NOT NULL REFERENCES ingredients(id),
    original_text TEXT,
    PRIMARY KEY (recipe_id, ingredient_id)
);

CREATE INDEX IF NOT EXISTS idx_recipe_ingredients_ingredient ON recipe_ingredients (ingredient_id, recipe_id);
