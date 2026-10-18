-- ingredients: one row per CNN class (names match classes in class_mapping.json exactly).
-- Used for SQL matching and ranking.
CREATE TABLE IF NOT EXISTS ingredients (
    id   SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- recipes: one row per dish.
--   steps               : ordered cooking steps, rendered as a numbered list
--   n_class_ingredients : distinct CNN-class ingredients in the recipe
--   n_other             : ingredient lines outside the CNN classes, including salt/oil/spices
--                         (always "missing" for ranking and display)
CREATE TABLE IF NOT EXISTS recipes (
    id                  SERIAL PRIMARY KEY,
    external_id         TEXT UNIQUE,
    title               TEXT NOT NULL,
    category            TEXT,
    area                TEXT,
    image_url           TEXT NOT NULL,
    source_url          TEXT,
    steps               TEXT[] NOT NULL,
    n_class_ingredients INT NOT NULL,
    n_other             INT NOT NULL DEFAULT 0
);

-- recipe_ingredients: EVERY ingredient line of a recipe, in order.
--   display_text  : full text with measurement, e.g. '1.5 lbs Italian sausage'
--   ingredient_id : the normalized CNN class (NULL unless kind = 'class')
--   kind          : 'class' (matchable), 'staple' (salt, oil, spices...), 'other' (everything else)
CREATE TABLE IF NOT EXISTS recipe_ingredients (
    id            SERIAL PRIMARY KEY,
    recipe_id     INT  NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    position      INT  NOT NULL,
    display_text  TEXT NOT NULL,
    ingredient_id INT  REFERENCES ingredients(id),
    kind          TEXT NOT NULL CHECK (kind IN ('class', 'staple', 'other'))
);

CREATE INDEX IF NOT EXISTS idx_ri_ingredient ON recipe_ingredients (ingredient_id, recipe_id) WHERE ingredient_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ri_recipe ON recipe_ingredients (recipe_id, position);
