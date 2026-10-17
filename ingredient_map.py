"""Map free-text recipe ingredients (e.g. "boneless skinless chicken breasts") onto the
standardized classes the CNN predicts (e.g. "chicken").

classify(text) -> ("class", name) | ("staple", None) | ("other", None)
  class  : one of the CNN classes
  staple : pantry basics (salt, oil, spices...) ignored when counting missing ingredients
  other  : anything else; counts as a missing ingredient

Within one ingredient string the class whose match ends furthest right wins (the head
noun is usually last: "grape tomatoes" -> tomato, "cream cheese" -> cheese), with ties
going to the longer match ("sweet potato" beats "potato").
"""
import re

# Words that mean the item is a processed/derived product, not the fresh food itself.
STOP = re.compile(
    r"\b(soup|broth|stock|bouillon|extract|flavou?ring|seasoning|powder|salt|spice|"
    r"dressing|liqueur|wine|vinegar|jam|jelly|cereal|cookies?|crackers?|crumbs?|pudding|"
    r"ice cream|syrup|flakes|cayenne|crushed|lemon pepper|marmalade)\b"
)

STAPLE = re.compile(
    r"\b(water|ice|salt|pepper|oil|cooking spray|baking soda|baking powder|vanilla|cinnamon|"
    r"nutmeg|paprika|cumin|oregano|basil|thyme|parsley|cilantro|rosemary|sage|dill|bay leaf|"
    r"vinegar|spices?|seasoning|cayenne|mustard|ketchup|mayonnaise|soy sauce|worcestershire|"
    r"honey|cornstarch|corn starch|yeast|broth|stock|bouillon|extract|powder|flakes|cloves?|"
    r"turmeric|curry|allspice|cardamom|chives?|mint)\b"
)

# class -> (include regex, optional exclude regex). Order is irrelevant (see module docstring).
RULES = {
    "asparagus": (r"asparagus", None),
    "avocado": (r"avocados?", None),
    "banana": (r"bananas?", None),
    "bread": (r"bread|baguettes?|ciabatta|sourdough|loaf|buns?|rolls?", r"bread ?crumbs?|egg roll|spring roll"),
    "broccoli": (r"broccoli|broccolini", None),
    "butter": (r"butter", r"peanut butter|almond butter|nut butter|cocoa butter|apple butter|butternut|butterscotch|shea butter"),
    "cauliflower": (r"cauliflowers?", None),
    "chocolate": (r"chocolate", None),
    "corn": (r"corn", r"corn ?(starch|syrup|meal|flour|oil|flakes|bread|tortillas?|chips|muffin)|cornish"),
    "flour": (r"flour", r"corn ?flour"),
    "garlic": (r"garlic", None),
    "grape": (r"grapes?", r"grape (juice|jelly|tomatoes)"),
    "ham": (r"ham", None),
    "kiwi": (r"kiwis?|kiwi ?fruits?", None),
    "leek": (r"leeks?", None),
    "lemon": (r"lemons?", None),
    "lettuce": (r"lettuce|romaine|iceberg", None),
    "lime": (r"limes?", r"lime juice concentrate"),
    "mango": (r"mangos?|mangoes", None),
    "milk": (r"milk", r"(coconut|almond|soy|rice|oat|condensed|evaporated|powdered|dry|nonfat dry) milk"),
    "onion": (r"onions?|scallions?|shallots?", None),
    "pea": (r"peas?|snow peas?|snap peas?", r"chick ?peas?|black[- ]eyed peas?|split peas?|pigeon peas?"),
    "peach": (r"peach(es)?", None),
    "pear": (r"pears?", None),
    "pineapple": (r"pineapples?", None),
    "potato": (r"potato(es)?", r"potato (chips|starch|flakes)|sweet potato"),
    "radish": (r"radish(es)?", None),
    "sausage": (r"sausages?|chorizo|kielbasa|bratwurst|andouille", None),
    "spinach": (r"spinach", None),
    "watermelon": (r"watermelons?", None),
    "yogurt": (r"yogh?urts?", None),
    "zucchini": (r"zucchinis?|courgettes?", None),
    "heavy_cream": (r"(heavy|whipping|double) cream|heavy whipping cream", None),
    "shrimp": (r"shrimps?|prawns?", None),
    "sugar": (r"sugar", r"sugar snap|sugar substitute|sugar free"),
    "bitter_gourd": (r"bitter (gourd|melon)|karela", None),
    "turnip": (r"turnips?", None),
    "apple": (r"apples?", r"apple (juice|cider|butter|jelly|pie filling)|pineapple"),
    "beans": (r"beans?", r"vanilla beans?|coffee beans?|cocoa beans?|jelly beans?|bean sprouts?|beans? paste|beans? thread"),
    "beef": (r"beef|steaks?|brisket|sirloin", r"beef (broth|stock|bouillon)"),
    "blueberry": (r"blueberr(y|ies)", None),
    "cabbage": (r"cabbages?", None),
    "carrot": (r"carrots?", None),
    "chicken": (r"chickens?|chicken breasts?|chicken thighs?|drumsticks?", None),
    "chilli": (r"chil+(i|e|y)(es|s)?|chilli?es|jalape[nñ]os?|serranos?|habaneros?|poblanos?", r"chil+[iey]s? (sauce|paste|oil|powder|flakes|beans|con carne)"),
    "cucumber": (r"cucumbers?", None),
    "eggs": (r"eggs?|egg yolks?|egg whites?", r"egg noodles?|egg roll|egg wash"),
    "eggplant": (r"eggplants?|aubergines?", None),
    "ginger": (r"ginger", r"ground ginger|ginger ale|ginger ?snaps?|crystallized|candied|ginger beer"),
    "mushroom": (r"mushrooms?|shiitake|portobello|portabella|cremini|champignons?", None),
    "orange": (r"oranges?", r"orange (juice|zest|peel|marmalade|roughy|liqueur)"),
    "pepper": (r"bell peppers?|capsicum|(green|red|yellow|orange|sweet) (bell )?peppers?", None),
    "strawberry": (r"strawberr(y|ies)", None),
    "sweet_potato": (r"sweet potato(es)?|yams?", None),
    "tomato": (r"tomato(es)?", r"sun[- ]?dried"),
    "cheese": (r"cheeses?|cheddar|mozzarella|parmesan|parmigiano|feta|ricotta|gouda|brie|provolone|mascarpone|gruyere|swiss cheese", r"cheese ?cake|cheese soup"),
    "chayote": (r"chayotes?", None),
}
_COMPILED = {
    name: (re.compile(rf"\b({inc})\b"), re.compile(rf"\b({exc})\b") if exc else None)
    for name, (inc, exc) in RULES.items()
}


def classify(text, allowed=None):
    """Classify one ingredient string. `allowed` optionally restricts to a set of class names."""
    t = re.sub(r"[^a-z0-9ñ\- ]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    best = None  # (end, span_length, name)
    if not STOP.search(t):
        for name, (inc, exc) in _COMPILED.items():
            if allowed is not None and name not in allowed:
                continue
            for m in inc.finditer(t):
                if exc and exc.search(t):
                    continue
                cand = (m.end(), m.end() - m.start(), name)
                if best is None or cand[:2] > best[:2]:
                    best = cand
    if best:
        return "class", best[2]
    if STAPLE.search(t):
        return "staple", None
    return "other", None
