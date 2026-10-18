"use strict";

const $ = (id) => document.getElementById(id);
const MAX_BYTES = 10 * 1024 * 1024;
const PAGE_SIZE = 5;

const state = {
  photo: null,          // File or null
  tags: [],             // typed ingredient strings
  active: [],           // canonical ingredient names behind the current results
  shown: 0,             // recipes rendered so far (next OFFSET)
  searchId: 0,          // guards against out-of-order responses
};

// helpers 

function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

const FALLBACK_IMG = "data:image/svg+xml;utf8," + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="560" viewBox="0 0 800 560">' +
  '<rect width="800" height="560" fill="#ece5d6"/>' +
  '<g fill="none" stroke="#b9ad94" stroke-width="6" stroke-linecap="round">' +
  '<circle cx="400" cy="270" r="74"/><circle cx="400" cy="270" r="48"/>' +
  '<path d="M270 220v100M250 220v40a20 20 0 0 0 40 0v-40M530 220c-16 20-16 60 0 80v40"/></g>' +
  '<text x="400" y="420" text-anchor="middle" font-family="sans-serif" font-size="26" fill="#8d8268">Image unavailable</text></svg>');

function showError(msg) {
  const box = $("error");
  box.textContent = msg || "";
  box.hidden = !msg;
}

// photo input 

function setPhoto(file) {
  showError("");
  if (file && !file.type.startsWith("image/")) return showError("Please choose an image file.");
  if (file && file.size > MAX_BYTES) return showError("That image is too large. The limit is 10 MB.");
  state.photo = file || null;
  const img = $("preview");
  if (file) {
    if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(file);
  } else {
    img.removeAttribute("src");
    $("file").value = "";
  }
  img.hidden = !file;
  $("dz-empty").hidden = !!file;
  $("remove-photo").hidden = !file;
}

$("file").addEventListener("change", (e) => e.target.files[0] && setPhoto(e.target.files[0]));
$("remove-photo").addEventListener("click", () => setPhoto(null));

const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && setPhoto(e.dataTransfer.files[0]));

// ingredient tags 

function renderTags() {
  const list = $("tags");
  list.replaceChildren();
  state.tags.forEach((t, i) => {
    const li = el("li", "tag");
    li.append(el("span", null, t));
    const rm = el("button", null, "×");
    rm.type = "button";
    rm.setAttribute("aria-label", `Remove ${t}`);
    rm.addEventListener("click", () => { state.tags.splice(i, 1); renderTags(); });
    li.append(rm);
    list.append(li);
  });
}

function addTags(text) {
  for (const raw of text.split(/[,;\n]/)) {
    const t = raw.trim().replace(/\s+/g, " ");
    if (t && t.length <= 60 && !state.tags.some((x) => x.toLowerCase() === t.toLowerCase())) state.tags.push(t);
  }
  renderTags();
}

const tagInput = $("tag-input");
tagInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === ",") {
    e.preventDefault();
    if (tagInput.value.trim()) { addTags(tagInput.value); tagInput.value = ""; }
    else if (e.key === "Enter") $("search").click();
  } else if (e.key === "Backspace" && !tagInput.value && state.tags.length) {
    state.tags.pop();
    renderTags();
  }
});
tagInput.addEventListener("paste", (e) => {
  const text = (e.clipboardData || window.clipboardData).getData("text");
  if (/[,;\n]/.test(text)) { e.preventDefault(); addTags(text); }
});
tagInput.addEventListener("blur", () => { if (tagInput.value.trim()) { addTags(tagInput.value); tagInput.value = ""; } });
$("tag-box").addEventListener("click", () => tagInput.focus());

// rendering 

function chipList(items, withConfidence) {
  const wrap = el("div", "chips");
  for (const i of items) {
    const c = el("span", "chip", i.label);
    if (withConfidence) c.append(el("small", null, Math.round(i.confidence * 100) + "%"));
    wrap.append(c);
  }
  return wrap;
}

function renderSummary(data) {
  const box = $("summary");
  box.replaceChildren();
  const section = (title, node) => { const d = el("div"); d.append(el("h3", null, title), node); box.append(d); };

  if (data.detected.length) section("Detected in your photo", chipList(data.detected, true));
  if (data.typed.length) section("Typed by you", chipList(data.typed, false));
  if (data.unrecognized.length) {
    const p = el("p", "note",
      "Not in our ingredient list, so not used for matching: " + data.unrecognized.join(", ") + ".");
    box.append(p);
  }
  if (!data.active_ingredients.length) {
    box.append(el("p", "note", state.photo
      ? "No known food items were detected in the photo. Try a clearer or closer picture, or type ingredients."
      : "None of those ingredients are in our list. Try common ones such as tomato, chicken or cheese."));
  }
  box.hidden = !box.childNodes.length;
}

function ingredientList(ingredients) {
  const ul = el("ul", "ingredients");
  const tagText = { have: "Have", need: "Need" };
  for (const ing of ingredients) {
    const li = el("li", "ing-" + ing.status);
    li.append(el("span", null, ing.text), el("span", "tagtext", tagText[ing.status]));
    ul.append(li);
  }
  return ul;
}

function recipeCard(r, activeCount) {
  const card = el("article", "card");

  const img = el("img", "card-img");
  img.alt = r.title;
  img.loading = "lazy";
  img.referrerPolicy = "no-referrer";
  img.addEventListener("error", () => { img.onerror = null; img.src = FALLBACK_IMG; }, { once: true });
  img.src = r.image_url || FALLBACK_IMG;
  card.append(img);

  const body = el("div", "card-body");
  const meta = [r.category, r.area].filter(Boolean).join("  /  ");
  if (meta) body.append(el("p", "card-meta", meta));
  body.append(el("h3", null, r.title));

  const match = el("div", "match");
  match.append(el("span", "pill pill-match",
    `Matches ${r.matched_count} of your ${activeCount} ingredient${activeCount === 1 ? "" : "s"}`));
  match.append(r.missing_count
    ? el("span", "pill pill-miss", `${r.missing_count} missing`)
    : el("span", "pill pill-ok", "You have everything"));
  body.append(match);

  const cols = el("div", "card-cols");

  const left = el("div");
  left.append(el("h4", null, "Ingredients"));
  const legend = el("div", "legend");
  legend.append(el("span", null, "Have: in your list"), el("span", null, "Need: not in your list"));
  left.append(legend, ingredientList(r.ingredients));

  const right = el("div");
  right.append(el("h4", null, "Method"));
  const ol = el("ol", "steps");
  for (const s of r.steps) ol.append(el("li", null, s));
  right.append(ol);
  if (r.source_url) {
    const p = el("p", "source");
    const a = el("a", null, "Original recipe");
    a.href = r.source_url; a.target = "_blank"; a.rel = "noopener noreferrer";
    p.append(a);
    right.append(p);
  }

  cols.append(left, right);
  body.append(cols);
  card.append(body);
  return card;
}

function appendRecipes(recipes) {
  const frag = document.createDocumentFragment();
  for (const r of recipes) frag.append(recipeCard(r, state.active.length));
  $("recipes").append(frag);
  state.shown += recipes.length;
}

// search 

async function postJson(url, payload) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  return res;
}

async function readError(res) {
  const body = await res.json().catch(() => ({}));
  return typeof body.detail === "string" ? body.detail : `Something went wrong (error ${res.status}).`;
}

async function search() {
  showError("");
  if (tagInput.value.trim()) { addTags(tagInput.value); tagInput.value = ""; }
  if (!state.photo && !state.tags.length) return showError("Upload a photo, type some ingredients, or both.");

  const id = ++state.searchId;
  const btn = $("search");
  btn.disabled = true;
  btn.textContent = state.photo ? "Analyzing your photo..." : "Searching...";

  try {
    const form = new FormData();
    if (state.photo) form.append("file", state.photo);
    form.append("ingredients", state.tags.join(", "));
    const res = await fetch("/api/analyze-fridge", { method: "POST", body: form });
    if (!res.ok) throw new Error(await readError(res));
    const data = await res.json();
    if (id !== state.searchId) return;

    state.active = data.active_ingredients;
    state.shown = 0;
    $("recipes").replaceChildren();
    renderSummary(data);
    appendRecipes(data.recipes);

    const any = data.recipes.length > 0;
    $("recipes-title").hidden = !any;
    const none = $("no-recipes");
    none.hidden = any || !data.active_ingredients.length;
    none.textContent = "No recipes matched those ingredients.";
    $("see-more").hidden = !data.has_more;
    $("results").hidden = false;
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
  } finally {
    if (id === state.searchId) {
      btn.disabled = false;
      btn.textContent = "Find recipes";
    }
  }
}

async function seeMore() {
  const btn = $("see-more");
  const id = state.searchId;
  btn.disabled = true;
  btn.textContent = "Loading...";
  try {
    const res = await postJson("/api/recipes", { ingredients: state.active, offset: state.shown });
    if (!res.ok) throw new Error(await readError(res));
    const data = await res.json();
    if (id !== state.searchId) return;
    appendRecipes(data.recipes);
    btn.hidden = !data.has_more;
  } catch (err) {
    showError(err.message || "Could not load more recipes.");
  } finally {
    btn.disabled = false;
    btn.textContent = "See more";
  }
}

$("search").addEventListener("click", search);
$("see-more").addEventListener("click", seeMore);
