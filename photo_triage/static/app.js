/* photo-triage: the whole client.
 *
 * One module rather than several: the grid, the selection and the actions all
 * read the same few pieces of state, and splitting them would mean an
 * interface between halves that can only be understood together.
 *
 * The two things worth knowing before reading:
 *
 * 1. The grid is windowed. `state.tiles` holds every loaded result, but only a
 *    viewport plus two screens of overscan exists in the DOM, and nodes are
 *    recycled as it scrolls. DESIGN.md calls 60fps at 24,000 rows a release
 *    gate, and this is the reason it holds. Two rules keep that true, and both
 *    are easy to break by accident: a tile that leaves the window is *parked*
 *    rather than discarded, so scrolling costs no allocation and no thumbnail
 *    re-decode; and a tile's position is written only when it can actually
 *    have moved. Scrolling never moves a tile -- the grid is in page
 *    coordinates -- so a scroll frame writes no geometry at all.
 * 2. Every mutation is optimistic. A triage session is thousands of decisions
 *    and a 40ms round-trip on each is ninety seconds of dead time. The DOM
 *    changes immediately and reconciles with the response; a refusal reverts
 *    the tile and says why in the selection bar, never in a toast.
 */

const GAP = 4;
const PAD_X = 8;
const PAD_TOP = 96;      /* toolbar height plus --s4; see style.css */
const FLOOR = { S: 96, M: 132, L: 168 };
const PAGE = 300;
const OVERSCAN_SCREENS = 2;
const DEBOUNCE_MS = 120;
/* Parked tiles kept for reuse. Roughly one window's worth: enough that a
   reversal of scroll direction never allocates, small enough that the images
   they still hold cannot accumulate. */
const POOL_MAX = 240;

const $ = (id) => document.getElementById(id);

const state = {
  tiles: [],           // every result loaded so far, in rank order
  matched: 0,          // how many match the filters in total
  like: null,          // row id being used as the query, for "more like this"
  lightboxOrigin: null,// the tile to return focus to when the lightbox closes
  selection: new Set(),
  extended: false,     // selection reaches beyond what has been loaded
  hovered: null,       // row id under the cursor; S and F target it
  anchor: null,        // index of the last plain click, for shift+click ranges
  stats: null,
  fetching: false,
  armed: false,        // the quarantine button is showing its confirmation
  layout: { cols: 1, cell: 0, rowHeight: 0 },
  geometry: 0,         // bumped whenever laid-out positions become stale
  nodes: new Map(),    // tile index -> live DOM node
  pool: [],            // detached tiles, ready to be refilled
};

/* ── talking to the server ─────────────────────────────────────────────── */

const queryString = () => {
  const params = new URLSearchParams({
    q: $('search').value.trim(),
    group: $('f-group').value,
    cat: $('f-cat').value,
    folder: $('f-folder').value,
    show: $('f-show').value,
    view: $('f-view').value,
  });
  if (state.like != null) params.set('like', state.like);
  return params;
};

const getJSON = (url) => fetch(url).then((r) => r.json());

const postJSON = (url, body) =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((r) => r.json());

/* ── loading results ───────────────────────────────────────────────────── */

/* Runs on every filter change and on a debounced keystroke. Replaces the
   result set, drops the selection, and re-lays-out from scroll position zero. */
async function reload() {
  state.tiles = [];
  state.matched = 0;
  clearSelection();
  const params = queryString();
  params.set('limit', PAGE);
  params.set('offset', 0);
  const page = await getJSON(`/api/search?${params}`);
  state.tiles = page.results;
  state.matched = page.matched;
  window.scrollTo(0, 0);
  /* reflow, not layout: a node is only built for a grid index that has none,
     so a result set that replaces another must drop the old nodes first or
     index 0 keeps showing the previous query's first image. */
  reflow();
}

/* Called when the window nears the end of what is loaded. Appends; never
   reorders, because the ranking the user is scrolling through must not move
   under them. */
async function loadMore() {
  if (state.fetching || state.tiles.length >= state.matched) return;
  state.fetching = true;
  const params = queryString();
  params.set('limit', PAGE);
  params.set('offset', state.tiles.length);
  const page = await getJSON(`/api/search?${params}`);
  state.tiles.push(...page.results);
  state.matched = page.matched;
  state.fetching = false;
  layout();
}

/* ── grid geometry and windowing ───────────────────────────────────────── */

/* Recomputes column count and cell size from the viewport, then repaints the
   window. Called on resize, density change and every result change: it is the
   only place that decides where a tile goes. */
function layout() {
  const width = document.documentElement.clientWidth;
  const floor = FLOOR[document.body.dataset.density];
  const cols = Math.max(1, Math.floor((width - 2 * PAD_X + GAP) / (floor + GAP)));
  const cell = (width - 2 * PAD_X - GAP * (cols - 1)) / cols;
  state.layout = { cols, cell, rowHeight: cell + GAP };
  state.geometry += 1;   /* every placed tile is now in the wrong place */
  const rows = Math.ceil(state.tiles.length / cols);
  $('grid').style.height = `${PAD_TOP + rows * state.layout.rowHeight + PAD_X}px`;
  paint();
  renderEmptyState();
}

/* Adds, moves and removes tile nodes so that exactly the visible range plus
   the overscan exists in the DOM. Called from a rAF on scroll. */
function paint() {
  const { cols, cell, rowHeight } = state.layout;
  const viewport = window.innerHeight;
  const overscan = viewport * OVERSCAN_SCREENS;
  const firstRow = Math.max(0, Math.floor((window.scrollY - PAD_TOP - overscan) / rowHeight));
  const lastRow = Math.ceil((window.scrollY - PAD_TOP + viewport + overscan) / rowHeight);
  const from = firstRow * cols;
  const to = Math.min(state.tiles.length, (lastRow + 1) * cols);

  for (const [index, node] of state.nodes) {
    if (index < from || index >= to) {
      park(index, node);
    }
  }
  const fragment = document.createDocumentFragment();
  for (let index = from; index < to; index += 1) {
    let node = state.nodes.get(index);
    if (!node) {
      node = state.pool.pop() || blankTile();
      fillTile(node, state.tiles[index], index);
      state.nodes.set(index, node);
      fragment.appendChild(node);
    }
    /* A scroll does not move anything: tiles are positioned in page
       coordinates, so their left/top only go stale when the geometry itself
       changes. Writing them every frame anyway is a style recalculation and a
       layout for every node in the window, sixty times a second, for values
       that are already correct. */
    if (node.geometry !== state.geometry) {
      node.geometry = state.geometry;
      node.style.width = `${cell}px`;
      node.style.height = `${cell}px`;
      node.style.left = `${PAD_X + (index % cols) * (cell + GAP)}px`;
      node.style.top = `${PAD_TOP + Math.floor(index / cols) * rowHeight}px`;
    }
  }
  $('grid').appendChild(fragment);

  /* paint() runs on every scroll frame, so nothing else belongs in it, the
     empty state is repainted by layout(), which runs only when the data or
     the geometry actually changed. */
  if (to > state.tiles.length - PAGE) loadMore();
}

/* Detach a tile and keep it for the next index that needs one. Scrolling a
   long grid otherwise builds and throws away five nodes and five <img>
   decodes per row, forever; a parked tile costs one property write to refill
   and keeps a thumbnail the user is quite likely to scroll back to. */
function park(index, node) {
  state.nodes.delete(index);
  node.remove();
  if (state.pool.length < POOL_MAX) state.pool.push(node);
}

/* An empty tile: every element a tile can ever need, built once. The clock is
   built with the rest and hidden rather than added and removed, because a tile
   parked from a video and refilled with a photograph must not carry a stale
   one, and "hidden" is a cheaper way to say that than a subtree edit. */
function blankTile() {
  const node = document.createElement('button');
  node.className = 'tile';
  node.setAttribute('role', 'gridcell');
  node.tabIndex = -1;

  const img = document.createElement('img');
  img.loading = 'lazy';
  img.decoding = 'async';
  img.alt = '';
  /* Not {once: true}: this node outlives the image in it. */
  img.addEventListener('load', () => img.classList.add('ready'));

  const vignette = document.createElement('div');
  vignette.className = 'vignette';

  const mark = document.createElement('span');
  mark.className = 'mark';

  const strip = document.createElement('span');
  strip.className = 'strip';

  /* A still frame of a video looks exactly like a photograph, so the length is
     the label: it says both "this moves" and how long it runs. */
  const clock = document.createElement('span');
  clock.className = 'clock';

  node.append(img, vignette, mark, strip, clock);
  node.parts = { img, mark, strip, clock };
  return node;
}

/* Point a tile at a result. Category is carried by the dot for scanning and by
   the label in words for everyone the dot does not work for (DESIGN.md §9). */
function fillTile(node, tile, index) {
  const { img, mark, strip, clock } = node.parts;
  node.dataset.id = tile.id;
  node.dataset.index = index;
  node.dataset.saved = tile.saved;
  node.dataset.quarantined = tile.quarantined;
  node.dataset.video = tile.duration > 0;
  delete node.dataset.leaving;
  node.setAttribute('aria-selected', state.selection.has(tile.id));
  node.setAttribute('aria-label', describe(tile));
  node.geometry = -1;   /* a refilled tile has not been placed at its index */

  const src = `/thumb/${tile.id}`;
  if (!img.src.endsWith(src)) {
    img.classList.remove('ready');
    img.src = src;
    /* Already in the browser's cache: no load event is coming, and without
       this the tile fades in from a permanent opacity 0. */
    if (img.complete && img.naturalWidth) img.classList.add('ready');
  }
  mark.dataset.group = tile.group;
  strip.textContent = stripText(tile);
  clock.hidden = !(tile.duration > 0);
  clock.textContent = tile.duration > 0 ? clockText(tile.duration) : '';
  return node;
}

const percent = (value) => `${Math.round(value * 100)}%`;

const clockText = (seconds) => {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
};

const describe = (tile) =>
  `${tile.name}, ${tile.duration > 0 ? 'video, ' : ''}${tile.category || 'unclassified'}` +
  (tile.saved ? ', saved' : '') +
  (tile.quarantined ? ', quarantined' : '');

const stripText = (tile) =>
  [tile.score != null ? `${tile.score.toFixed(2)}` : '',
   tile.confidence ? percent(tile.confidence) : ''].filter(Boolean).join(' · ');

/* ── selection ─────────────────────────────────────────────────────────── */

function setSelected(id, on) {
  if (on) state.selection.add(id); else state.selection.delete(id);
  for (const node of state.nodes.values()) {
    if (Number(node.dataset.id) === id) node.setAttribute('aria-selected', on);
  }
}

function clearSelection() {
  state.selection.clear();
  state.extended = false;
  state.anchor = null;
  for (const node of state.nodes.values()) node.setAttribute('aria-selected', 'false');
  renderSelectionBar();
}

function selectShown() {
  for (const tile of state.tiles) state.selection.add(tile.id);
  for (const node of state.nodes.values()) node.setAttribute('aria-selected', 'true');
  renderSelectionBar();
}

/* The escalation from "everything loaded" to "everything matching". Offered
   only once the user has shown intent by pressing A, never as a second button
   sitting next to the first, the two act on wildly different numbers. */
async function selectAllMatching() {
  const { ids } = await getJSON(`/api/ids?${queryString()}`);
  state.selection = new Set(ids);
  state.extended = true;
  for (const node of state.nodes.values()) node.setAttribute('aria-selected', 'true');
  renderSelectionBar();
}

function selectRange(toIndex) {
  const from = state.anchor == null ? toIndex : state.anchor;
  const [low, high] = from < toIndex ? [from, toIndex] : [toIndex, from];
  for (let index = low; index <= high; index += 1) {
    setSelected(state.tiles[index].id, true);
  }
  renderSelectionBar();
}

/* ── actions ───────────────────────────────────────────────────────────── */

const chosen = () => [...state.selection];

/* Targets the selection, or the tile under the cursor when there is none, so
   the user can sweep through hovering and tapping without ever selecting. */
const targets = () =>
  state.selection.size ? chosen() : state.hovered != null ? [state.hovered] : [];

async function quarantine() {
  const ids = targets();
  if (!ids.length) return;
  const gate = reviewGate();
  if (gate) { openReview(gate); return; }
  if (!state.armed) { armDelete(true); return; }
  armDelete(false);

  const removed = new Set(ids);
  animateOut(removed);
  const result = await postJSON('/api/quarantine', { ids });
  const moved = new Set(result.moved);
  state.tiles = state.tiles.filter((tile) => !moved.has(tile.id));
  state.matched -= moved.size;
  clearSelection();
  reflow();
  showRefusals(result.refused, 'protected images were kept');
  refreshStats();
}

async function restore() {
  const ids = targets();
  if (!ids.length) return;
  const result = await postJSON('/api/restore', { ids });
  const moved = new Set(result.moved);
  if ($('f-view').value === 'quarantine') {
    state.tiles = state.tiles.filter((tile) => !moved.has(tile.id));
    state.matched -= moved.size;
  }
  clearSelection();
  reflow();
  showRefusals(result.refused, 'some files could not be put back');
  refreshStats();
}

async function toggleSaved() {
  const ids = targets();
  if (!ids.length) return;
  const anyUnsaved = state.tiles.some((t) => ids.includes(t.id) && !t.saved);
  const result = await postJSON('/api/save', { ids, protected: anyUnsaved });
  const moved = new Set(result.moved);
  for (const tile of state.tiles) if (moved.has(tile.id)) tile.saved = anyUnsaved;
  /* Protected images drop out of the default view: that is what makes the
     remaining pile shrink with every pass. */
  if ($('f-show').value === 'unsaved' && anyUnsaved) {
    state.tiles = state.tiles.filter((tile) => !moved.has(tile.id));
    state.matched -= moved.size;
  }
  clearSelection();
  reflow();
  refreshStats();
}

/* Undo restores the most recent quarantine batch. Restoring an *older* batch
   is the same call with different ids, which is what the quarantine view's
   own selection provides (PROJECT.md §9.3). */
async function undo() {
  const { batches } = await getJSON('/api/batches');
  if (!batches.length) return;
  await postJSON('/api/restore', { ids: batches[batches.length - 1].ids });
  reload();
  refreshStats();
}

function findSimilar(id) {
  state.like = id;
  $('search').value = '';
  reload();
}

function animateOut(ids) {
  for (const node of state.nodes.values()) {
    if (ids.has(Number(node.dataset.id))) node.dataset.leaving = 'true';
  }
}

/* Rebuild the window against the current tiles array. Every node is refilled
   rather than moved, because a removal shifts the index of everything after
   it and a node's index is the only thing tying it to a result. */
function reflow() {
  for (const [index, node] of state.nodes) park(index, node);
  layout();
  renderCounts();
}

/* ── the review gate ───────────────────────────────────────────────────── */

/* Returns the category that must be reviewed before this bulk action, or null.
   Only bites when the action reaches past the viewport into a whole category , 
   sweeping a handful of tiles by hand was never the dangerous case. */
function reviewGate() {
  const category = $('f-cat').value;
  if (!category || !state.extended || !state.stats) return null;
  return state.stats.reviewed.includes(category) ? null : category;
}

async function openReview(category) {
  document.body.dataset.view = 'review';
  const params = new URLSearchParams({
    cat: category, show: 'all', view: 'active', order: 'confidence', limit: 1000,
  });
  const page = await getJSON(`/api/search?${params}`);
  $('review-title').textContent = category;
  $('review-sub').textContent =
    `${page.matched.toLocaleString()} images, most confident first`;

  const sheet = $('review-sheet');
  sheet.textContent = '';
  let ruleDrawn = false;
  for (const tile of page.results) {
    if (!ruleDrawn && tile.confidence < state.stats.guessing_below) {
      const rule = document.createElement('div');
      rule.className = 'rule';
      rule.textContent = 'below here, the model is guessing';
      sheet.appendChild(rule);
      ruleDrawn = true;
    }
    sheet.appendChild(buildCard(tile));
  }
  await postJSON('/api/reviewed', { category });
  refreshStats();
}

function buildCard(tile) {
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.id = tile.id;
  card.setAttribute('role', 'gridcell');
  card.setAttribute('aria-selected', state.selection.has(tile.id));
  card.setAttribute('aria-label', describe(tile));

  const img = document.createElement('img');
  img.loading = 'lazy';
  img.decoding = 'async';
  img.alt = '';
  img.src = `/thumb/${tile.id}`;

  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.style.width = percent(tile.confidence);
  bar.style.background = `var(--${tile.group})`;

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.innerHTML =
    `<b></b><br><span></span>`;
  meta.querySelector('b').textContent = tile.name;
  meta.querySelector('span').textContent =
    `${tile.category} ${percent(tile.confidence)} · ${tile.width}×${tile.height}` +
    (tile.folder ? ` · ${tile.folder}` : '');

  card.append(img, bar, meta);
  return card;
}

function closeReview() {
  document.body.dataset.view = 'grid';
  layout();
}

/* ── rendering the chrome ──────────────────────────────────────────────── */

function renderCounts() {
  const stats = state.stats;
  const parts = [
    `${state.matched.toLocaleString()} match`,
    `showing ${state.tiles.length.toLocaleString()}`,
  ];
  if (stats) {
    parts.push(`${stats.active.toLocaleString()} on disk`);
    parts.push(`${stats.saved.toLocaleString()} saved`);
    if (stats.quarantined) parts.push(`${stats.quarantined.toLocaleString()} quarantined`);
  }
  $('counts').textContent = parts.join(' · ');

  const category = $('f-cat').value;
  const gateable = category && stats && !stats.reviewed.includes(category);
  const button = $('btn-review');
  button.hidden = !category;
  button.textContent = gateable
    ? `Review ${(stats.categories[category]?.count ?? 0).toLocaleString()} first`
    : `Review ${category}`;
}

function renderSelectionBar() {
  const count = state.selection.size;
  const bar = $('selbar');
  bar.dataset.open = count > 0;
  if (!count) { armDelete(false); return; }

  const label = state.extended
    ? `all ${count.toLocaleString()} matching`
    : `${count.toLocaleString()} selected`;
  $('selcount').textContent = label;
  $('selcount').dataset.extended = state.extended;

  const canEscalate = !state.extended && state.matched > state.selection.size;
  const escalate = $('escalate');
  escalate.hidden = !canEscalate;
  escalate.textContent = `select all ${state.matched.toLocaleString()} matching`;

  const quarantineView = $('f-view').value === 'quarantine';
  $('btn-restore').hidden = !quarantineView;
  $('btn-delete').hidden = quarantineView;

  const gate = reviewGate();
  $('btn-delete').textContent = gate
    ? `Review ${(state.stats.categories[gate]?.count ?? 0).toLocaleString()} first`
    : state.armed
      ? `Quarantine ${count.toLocaleString()}?`
      : 'Quarantine';
}

function armDelete(on) {
  state.armed = on;
  $('btn-delete').dataset.armed = on;
  if (state.selection.size) renderSelectionBar();
}

function showRefusals(refused, summary) {
  const count = Object.keys(refused || {}).length;
  const message = $('selmsg');
  message.hidden = !count;
  if (count) {
    message.textContent = `${count} kept, ${summary}`;
    $('selbar').dataset.open = 'true';
    setTimeout(() => { message.hidden = true; renderSelectionBar(); }, 4000);
  }
}

/* An empty grid has several quite different causes, and saying the wrong one
   is worse than saying nothing: "2,225 images ready" under a filter that just
   matched none of them reads as a bug in the counts. Each branch names the
   reason this particular grid is empty and what would fill it. */
function renderEmptyState() {
  const box = $('empty');
  if (state.tiles.length) { box.dataset.open = 'false'; return; }
  box.dataset.open = 'true';

  const stats = state.stats;
  const searching = $('search').value.trim() || state.like != null;
  const filtered = Boolean(
    $('f-group').value || $('f-cat').value || $('f-folder').value
    || $('f-show').value !== 'unsaved',
  );

  if (!stats || !stats.total) {
    box.innerHTML = '<b>Nothing indexed yet</b><p>The scan is still running. ' +
      'Results appear as soon as the first images are embedded.</p>';
  } else if ($('f-view').value === 'quarantine' && !searching && !filtered) {
    box.innerHTML = '<b>The quarantine is empty</b>' +
      '<p>Anything you quarantine lands here, and can be put back from here.</p>';
  } else if (searching) {
    box.innerHTML = '<b>Nothing matches that</b><p>Search is ranked rather ' +
      'than filtered, so a broader phrase will always return something.</p>';
  } else if (filtered) {
    box.innerHTML = '<b>Nothing left under these filters</b>' +
      `<p class="num">${stats.active.toLocaleString()} images are still in ` +
      'the folder. Widen the filters to see them.</p>';
  } else if (stats.active === 0) {
    box.innerHTML = '<b>Nothing left in this folder</b>' +
      `<p class="num">${stats.saved.toLocaleString()} saved, ` +
      `${stats.quarantined.toLocaleString()} quarantined.</p>`;
  } else {
    box.innerHTML = `<b class="num">${stats.total.toLocaleString()} images ready</b>` +
      '<p>Try <code>screenshot</code>, <code>receipt</code> or ' +
      '<code>people at a table</code>. It searches what the picture looks ' +
      'like, not the filename.</p>';
  }
}

async function refreshStats() {
  state.stats = await getJSON('/api/stats');
  fillOptions($('f-cat'), Object.keys(state.stats.categories).sort(), 'all categories');
  fillOptions($('f-folder'), state.stats.folders.filter(Boolean), 'all folders');
  renderCounts();
  renderEmptyState();
}

function fillOptions(select, values, placeholder) {
  const chosenValue = select.value;
  select.textContent = '';
  select.appendChild(new Option(placeholder, ''));
  for (const value of values) select.appendChild(new Option(value, value));
  select.value = values.includes(chosenValue) ? chosenValue : '';
}

/* ── the embed pass, watched from the page ─────────────────────────────── */

/* A 20-minute pass must not look frozen. Polls until the build stops, then
   reloads once so the newly embedded images appear. */
async function watchProgress() {
  const bar = $('progress');
  let wasRunning = false;
  for (;;) {
    const progress = await getJSON('/api/progress');
    bar.hidden = !progress.running;
    if (progress.running && progress.total) {
      bar.style.width = `${(progress.done / progress.total) * 100}%`;
      $('counts').title =
        `${progress.stage} ${progress.done}/${progress.total} on ${progress.device}`;
    }
    if (wasRunning && !progress.running) { await refreshStats(); await reload(); }
    wasRunning = progress.running;
    await new Promise((done) => setTimeout(done, progress.running ? 700 : 3000));
  }
}

/* ── events ────────────────────────────────────────────────────────────── */

/* Invoked by the browser on every click inside the grid, including on the
   images inside tiles, hence the closest() lookup. */
function onGridClick(event) {
  const node = event.target.closest('.tile');
  if (!node) return;
  const index = Number(node.dataset.index);
  const id = Number(node.dataset.id);
  if (event.shiftKey) { selectRange(index); return; }
  const nowSelected = !state.selection.has(id);
  setSelected(id, nowSelected);
  state.anchor = index;
  state.extended = false;
  renderSelectionBar();
}

function onGridDblClick(event) {
  const node = event.target.closest('.tile');
  if (node) openLightbox(Number(node.dataset.id));
}

function onGridPointerOver(event) {
  const node = event.target.closest('.tile');
  state.hovered = node ? Number(node.dataset.id) : null;
}

function openLightbox(id) {
  const tile = state.tiles.find((candidate) => candidate.id === id);
  if (!tile) return;
  state.lightboxOrigin = document.activeElement;

  /* Same endpoint either way: /full serves the original bytes, and a browser
     that can play the container will. */
  const playing = tile.duration > 0;
  $('lightbox-img').hidden = playing;
  $('lightbox-video').hidden = !playing;
  if (playing) {
    $('lightbox-video').src = `/full/${id}`;
    $('lightbox-video').play().catch(() => {});
  } else {
    $('lightbox-img').src = `/full/${id}`;
  }

  $('lightbox-name').textContent = tile.name;
  $('lightbox-meta').textContent =
    `${tile.duration > 0 ? `${clockText(tile.duration)} · ` : ''}` +
    `${tile.category || 'unclassified'} ${percent(tile.confidence)} · ` +
    `${tile.width}×${tile.height}${tile.folder ? ` · ${tile.folder}` : ''}`;
  $('lightbox').dataset.open = 'true';
  $('lightbox').focus();
}

function closeLightbox() {
  $('lightbox').dataset.open = 'false';
  $('lightbox-img').removeAttribute('src');
  const player = $('lightbox-video');
  player.pause();
  player.removeAttribute('src');
  player.load();
  if (state.lightboxOrigin) state.lightboxOrigin.focus();
}

/* Invoked by the browser for every keystroke on the document. Ignores keys
   typed into a field, and any key held with a modifier, otherwise Ctrl+A
   would mean "select all shown" as well as "select all text". */
function onKeyDown(event) {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const tag = event.target.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') {
    if (event.key === 'Enter' && event.target.id === 'search') {
      state.like = null;
      reload();
    }
    if (event.key === 'Escape') event.target.blur();
    return;
  }
  const actions = {
    a: selectShown,
    d: quarantine,
    s: toggleSaved,
    c: clearSelection,
    u: undo,
    r: restore,
    f: () => state.hovered != null && findSimilar(state.hovered),
    '?': () => { const help = $('help'); help.dataset.open = help.dataset.open !== 'true'; },
    escape: () => {
      if ($('lightbox').dataset.open === 'true') closeLightbox();
      else if (document.body.dataset.view === 'review') closeReview();
      else if ($('help').dataset.open === 'true') $('help').dataset.open = 'false';
      else clearSelection();
    },
  };
  const action = actions[event.key.toLowerCase()];
  if (action) { event.preventDefault(); action(); }
}

let debounce;
function onSearchInput() {
  clearTimeout(debounce);
  debounce = setTimeout(() => { state.like = null; reload(); }, DEBOUNCE_MS);
}

let painting = false;
function onScroll() {
  if (painting) return;
  painting = true;
  requestAnimationFrame(() => { painting = false; paint(); });
}

function start() {
  $('grid').addEventListener('click', onGridClick);
  $('grid').addEventListener('dblclick', onGridDblClick);
  $('grid').addEventListener('pointerover', onGridPointerOver);
  $('review-sheet').addEventListener('click', (event) => {
    const card = event.target.closest('.card');
    if (card) openLightbox(Number(card.dataset.id));
  });
  $('review-done').addEventListener('click', closeReview);
  $('btn-review').addEventListener('click', () => openReview($('f-cat').value));
  $('btn-save').addEventListener('click', toggleSaved);
  $('btn-delete').addEventListener('click', quarantine);
  $('btn-restore').addEventListener('click', restore);
  $('btn-clear').addEventListener('click', clearSelection);
  $('escalate').addEventListener('click', selectAllMatching);
  /* Backdrop only. The lightbox now contains a video with its own controls,
     and closing on any click inside it would make the scrubber unusable. */
  $('lightbox').addEventListener('click', (event) => {
    if (event.target === $('lightbox')) closeLightbox();
  });
  $('search').addEventListener('input', onSearchInput);
  for (const id of ['f-group', 'f-cat', 'f-folder', 'f-show', 'f-view']) {
    $(id).addEventListener('change', () => { state.like = null; reload(); });
  }
  for (const button of $('density').querySelectorAll('button')) {
    button.addEventListener('click', () => {
      document.body.dataset.density = button.dataset.size;
      for (const other of $('density').querySelectorAll('button')) {
        other.setAttribute('aria-pressed', other === button);
      }
      reflow();
    });
  }
  document.addEventListener('keydown', onKeyDown);
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', layout);

  refreshStats().then(reload);
  watchProgress();
}

start();
