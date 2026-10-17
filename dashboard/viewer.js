// ROBOT CI — pixel-directed research console.
//
// This is a recorded-experiment dashboard. Every number it shows is recomputed
// from the frozen evidence under experiments/results/ by
// experiments/build_dashboard_data.py, and served as dashboard/data/evidence.json.
// Nothing here is interpolated, projected or invented: a missing metric stays
// "Not measured", and the central robot is a telemetry-driven reconstruction of
// a recorded episode -- never presented as a live camera.

import * as THREE from 'three';

// --------------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};
const fmt = (v, digits = 2) => (v == null ? 'Not measured' : Number(v).toFixed(digits));
const pct = (v, digits = 1) => (v == null ? 'Not measured' : `${(v * 100).toFixed(digits)}%`);
const NA = '<span class="muted">Not measured</span>';

let EVIDENCE = null;

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove('show'), 2200);
}

// --------------------------------------------------------------------------
// layout skeleton (the three columns), created once
// --------------------------------------------------------------------------
function buildSkeleton() {
  const consoleEl = $('console');
  consoleEl.innerHTML = '';

  // ---- left column ----
  const left = el('section', 'col col-left');
  left.id = 'col-left';
  left.innerHTML = `
    <div class="panel"><h2 class="panel-title">Experiment Configuration</h2>
      <dl class="kv" id="config-list"></dl></div>
    <div class="panel"><h2 class="panel-title">Research Metrics</h2>
      <div class="metric-grid" id="metric-grid"></div></div>
    <div class="panel"><h2 class="panel-title">Policy Comparison</h2>
      <div id="policy-comparison"></div></div>
    <div class="panel"><h2 class="panel-title">Subsystem Attribution <span class="sub">RQ3</span></h2>
      <div id="rq3-attribution"></div></div>
    <div class="panel"><h2 class="panel-title">Reproducibility</h2>
      <dl class="kv" id="reproducibility"></dl>
      <p class="finding" id="repro-interp"></p>
      <button class="btn copy" id="copy-repro">Copy reproduction command</button></div>`;

  // ---- center column ----
  const center = el('section', 'col col-center');
  center.innerHTML = `
    <div class="panel viewport-panel">
      <div class="vp-toolbar">
        <button class="vbtn" id="btn-play" title="Play / pause replay" aria-label="Play">▶</button>
        <button class="vbtn" id="btn-restart" title="Restart replay" aria-label="Restart">⟲</button>
        <button class="vbtn" id="btn-view-reset" title="Reset camera view" aria-label="Reset view">⌖</button>
        <div class="speed-group" role="group" aria-label="Playback speed">
          <button class="vbtn speed active" data-speed="1">1×</button>
          <button class="vbtn speed" data-speed="2">2×</button>
          <button class="vbtn speed" data-speed="4">4×</button>
        </div>
        <input class="scrub" id="scrub" type="range" min="0" max="1000" value="0" aria-label="Timeline" />
        <span class="vp-clock mono" id="vp-clock">t = 0.00 s</span>
      </div>
      <div id="viewport" class="viewport"></div>
      <div class="vp-footer">
        <div class="vp-telemetry" id="vp-telemetry"></div>
        <div class="vp-label">TELEMETRY-DRIVEN RECONSTRUCTION</div>
      </div>
    </div>

    <div class="panel" id="panel-discovery">
      <h2 class="panel-title">Counterexample Discovery <span class="stages" id="stages"></span></h2>
      <div class="cell-detail" id="ce-card"></div>
      <div class="stage-row" id="stage-row"></div>
      <div class="stage-detail" id="stage-detail"></div>
      <div class="chart-host"><canvas id="search-chart"></canvas></div>
      <p class="finding" id="search-limitation"></p>
    </div>

    <div class="panel" id="panel-settle">
      <h2 class="panel-title">Settle-Window Trade-off <span class="sub" id="settle-sub"></span></h2>
      <div class="chart-host"><canvas id="settle-chart"></canvas></div>
      <p class="finding" id="settle-finding"></p>
    </div>

    <div class="panel" id="panel-rq5">
      <h2 class="panel-title">Perturbation Boundaries <span class="sub" id="rq5-sub"></span></h2>
      <div class="chart-host"><canvas id="rq5-chart"></canvas></div>
      <p class="finding" id="rq5-finding"></p>
    </div>`;

  // ---- right column ----
  const right = el('section', 'col col-right');
  right.innerHTML = `
    <div class="panel">
      <h2 class="panel-title">Behavioral Boundary Map</h2>
      <div class="heatmap-wrap">
        <canvas id="boundary-map"></canvas>
        <div class="legend" id="boundary-legend"></div>
      </div>
      <p class="axis-note" id="boundary-note"></p>
      <div class="cell-detail" id="cell-detail"><span class="muted">Select a cell to inspect its measured configuration.</span></div>
    </div>

    <div class="panel">
      <h2 class="panel-title">Evidence &amp; Artifacts</h2>
      <div class="evidence-list" id="evidence-list"></div>
      <details class="checklog" id="checklog"><summary></summary><div id="checklog-body"></div></details>
      <div class="provenance" id="provenance"></div>
    </div>`;

  consoleEl.append(left, center, right);
}

// --------------------------------------------------------------------------
// left column rendering
// --------------------------------------------------------------------------
function renderStatus() {
  const git = EVIDENCE.git || {};
  $('status-rev').textContent = git.short || EVIDENCE.revision;
  $('status-sim').textContent = EVIDENCE.config?.backend || 'MuJoCo';
}

function renderConfig() {
  const c = EVIDENCE.config;
  const rows = [
    ['Experiment', c.experiment_id],
    ['Research question', c.research_question],
    ['Baseline policy', c.baseline],
    ['Candidate policy', c.candidate],
    ['Perturbation', `${c.perturbation} (${c.perturbation_unit})`],
    ['Value range', c.value_range?.map((v) => `${v}`).join(' – ') + ` ${c.perturbation_unit}`],
    ['Search algorithm', c.search_algorithm],
    ['Evaluation budget', c.budget],
    ['Discovery seed', c.seed],
    ['Backend', c.backend],
    ['Code revision', EVIDENCE.git?.short || EVIDENCE.revision],
    // real scenario hash from the counterexample artifact; None -> "Not measured".
    // Shown truncated like the git short-revision; hover for the full 71-char value.
    ['Config fingerprint', c.config_hash ? c.config_hash.slice(0, 15) + '…' : null],
    ['Experiment state', c.state],
    ['Evidence provenance', c.evidence_provenance],
  ];
  const dl = $('config-list');
  dl.innerHTML = '';
  for (const [k, v] of rows) {
    dl.append(el('dt', null, k));
    const dd = el('dd', null, v == null ? 'Not measured' : String(v));
    if (k === 'Config fingerprint' && c.config_hash) dd.title = c.config_hash;
    dl.append(dd);
  }
}

function renderMetrics() {
  const m = EVIDENCE.metrics;
  const tiles = [
    { v: m.episodes_evaluated, lbl: 'Episodes evaluated', ctx: 'nominal + settle sweep', int: true },
    { v: m.nominal_solvability, lbl: 'Nominal solvability', ctx: 'policy_v1, 120 seeds', pct: true },
    { v: m.mean_execution_steps, lbl: 'Mean execution steps', ctx: 'policy_v1 nominal' },
    { v: m.counterexample_discovery_cost, lbl: 'Discovery cost', ctx: 'adaptive oracle evals', int: true },
    { v: m.boundary_estimation_error, lbl: 'Boundary-est. error', ctx: 'not stored in evidence' },
    { v: m.regression_reproduction_rate, lbl: 'Regression repro. rate', ctx: 'RQ4, 11 replay seeds', pct: true },
  ];
  const grid = $('metric-grid');
  grid.innerHTML = '';
  for (const t of tiles) {
    const card = el('div', 'metric');
    const val = el('div', 'val');
    if (t.v == null) {
      val.classList.add('na');
      val.textContent = 'Not measured';
    } else if (t.pct) {
      val.textContent = (t.v * 100).toFixed(1) + '%';
    } else if (t.int) {
      val.textContent = Math.round(t.v).toLocaleString();
    } else {
      val.textContent = Number(t.v).toFixed(1);
    }
    card.append(val, el('div', 'lbl', t.lbl), el('div', 'ctx', t.ctx));
    grid.append(card);
  }
}

function renderPolicyComparison() {
  const pc = EVIDENCE.policy_comparison;
  const host = $('policy-comparison');
  if (!pc?.available) { host.innerHTML = NA; return; }
  const table = el('table', 'pc-table');
  table.innerHTML = `<thead><tr><th>condition</th><th>${pc.baseline}</th><th>${pc.candidate}</th><th>Δ</th></tr></thead>`;
  const body = el('tbody');
  for (const c of pc.cells) {
    const tr = el('tr');
    tr.append(el('td', null, `depth ${c.depth_noise_mm} mm`));
    tr.append(el('td', null, pct(c.baseline_success, 0)));
    tr.append(el('td', null, pct(c.candidate_success, 0)));
    const dcls = c.delta > 0 ? 'pos' : c.delta < 0 ? 'neg' : 'zero';
    tr.append(el('td', `delta ${dcls}`, (c.delta > 0 ? '+' : '') + c.delta.toFixed(2)));
    body.append(tr);
  }
  // aggregate row with measured execution steps
  const agg = el('tr');
  agg.append(el('td', null, 'aggregate'));
  agg.append(el('td', null, pct(pc.aggregate.baseline_success, 0)));
  agg.append(el('td', null, pct(pc.aggregate.candidate_success, 0)));
  const ad = pc.aggregate.delta;
  agg.append(el('td', `delta ${ad > 0 ? 'pos' : ad < 0 ? 'neg' : 'zero'}`, (ad > 0 ? '+' : '') + ad.toFixed(3)));
  body.append(agg);
  table.append(body);
  host.innerHTML = '';
  host.append(table);
  if (pc.baseline_steps_mean != null && pc.candidate_steps_mean != null) {
    const note = el('div', 'finding');
    note.innerHTML = `Execution steps: <b>${fmt(pc.baseline_steps_mean, 1)}</b> (${pc.baseline}) vs ` +
      `<b>${fmt(pc.candidate_steps_mean, 1)}</b> (${pc.candidate}) → Δ ${fmt(pc.steps_delta, 1)} steps (nominal).`;
    host.append(note);
  }
  // per-policy nominal baselines (all three solve 120/120; cost differs)
  const nom = EVIDENCE.nominal;
  if (nom?.available && nom.policies) {
    const table2 = el('table', 'pc-table');
    table2.innerHTML = '<thead><tr><th>nominal policy</th><th>success</th><th>steps</th></tr></thead>';
    const body2 = el('tbody');
    for (const p of Object.values(nom.policies)) {
      const tr = el('tr');
      tr.append(el('td', null, p.policy));
      tr.append(el('td', null, `${p.successes}/${p.seeds}`));
      tr.append(el('td', null, fmt(p.steps_mean, 1)));
      body2.append(tr);
    }
    table2.append(body2);
    host.append(table2);
  }
}

// --------------------------------------------------------------------------
// RQ3 — subsystem attribution table (correlation vs OFAT vs interaction-aware)
// --------------------------------------------------------------------------
function renderAttribution() {
  const rq3 = EVIDENCE.rq3;
  const host = $('rq3-attribution');
  if (!rq3?.methods) { host.innerHTML = NA; return; }
  const gt = rq3.ground_truth || {};
  const table = el('table', 'pc-table');
  table.innerHTML = '<thead><tr><th>method</th><th>implicated</th><th>episodes</th><th>correct?</th></tr></thead>';
  const body = el('tbody');
  const rows = [
    ['correlation only', rq3.methods.correlation_only],
    ['OFAT', rq3.methods.ofat],
    ['interaction-aware', rq3.methods.interaction_aware],
  ];
  let correct = 0;
  for (const [label, m] of rows) {
    if (!m) continue;
    if (m.correct) correct += 1;
    const tr = el('tr');
    tr.append(el('td', null, label));
    tr.append(el('td', null, m.implicated ?? 'Not measured'));
    tr.append(el('td', null, m.cost_episodes == null ? 'Not measured' : m.cost_episodes.toLocaleString()));
    tr.append(el('td', `delta ${m.correct ? 'pos' : 'neg'}`, m.correct ? 'yes' : 'no'));
    body.append(tr);
  }
  table.append(body);
  host.innerHTML = '';
  host.append(table);
  const note = el('p', 'finding');
  note.innerHTML = `Ground truth: <b>${gt.subsystem ?? 'Not measured'}</b>` +
    (gt.parameter ? ` · ${gt.parameter}` : '') +
    ` — <b>${correct}/${rows.filter((r) => r[1]).length}</b> methods named it.` +
    (gt.reason ? ` ${gt.reason}` : '');
  host.append(note);
}

function renderReproducibility() {
  const r = EVIDENCE.reproducibility;
  const dl = $('reproducibility');
  dl.innerHTML = '';
  const cond = r?.condition?.depth_noise_mm;
  const rows = [
    ['Source revision', EVIDENCE.git?.revision || EVIDENCE.revision],
    ['Discovery seed', r?.discovery_seed],
    ['Replay seeds', `${r?.replay_count} seeds`],
    ['Reproduced', `${r?.reproduced}/${r?.replay_count}`],
    ['Reproduction rate', r?.available ? pct(r.rate) : 'Not measured'],
    ['Condition', cond != null ? `depth_noise_mm = ${cond}` : 'Not measured'],
  ];
  for (const [k, v] of rows) { dl.append(el('dt', null, k)); dl.append(el('dd', null, v == null ? 'Not measured' : String(v))); }
  // how to read a rate below 1.0 — the frozen interpretation, verbatim
  const interp = $('repro-interp');
  if (interp) interp.innerHTML = r?.interpretation ? `<i>Interpretation:</i> ${r.interpretation}` : '';

  $('copy-repro').addEventListener('click', async () => {
    const cmd = `python experiments/rq4_reproducibility.py`;
    try {
      await navigator.clipboard.writeText(cmd);
      toast('Reproduction command copied');
    } catch {
      toast('Copy failed — command: ' + cmd);
    }
  });
}

// --------------------------------------------------------------------------
// central robot viewport — telemetry-driven reconstruction
// --------------------------------------------------------------------------
const ARM_JOINTS = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7'];

// Default framing of the 3D view — the orbit camera's "home" pose. Drag to
// orbit, wheel to zoom, right-drag to pan; the ⌖ toolbar button returns here.
const VIEW_HOME = {
  position: new THREE.Vector3(0.92, -1.06, 0.74),
  target: new THREE.Vector3(0.0, 0.05, 0.34),
};

const REPLAY = {
  playing: true,
  frame: 0,
  speed: 1,
  acc: 0,
  lastT: 0,
  robot: null,
  scene: null,
  camera: null,
  renderer: null,
  linkMeshes: [],
  fingerMeshes: [],
  objectMesh: null,
  frames: [],
  phases: [],
};

function initViewport() {
  const host = $('viewport');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060d17);
  scene.fog = new THREE.Fog(0x060d17, 2.2, 4.5);

  const camera = new THREE.PerspectiveCamera(44, 1, 0.05, 100);
  camera.up.set(0, 0, 1); // URDF is Z-up
  camera.position.copy(VIEW_HOME.position);
  camera.lookAt(VIEW_HOME.target);

  // studio lighting — bright enough that the white arm reads clearly against
  // the dark-navy backdrop instead of falling into shadow.
  scene.add(new THREE.HemisphereLight(0xbfd8f2, 0x12202f, 1.35));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(0.8, -1.4, 1.6);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left = key.shadow.camera.bottom = -0.8;
  key.shadow.camera.right = key.shadow.camera.top = 0.8;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x66ccff, 0.8);
  rim.position.set(-1.2, 1.0, 0.8);
  scene.add(rim);
  // front fill from the camera side so the links facing the viewer are lit
  const fill = new THREE.DirectionalLight(0xdfeaf5, 0.7);
  fill.position.set(0.9, -1.1, 0.5);
  scene.add(fill);

  // ground + table
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(2.2, 48),
    new THREE.MeshStandardMaterial({ color: 0x0a1522, roughness: 1.0 }),
  );
  ground.position.z = -0.001;
  ground.receiveShadow = true;
  scene.add(ground);

  const sc = EVIDENCE.scene || {};
  const tableH = sc.table_height_m ?? 0;
  const tableMesh = new THREE.Mesh(
    new THREE.BoxGeometry(sc.table_extent_m?.[0] ?? 0.6, sc.table_extent_m?.[1] ?? 0.4, 0.03),
    new THREE.MeshStandardMaterial({ color: 0x182436, roughness: 0.85, metalness: 0.05 }),
  );
  tableMesh.position.set(sc.workspace_center?.[0] ?? 0, sc.workspace_center?.[1] ?? 0, tableH - 0.015);
  tableMesh.receiveShadow = true;
  scene.add(tableMesh);

  // target zone ring
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.06, 0.075, 32),
    new THREE.MeshBasicMaterial({ color: 0x35c5e0, transparent: true, opacity: 0.5, side: THREE.DoubleSide }),
  );
  ring.position.set(0.22, 0.12, tableH + 0.002);
  scene.add(ring);

  // manipulable object
  const objectMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.04, 0.04, 0.04),
    new THREE.MeshStandardMaterial({ color: 0xe0a92e, roughness: 0.5 }),
  );
  objectMesh.castShadow = true;
  scene.add(objectMesh);

  Object.assign(REPLAY, { renderer, scene, camera, objectMesh });

  // ---- camera orbit controls ---------------------------------------------
  // Z-up spherical orbit (matches camera.up = Z): azimuth around the vertical
  // axis, elevation above the ground plane. Left-drag = orbit, wheel = zoom,
  // right/middle-drag = pan, ⌖ button or double-click = reset to VIEW_HOME.
  const canvas = renderer.domElement;
  canvas.style.touchAction = 'none'; // pointer drags must not scroll the page
  const homeOffset = VIEW_HOME.position.clone().sub(VIEW_HOME.target);
  const orbit = {
    target: VIEW_HOME.target.clone(),
    radius: homeOffset.length(),
    azimuth: Math.atan2(homeOffset.y, homeOffset.x),
    elevation: Math.asin(homeOffset.z / homeOffset.length()),
    dragging: false,
    panning: false,
    lastX: 0,
    lastY: 0,
  };

  function applyOrbit() {
    // clamp: stay above the ground plane, off the degenerate poles, sane zoom
    orbit.elevation = THREE.MathUtils.clamp(orbit.elevation, 0.05, 1.45);
    orbit.radius = THREE.MathUtils.clamp(orbit.radius, 0.35, 6.0);
    const ce = Math.cos(orbit.elevation);
    camera.position.set(
      orbit.target.x + orbit.radius * ce * Math.cos(orbit.azimuth),
      orbit.target.y + orbit.radius * ce * Math.sin(orbit.azimuth),
      orbit.target.z + orbit.radius * Math.sin(orbit.elevation),
    );
    camera.lookAt(orbit.target);
  }

  function resetView() {
    orbit.target.copy(VIEW_HOME.target);
    orbit.radius = homeOffset.length();
    orbit.azimuth = Math.atan2(homeOffset.y, homeOffset.x);
    orbit.elevation = Math.asin(homeOffset.z / homeOffset.length());
    applyOrbit();
  }

  canvas.addEventListener('pointerdown', (e) => {
    orbit.dragging = e.button === 0;
    orbit.panning = e.button === 1 || e.button === 2;
    orbit.lastX = e.clientX;
    orbit.lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!orbit.dragging && !orbit.panning) return;
    const dx = e.clientX - orbit.lastX;
    const dy = e.clientY - orbit.lastY;
    orbit.lastX = e.clientX;
    orbit.lastY = e.clientY;
    if (orbit.dragging) {
      orbit.azimuth -= dx * 0.006;
      orbit.elevation += dy * 0.006;
    } else {
      // pan: move target in the camera's screen plane
      const scale = orbit.radius * 0.0016;
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      camera.matrix.extractBasis(right, up, new THREE.Vector3());
      orbit.target.addScaledVector(right, dx * scale).addScaledVector(up, -dy * scale);
    }
    applyOrbit();
  });
  const endDrag = (e) => {
    orbit.dragging = false;
    orbit.panning = false;
    if (e.pointerId !== undefined && canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('dblclick', resetView);
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    orbit.radius *= Math.exp(e.deltaY * 0.0012);
    applyOrbit();
  }, { passive: false });
  $('btn-view-reset').addEventListener('click', resetView);
  applyOrbit();

  function resize() {
    const w = host.clientWidth;
    const h = host.clientHeight;
    if (w === 0 || h === 0) return;
    // updateStyle=true (default): keeps the canvas CSS box equal to the host,
    // otherwise on HiDPI screens (dpr 2) the canvas renders at 2x CSS pixels
    // and the overflow:hidden host shows only a cropped quarter of the scene.
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  // Re-fit on any layout change (panel toggles, font load, sidebar collapse…),
  // not just window resizes — a stale buffer size is what crops the view.
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => resize()).observe(host);
  }
  resize();
  REPLAY._resize = resize;
  animate();
}

// Build link meshes from the projected URDF (real cylinders / boxes).
function buildRobot(robot) {
  const robotGroup = new THREE.Group();
  const armMat = new THREE.MeshStandardMaterial({ color: 0xdfe6ee, roughness: 0.42, metalness: 0.25 });
  const jointMat = new THREE.MeshStandardMaterial({ color: 0x2b3a4d, roughness: 0.5, metalness: 0.3 });
  const fingerMat = new THREE.MeshStandardMaterial({ color: 0x9fb3c8, roughness: 0.4, metalness: 0.3 });

  const byName = new Map();
  for (const link of robot.links) {
    const group = new THREE.Group();
    group.name = link.name;
    for (const vis of link.visuals) {
      let geo;
      const s = vis.shape;
      if (s.kind === 'cylinder') geo = new THREE.CylinderGeometry(s.radius, s.radius, s.length, 20);
      else if (s.kind === 'box') geo = new THREE.BoxGeometry(s.size[0], s.size[1], s.size[2]);
      else if (s.kind === 'sphere') geo = new THREE.SphereGeometry(s.radius, 18, 14);
      else continue;
      const mat = link.name.includes('finger') ? fingerMat
        : link.name === 'panda_hand' ? jointMat : armMat;
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(vis.xyz[0], vis.xyz[1], vis.xyz[2]);
      mesh.rotation.set(vis.rpy[0], vis.rpy[1], vis.rpy[2]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
      if (link.name.includes('finger')) REPLAY.fingerMeshes.push(group);
    }
    byName.set(link.name, group);
    robotGroup.add(group);
  }
  REPLAY.scene.add(robotGroup);
  REPLAY.robot = robotGroup;
  REPLAY.linkGroups = byName;
  REPLAY.joints = robot.joints;
}
// Forward kinematics: compute world matrices for every link from joint angles.
function computeFrames(angles) {
  const joints = REPLAY.joints;
  const byParent = new Map();
  const childLinks = new Set(joints.map((j) => j.child));
  for (const j of joints) {
    if (!byParent.has(j.parent)) byParent.set(j.parent, []);
    byParent.get(j.parent).push(j);
  }
  const root = REPLAY.linkGroups.keys().next().value;
  const rootLink = [...childLinks].length ? joints.find((j) => !childLinks.has(j.parent))?.parent : root;
  const base = REPLAY.linkGroups.get(rootLink || root);
  const worldOf = new Map([[rootLink || root, base.matrix.clone()]]);
  const queue = [[rootLink || root, base.matrix.clone()]];
  while (queue.length) {
    const [link, parentMat] = queue.shift();
    for (const j of byParent.get(link) || []) {
      const m = parentMat.clone();
      m.multiply(new THREE.Matrix4().makeTranslation(j.xyz[0], j.xyz[1], j.xyz[2]));
      m.multiply(new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(j.rpy[0], j.rpy[1], j.rpy[2], 'ZYX')));
      const value = angles[j.name] ?? 0;
      const axis = new THREE.Vector3(j.axis[0], j.axis[1], j.axis[2]).normalize();
      if (j.type === 'prismatic') {
        m.multiply(new THREE.Matrix4().makeTranslation(axis.x * value, axis.y * value, axis.z * value));
      } else if (j.type !== 'fixed') {
        m.multiply(new THREE.Matrix4().makeRotationAxis(axis, value));
      }
      worldOf.set(j.child, m);
      queue.push([j.child, m]);
    }
  }
  return worldOf;
}

// Map telemetry CSV joint names to URDF joint names.
function angleMap(frame) {
  const map = {};
  ARM_JOINTS.forEach((j, i) => { map['joint' + (i + 1)] = frame.joints[j]; });
  return map;
}

function applyFrame(frame) {
  if (!REPLAY.linkGroups) return;
  // reset group matrices to identity base (robot base sits on table)
  const worldOf = computeFrames(angleMap(frame));
  for (const [name, group] of REPLAY.linkGroups) {
    const m = worldOf.get(name);
    if (m) group.matrix.copy(m);
    else group.matrix.identity();
    group.matrix.decompose(group.position, group.quaternion, group.scale);
  }
  // lift the whole arm onto the table surface
  const tableH = EVIDENCE.scene?.table_height_m ?? 0;
  REPLAY.robot.position.z = tableH;

  // object follows its recorded pose
  REPLAY.objectMesh.position.set(frame.obj[0], frame.obj[1], frame.obj[2]);
  const grasped = frame.grasped === 1;
  REPLAY.objectMesh.material.color.setHex(grasped ? 0x35c46a : 0xe0a92e);
}

function renderTelemetry(frame) {
  const host = $('vp-telemetry');
  const grasped = frame.grasped === 1;
  const items = [
    ['Phase', `<span class="phase-pill">${frame.phase}</span>`],
    ['Sim time', `${frame.t.toFixed(2)} s`],
    ['Gripper', grasped ? 'closed' : 'open'],
    ['Aperture', `${(frame.aperture * 1000).toFixed(1)} mm`],
    ['Object lift', `${(frame.obj[2] * 1000).toFixed(0)} mm`],
    ['Policy', 'policy_v1'],
    ['Seed', '3'],
  ];
  host.innerHTML = items
    .map(([k, v]) => `<div><div class="t-k">${k}</div><div class="t-v">${v}</div></div>`)
    .join('');
  $('vp-clock').textContent = `t = ${frame.t.toFixed(2)} s`;
}

function animate() {
  requestAnimationFrame(animate);
  const frames = REPLAY.frames;
  if (frames.length && REPLAY.linkGroups) {
    const now = performance.now();
    const dt = Math.min(0.1, (now - REPLAY.lastT) / 1000) || 0;
    REPLAY.lastT = now;
    if (REPLAY.playing) {
      REPLAY.acc += dt * REPLAY.speed * 18; // ~18 fps recorded cadence
      while (REPLAY.acc >= 1) {
        REPLAY.frame = (REPLAY.frame + 1) % frames.length;
        REPLAY.acc -= 1;
      }
    }
    const frame = frames[REPLAY.frame];
    applyFrame(frame);
    renderTelemetry(frame);
    $('scrub').value = String(Math.round((REPLAY.frame / (frames.length - 1)) * 1000));
  }
  REPLAY.renderer.render(REPLAY.scene, REPLAY.camera);
}

function wireControls() {
  $('btn-play').addEventListener('click', () => {
    REPLAY.playing = !REPLAY.playing;
    $('btn-play').textContent = REPLAY.playing ? '▶' : '❚❚';
  });
  $('btn-restart').addEventListener('click', () => {
    REPLAY.frame = 0; REPLAY.acc = 0;
  });
  document.querySelectorAll('.speed').forEach((b) => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.speed').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      REPLAY.speed = Number(b.dataset.speed);
    });
  });
  $('scrub').addEventListener('input', (e) => {
    const frames = REPLAY.frames;
    if (!frames.length) return;
    REPLAY.frame = Math.round((Number(e.target.value) / 1000) * (frames.length - 1));
    REPLAY.acc = 0;
  });
}

// --------------------------------------------------------------------------
// discovery stages (SEARCH → MINIMIZE → REPRODUCE → COMPARE)
// --------------------------------------------------------------------------
function renderDiscovery() {
  const search = EVIDENCE.search;
  const repro = EVIDENCE.reproducibility;
  const opp = EVIDENCE.opposing_effect;
  const ce = EVIDENCE.counterexample;
  if (!search?.available) { $('stage-row').innerHTML = NA; return; }

  // the discovered counterexample itself (id, swept value, outcomes)
  const ceCard = $('ce-card');
  if (ce?.available) {
    ceCard.innerHTML =
      '<div class="cd-title">Found: <span class="mono">' + ce.id + '</span></div>' +
      '<dl class="cd-grid">' +
      '<dt>depth noise</dt><dd>' + (ce.depth_noise_mm != null ? ce.depth_noise_mm + ' mm' : 'Not measured') + '</dd>' +
      '<dt>seed</dt><dd>' + (ce.seed ?? 'Not measured') + '</dd>' +
      '<dt>baseline</dt><dd>' + (ce.baseline?.policy ?? '?') + ' → ' + (ce.baseline?.result ?? '?') + '</dd>' +
      '<dt>candidate</dt><dd>' + (ce.candidate?.policy ?? '?') + ' → ' + (ce.candidate?.result ?? '?') + '</dd>' +
      '<dt>artifact</dt><dd>' + ce.file + '</dd>' +
      '</dl>';
  } else {
    ceCard.innerHTML = '<span class="muted">No counterexample recorded in evidence.</span>';
  }

  const stages = [
    { id: 'search', label: 'SEARCH', detail: `Adaptive search over ${search.axis?.name} in [${search.axis?.low}, ${search.axis?.high}] ${search.axis?.unit}; budget ${search.budget}; seed ${search.seed}. Source: rq1_search_strategies.json.` },
    { id: 'minimize', label: 'MINIMIZE', detail: search.strategies?.find((s) => s.strategy === 'adaptive')?.boundary != null
        ? `Bisection narrowed the boundary to ${search.strategies.find((s) => s.strategy === 'adaptive').boundary} ${search.axis?.unit}. Source: rq1_search_strategies.json.`
        : 'Boundary not bracketed on this axis.' },
    { id: 'reproduce', label: 'REPRODUCE', detail: repro?.available
        ? `Replayed the counterexample at depth_noise=${repro.condition?.depth_noise_mm} mm across ${repro.replay_count} seeds; reproduced ${repro.reproduced}/${repro.replay_count} (${pct(repro.rate)}). Source: rq4_reproducibility.json.`
        : 'No reproduction record.' },
    { id: 'compare', label: 'COMPARE', detail: opp?.available
        ? `Opposing-effect sweep: verdict “${opp.verdict}”, sign-flip at ${(opp.flipped_levels || []).join(', ') || 'none'} mm. ` +
          (opp.hypothesis_status ? `<span class="badge-hyp">Hypothesis ${opp.hypothesis_status}</span> ` : '') +
          'Source: rq2_opposing_effect_current.json.'
        : 'No opposing-effect record.' },
  ];
  const row = $('stage-row');
  row.innerHTML = '';
  stages.forEach((s, i) => {
    const b = el('button', 'stage' + (i === 0 ? ' active' : ''), s.label);
    b.addEventListener('click', () => {
      document.querySelectorAll('.stage').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      $('stage-detail').innerHTML = s.detail;
    });
    row.append(b);
  });
  $('stage-detail').innerHTML = stages[0].detail;
  // the search's own stated limitation — shown, never paraphrased away
  $('search-limitation').innerHTML = search.limitation ? `<i>Limitation:</i> ${search.limitation}` : '';
  drawSearchChart();
}

// Simple canvas line/bar chart helper.
function setupCanvas(id) {
  const cv = $(id);
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}

function drawSearchChart() {
  const search = EVIDENCE.search;
  if (!search?.available) return;
  const { ctx, w, h } = setupCanvas('search-chart');
  const pad = { l: 34, r: 10, t: 12, b: 26 };
  const strategies = search.strategies;
  const maxEval = Math.max(...strategies.map((s) => s.oracle_evaluations));
  const cw = (w - pad.l - pad.r) / strategies.length;
  const colors = { grid: '#24384f', adaptive: '#35c5e0', other: '#3a5570' };
  ctx.clearRect(0, 0, w, h);
  // axes
  ctx.strokeStyle = colors.grid; ctx.fillStyle = '#7e93ad'; ctx.font = '9px sans-serif'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = (maxEval / 4) * i;
    const y = h - pad.b - (v / maxEval) * (h - pad.t - pad.b);
    ctx.fillText(v.toFixed(0), pad.l - 4, y + 3);
  }
  ctx.textAlign = 'center';
  strategies.forEach((s, i) => {
    const bh = (s.oracle_evaluations / maxEval) * (h - pad.t - pad.b);
    const x = pad.l + i * cw + cw * 0.25;
    const bw = cw * 0.5;
    ctx.fillStyle = s.strategy === 'adaptive' ? colors.adaptive : colors.other;
    ctx.fillRect(x, h - pad.b - bh, bw, bh);
    ctx.fillStyle = '#7e93ad';
    ctx.fillText(s.strategy, x + bw / 2, h - pad.b + 12);
    ctx.fillStyle = '#dbe6f2';
    ctx.fillText(String(s.oracle_evaluations), x + bw / 2, h - pad.b - bh - 4);
  });
  ctx.save();
  ctx.translate(10, pad.t + 4); ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#7e93ad'; ctx.textAlign = 'left';
  ctx.fillText('oracle evaluations to failure', 0, 0);
  ctx.restore();
}

function drawSettleChart() {
  const st = EVIDENCE.settle_tradeoff;
  if (!st?.available) return;
  $('settle-sub').textContent = `revision ${st.revision} · ${st.n_significant}/${st.comparisons.length} paired McNemar significant`;
  const { ctx, w, h } = setupCanvas('settle-chart');
  const pad = { l: 38, r: 12, t: 12, b: 26 };
  const series = st.series;
  const steps = series.map((s) => s.steps_mean).filter((v) => v != null);
  const minS = Math.min(...steps) - 2, maxS = Math.max(...steps) + 2;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#24384f'; ctx.fillStyle = '#7e93ad'; ctx.font = '9px sans-serif'; ctx.lineWidth = 1;
  // axes
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = minS + ((maxS - minS) / 4) * i;
    const y = h - pad.b - (i / 4) * (h - pad.t - pad.b);
    ctx.fillText(v.toFixed(0), pad.l - 4, y + 3);
    if (i > 0) { ctx.strokeStyle = '#16222f'; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.strokeStyle = '#24384f'; }
  }
  const cw = (w - pad.l - pad.r) / series.length;
  ctx.textAlign = 'center';
  // line + points for mean steps
  ctx.strokeStyle = '#35c5e0'; ctx.lineWidth = 2; ctx.beginPath();
  const pts = series.map((s, i) => [pad.l + i * cw + cw / 2, s.steps_mean == null ? null : h - pad.b - ((s.steps_mean - minS) / (maxS - minS)) * (h - pad.t - pad.b)]);
  pts.forEach(([x, y], i) => { if (y == null) return; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.stroke();
  series.forEach((s, i) => {
    const [x, y] = pts[i];
    if (y == null) return;
    ctx.fillStyle = '#35c5e0'; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#7e93ad'; ctx.fillText(`w${s.window}`, x, h - pad.b + 12);
    ctx.fillStyle = '#dbe6f2'; ctx.fillText(s.steps_mean.toFixed(1), x, y - 8);
  });
  ctx.save(); ctx.translate(10, pad.t + 4); ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#7e93ad'; ctx.textAlign = 'left'; ctx.fillText('mean execution steps', 0, 0); ctx.restore();

  $('settle-finding').innerHTML =
    `<b>Finding:</b> ${st.finding} — execution steps rise with the settle window while ` +
    `robustness shows <b>${st.n_significant}</b> significant paired change(s). ` +
    `<i>Caveat:</i> ${st.caveat}.`;
}

// --------------------------------------------------------------------------
// RQ5 — one-dimensional perturbation boundary curves (baseline vs candidate)
// --------------------------------------------------------------------------
function drawRq5Chart() {
  const rq5 = EVIDENCE.rq5;
  if (!rq5?.available) return;
  const curves = rq5.curves || {};
  // draw the axis that separates (boundary set); annotate the rest
  const sepName = Object.keys(rq5.boundaries || {})[0];
  const curve = sepName ? curves[sepName] : null;
  $('rq5-sub').textContent = `${rq5.baseline} vs ${rq5.candidate}`;
  const noSep = rq5.axes_with_no_separation || [];
  if (!curve?.points?.length) {
    $('rq5-finding').innerHTML = noSep.length
      ? `No separating axis among ${noSep.length + Object.keys(curves).length} swept; no separation on any.`
      : NA;
    return;
  }
  const { ctx, w, h } = setupCanvas('rq5-chart');
  const pad = { l: 38, r: 14, t: 12, b: 26 };
  const pts = curve.points;
  const maxX = Math.max(...pts.map((p) => p.value)) || 1;
  const X = (v) => pad.l + (v / maxX) * (w - pad.l - pad.r);
  const Y = (r) => h - pad.b - r * (h - pad.t - pad.b); // success rate 0..1
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#24384f'; ctx.fillStyle = '#7e93ad'; ctx.font = '9px sans-serif'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = h - pad.b - (i / 4) * (h - pad.t - pad.b);
    ctx.fillText((i * 25) + '%', pad.l - 4, y + 3);
    if (i > 0) { ctx.strokeStyle = '#16222f'; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.strokeStyle = '#24384f'; }
  }
  ctx.textAlign = 'center';
  // two success curves: baseline (v3) and candidate (v1)
  for (const [key, color] of [['baseline', '#35c5e0'], ['candidate', '#e0a92e']]) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach((p, i) => { const x = X(p.value), y = Y(p[key]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();
    pts.forEach((p) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(X(p.value), Y(p[key]), 2.5, 0, Math.PI * 2); ctx.fill(); });
  }
  // x labels + boundary marker
  pts.forEach((p) => {
    ctx.fillStyle = '#7e93ad'; ctx.fillText(String(p.value), X(p.value), h - pad.b + 12);
    if (curve.boundary_value != null && p.value === curve.boundary_value) {
      ctx.strokeStyle = '#e0565f'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(p.value), pad.t); ctx.lineTo(X(p.value), h - pad.b); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#e0565f'; ctx.fillText('boundary', X(p.value), pad.t - 2);
    }
  });
  ctx.save(); ctx.translate(10, pad.t + 4); ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = '#7e93ad'; ctx.textAlign = 'left'; ctx.fillText('success rate', 0, 0); ctx.restore();
  ctx.fillStyle = '#7e93ad'; ctx.textAlign = 'center';
  ctx.fillText(curve.path + ' (' + (curve.unit || '') + ')', pad.l + (w - pad.l - pad.r) / 2, h - 4);

  $('rq5-finding').innerHTML =
    `<b>Finding:</b> boundary at ${sepName} = <b>${curve.boundary_value} ${curve.unit || ''}</b> ` +
    `(first level where ${rq5.baseline} succeeds and ${rq5.candidate} fails). ` +
    `${noSep.length ? noSep.length + ' other axes show no separation: ' + noSep.join(', ') + '.' : ''} ` +
    `<i>Method:</i> ${rq5.method}.`;
}

// --------------------------------------------------------------------------
// behavioral boundary map (data-driven 2D heatmap)
// --------------------------------------------------------------------------
const CLASS_COLORS = {
  success: '#1f7a43',
  failure: '#7a2530',
  mixed: '#7a5a1a',
  untested: '#16222f',
};
let BOUNDARY_CELLS = [];

function classifyColor(rate) {
  if (rate == null) return CLASS_COLORS.untested;
  if (rate >= 0.999) return CLASS_COLORS.success;
  if (rate <= 0.001) return CLASS_COLORS.failure;
  return CLASS_COLORS.mixed;
}

function drawBoundaryMap() {
  const bm = EVIDENCE.boundary_map;
  if (!bm?.available) { $('boundary-note').textContent = 'Boundary data unavailable.'; return; }
  const { ctx, w, h } = setupCanvas('boundary-map');
  const pad = { l: 46, r: 12, t: 12, b: 30 };
  const xs = bm.x.values; // depth noise
  const ys = bm.y.values; // settle window
  const cw = (w - pad.l - pad.r) / xs.length;
  const ch = (h - pad.t - pad.b) / ys.length;
  ctx.clearRect(0, 0, w, h);
  ctx.font = '9px sans-serif';
  BOUNDARY_CELLS = [];
  // Keys are Python float reprs like "0.0|0.0"; normalise numerically so lookups
  // match regardless of trailing ".0" formatting.
  const cellIndex = new Map();
  for (const [k, v] of Object.entries(bm.cells || {})) {
    const [ws, ns] = k.split('|');
    cellIndex.set(`${Number(ws)}|${Number(ns)}`, v);
  }
  for (let ri = 0; ri < ys.length; ri++) {
    for (let ci = 0; ci < xs.length; ci++) {
      const wy = ys[ri];
      const nx = xs[ci];
      const cell = cellIndex.get(`${Number(wy)}|${Number(nx)}`);
      const x = pad.l + ci * cw;
      const y = pad.t + ri * ch;
      const rate = cell ? cell.rate : null;
      ctx.fillStyle = cell ? classifyColor(rate) : CLASS_COLORS.untested;
      ctx.fillRect(x + 1, y + 1, cw - 2, ch - 2);
      if (!cell) {
        ctx.save();
        ctx.beginPath(); ctx.rect(x + 1, y + 1, cw - 2, ch - 2); ctx.clip();
        ctx.strokeStyle = 'rgba(126,147,173,0.28)'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let d = -ch; d < cw; d += 6) { ctx.moveTo(x + 1 + d, y + 1); ctx.lineTo(x + 1 + d + ch, y + 1 + ch); }
        ctx.stroke();
        ctx.restore();
      } else {
        ctx.fillStyle = '#dbe6f2'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText((rate * 100).toFixed(0), x + cw / 2, y + ch / 2);
      }
      BOUNDARY_CELLS.push({ x, y, w: cw, h: ch, wy, nx, cell });
    }
  }
  ctx.fillStyle = '#7e93ad'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  xs.forEach((nx, ci) => ctx.fillText(String(nx), pad.l + ci * cw + cw / 2, h - pad.b + 5));
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ys.forEach((wy, ri) => ctx.fillText(String(wy), pad.l - 5, pad.t + ri * ch + ch / 2));
  ctx.textAlign = 'center';
  ctx.fillText(bm.x.label + ' (' + bm.x.unit + ')', pad.l + (w - pad.l - pad.r) / 2, h - 8);
  ctx.save();
  ctx.translate(11, pad.t + (h - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText(bm.y.label + ' (' + bm.y.unit + ')', 0, 0);
  ctx.restore();

  $('boundary-note').innerHTML =
    'Measured over ' + bm.n_seeds + ' matched seeds · revision ' + bm.revision + '. ' +
    'Hatched = untested conditions (never inferred). Cell value = success rate (%).';

  $('boundary-legend').innerHTML = ['success', 'mixed', 'failure', 'untested']
    .map((k) => '<span><span class="sw" style="background:' + CLASS_COLORS[k] + '"></span>' + k + '</span>')
    .join('');

  const cv = $('boundary-map');
  cv.onclick = (e) => {
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const hit = BOUNDARY_CELLS.find((c) => mx >= c.x && mx <= c.x + c.w && my >= c.y && my <= c.y + c.h);
    if (hit) showCellDetail(hit);
  };
}

function showCellDetail(hit) {
  const host = $('cell-detail');
  if (!hit.cell) {
    host.innerHTML = '<div class="cd-title">Untested condition</div>' +
      '<dl class="cd-grid"><dt>settle window</dt><dd>' + hit.wy + '</dd>' +
      '<dt>depth noise</dt><dd>' + hit.nx + ' mm</dd>' +
      '<dt>status</dt><dd><span class="cls-badge cls-untested">not measured</span></dd></dl>';
    return;
  }
  const c = hit.cell;
  // the sweep varied policy_v1's settle window — label read from the evidence,
  // not hardcoded (an earlier revision claimed policy_v3, which is wrong: v3
  // *is* settle_steps=6; every swept cell is v1 with the window overridden).
  const sweepPolicy = EVIDENCE.boundary_map?.policy;
  host.innerHTML = '<div class="cd-title">settle ' + hit.wy + ' · noise ' + hit.nx + ' mm ' +
    '<span class="cls-badge cls-' + c.classification + '">' + c.classification + '</span></div>' +
    '<dl class="cd-grid">' +
    '<dt>policy</dt><dd>' + (sweepPolicy ? sweepPolicy + ' (window overridden)' : 'Not measured') + '</dd>' +
    '<dt>seeds</dt><dd>' + c.n + '</dd>' +
    '<dt>successes</dt><dd>' + c.successes + '</dd>' +
    '<dt>success rate</dt><dd>' + pct(c.rate) + ' [' + pct(c.ci[0], 0) + ', ' + pct(c.ci[1], 0) + ']</dd>' +
    '<dt>mean steps</dt><dd>' + fmt(c.steps_mean, 1) + ' [' + fmt(c.steps_ci[0], 1) + ', ' + fmt(c.steps_ci[1], 1) + ']</dd>' +
    '<dt>artifact</dt><dd>rq2_settle_window_sweep.json</dd>' +
    '</dl>';
}



// --------------------------------------------------------------------------
// evidence & artifacts panel + provenance
// --------------------------------------------------------------------------
function renderEvidence() {
  const files = EVIDENCE.evidence_files || [];
  const list = $('evidence-list');
  list.innerHTML = '';
  for (const f of files) {
    const row = el('div', 'ev-row');
    row.append(el('span', 'ev-status ' + (f.valid ? 'ok' : 'bad')));
    row.append(el('span', 'ev-name', f.title));
    row.append(el('span', 'ev-kind', f.kind));
    const link = el('a', null, 'open');
    link.href = '../' + f.path;
    link.target = '_blank';
    link.rel = 'noopener';
    link.title = f.path + ' · ' + f.detail;
    row.append(link);
    list.append(row);
  }

  const checks = EVIDENCE.consistency_checks || [];
  const failed = checks.filter((c) => !c.ok).length;
  // full check log behind a <details> — every derived value, expandable
  const log = $('checklog');
  log.querySelector('summary').textContent =
    'Consistency log (' + (checks.length - failed) + '/' + checks.length + ' passed)';
  log.querySelector('summary').title = 'Each derived dashboard value vs the value stored in the frozen JSON';
  $('checklog-body').innerHTML = checks.map((c) =>
    '<div class="check-row' + (c.ok ? '' : ' bad') + '"><span class="check-label">' + c.label + '</span>' +
    '<span class="check-v">' + (c.ok ? 'ok' : 'derived ' + c.derived + ' ≠ stored ' + c.stored) + '</span></div>'
  ).join('');
  const prov = $('provenance');
  prov.innerHTML =
    'Provenance: ' + (EVIDENCE.git?.available
      ? 'git <span class="mono">' + (EVIDENCE.git.short) + '</span> (' + EVIDENCE.git.branch + ')'
      : 'revision <span class="mono">' + EVIDENCE.revision + '</span>') +
    ' · consistency checks: <span class="checks' + (failed ? ' fail' : '') + '">' +
    (checks.length - failed) + '/' + checks.length + ' passed</span>' +
    (failed ? ' — <b>dashboard disagrees with frozen JSON</b>' : '');
}

// --------------------------------------------------------------------------
// navigation (scrolls to / focuses the relevant section)
// --------------------------------------------------------------------------
function wireNav() {
  document.querySelectorAll('.nav-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const targets = {
        overview: null,
        discovery: $('panel-discovery'),
        boundary: $('boundary-map'),
        comparison: $('policy-comparison'),
        results: $('panel-rq5'),
        logs: $('checklog'),
      };
      const t = targets[btn.dataset.view];
      if (t) t.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });
}

// --------------------------------------------------------------------------
// boot
// --------------------------------------------------------------------------
async function boot() {
  buildSkeleton();
  try {
    const res = await fetch('data/evidence.json');
    if (!res.ok) throw new Error('evidence.json HTTP ' + res.status);
    EVIDENCE = await res.json();
  } catch (err) {
    $('config-list').innerHTML = '<span class="muted">Failed to load evidence bundle: ' + err.message +
      '. Run <code>python experiments/build_dashboard_data.py</code>.</span>';
    return;
  }

  renderStatus();
  renderConfig();
  renderMetrics();
  renderPolicyComparison();
  renderAttribution();
  renderReproducibility();
  renderDiscovery();
  renderEvidence();
  wireNav();
  wireControls();

  // Draw all 2D charts first so they render even where WebGL is unavailable
  // (e.g. a headless CI browser). The robot viewport is best-effort below.
  drawBoundaryMap();
  drawSettleChart();
  drawSearchChart();
  drawRq5Chart();

  if (EVIDENCE.episode?.available) {
    REPLAY.frames = EVIDENCE.episode.frames;
    REPLAY.phases = EVIDENCE.episode.phases;
  } else {
    $('vp-telemetry').innerHTML = '<span class="muted">No recorded telemetry available for replay.</span>';
  }

  // robot viewport (best-effort: a missing/failed WebGL context must not take
  // the rest of the dashboard down with it)
  try {
    initViewport();
    const robot = await (await fetch('data/robot.json')).json();
    buildRobot(robot);
  } catch (err) {
    console.error('robot viewport unavailable', err);
    const vp = $('viewport');
    if (vp) {
      vp.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;' +
        'color:#e0565f;font-size:10px;text-align:center;padding:20px;font-family:monospace;">' +
        'Robot viewport error: ' + String(err && err.message ? err.message : err) + '</div>';
    }
  }

  let rt;
  window.addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(() => {
      drawBoundaryMap();
      drawSettleChart();
      drawSearchChart();
      drawRq5Chart();
      if (REPLAY._resize) REPLAY._resize();
    }, 150);
  });
}

boot();
