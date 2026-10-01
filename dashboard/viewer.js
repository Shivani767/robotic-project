// ROBOT CI — live Gazebo viewer.
//
// Every angle shown here is a value Gazebo published on
// /world/<world>/model/panda/joint_state. Nothing is interpolated, replayed or
// scripted in the browser: if the simulation stalls, the arm stops.
//
// Limb geometry is built from the joint origins, axes and limits in the Panda
// URDF, fetched from the bridge at /robot.json. That file is a projection of the
// same URDF Gazebo loads, so the drawing cannot drift from the simulation.

import * as THREE from 'three';

const viewport = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);

// The model is Z-up (URDF/SDF convention), so the camera's up axis is Z.
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
camera.up.set(0, 0, 1);
camera.position.set(1.5, -1.4, 1.1);
camera.lookAt(0.1, 0.1, 0.55);

scene.add(new THREE.HemisphereLight(0xdfe6ff, 0x2a3038, 1.5));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(2, -3, 4);
scene.add(key);

const grid = new THREE.GridHelper(4, 40, 0x2c323c, 0x22272e);
grid.rotation.x = Math.PI / 2; // GridHelper is XZ; the world is XY at z=0.
scene.add(grid);

// The arm hangs off this group so the whole robot can be lifted onto the table
// without disturbing the frame maths, which works in the model's own base frame.
const robotGroup = new THREE.Group();
scene.add(robotGroup);

const ARM_MATERIAL = new THREE.MeshStandardMaterial({ color: 0xd9dde3, roughness: 0.5, metalness: 0.1 });
const JOINT_MATERIAL = new THREE.MeshStandardMaterial({ color: 0x61afef, roughness: 0.4 });
const FINGER_MATERIAL = new THREE.MeshStandardMaterial({ color: 0x4ec9a0, roughness: 0.5 });
const TABLE_MATERIAL = new THREE.MeshStandardMaterial({ color: 0x4a515e, roughness: 0.9 });
const OBJECT_MATERIAL = new THREE.MeshStandardMaterial({ color: 0xe0b341, roughness: 0.6 });

// A unit-height cylinder along +Y, scaled to span each bone. Building the arm
// from primitives keeps the viewer dependency-free while the kinematics still
// come from the real model.
const UP = new THREE.Vector3(0, 1, 0);
function makeBone(radius, material) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 16), material);
  robotGroup.add(mesh);
  return mesh;
}
function makeJointSphere(radius, material) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 12), material);
  robotGroup.add(mesh);
  return mesh;
}
function makeBox(size, material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), material);
  robotGroup.add(mesh);
  return mesh;
}

// Seven arm bones, seven joint spheres, two fingers, two finger joints.
const bones = Array.from({ length: 7 }, () => makeBone(0.035, ARM_MATERIAL));
const jointSpheres = Array.from({ length: 7 }, () => makeJointSphere(0.045, JOINT_MATERIAL));
const fingers = [makeBox([0.02, 0.02, 0.06], FINGER_MATERIAL), makeBox([0.02, 0.02, 0.06], FINGER_MATERIAL)];

let tableMesh = null;
let objectMesh = null;
let kinematics = null;
let jointOrder = [];

/** Forward kinematics: world matrix for every joint frame, given live angles. */
function computeFrames(kin, angles) {
  const byParent = new Map();
  for (const j of kin.joints) {
    if (!byParent.has(j.parent)) byParent.set(j.parent, []);
    byParent.get(j.parent).push(j);
  }
  // The root is the link that no joint drives.
  const childLinks = new Set(kin.joints.map((j) => j.child));
  const root = kin.links.find((l) => !childLinks.has(l)) || kin.links[0];

  const worldOf = new Map([[root, new THREE.Matrix4()]]);
  const frames = new Map();
  const queue = [root];
  while (queue.length) {
    const link = queue.shift();
    for (const j of byParent.get(link) || []) {
      const base = worldOf.get(link).clone();
      base.multiply(new THREE.Matrix4().makeTranslation(j.xyz[0], j.xyz[1], j.xyz[2]));
      // URDF rpy is R = Rz(yaw) * Ry(pitch) * Rx(roll), i.e. Euler order 'ZYX'.
      base.multiply(new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(j.rpy[0], j.rpy[1], j.rpy[2], 'ZYX'),
      ));

      const value = angles[j.name] ?? 0;
      const axis = new THREE.Vector3(j.axis[0], j.axis[1], j.axis[2]).normalize();
      if (j.type === 'prismatic') {
        base.multiply(new THREE.Matrix4().makeTranslation(axis.x * value, axis.y * value, axis.z * value));
      } else if (j.type !== 'fixed') {
        base.multiply(new THREE.Matrix4().makeRotationAxis(axis, value));
      }

      worldOf.set(j.child, base);
      frames.set(j.name, base.clone());
      queue.push(j.child);
    }
  }
  return frames;
}

const positionOf = (m) => new THREE.Vector3().setFromMatrixPosition(m);

/** Orient a unit cylinder so it spans a -> b. */
function span(mesh, a, b) {
  const direction = b.clone().sub(a);
  const length = direction.length();
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  if (length > 1e-6) {
    mesh.quaternion.setFromUnitVectors(UP, direction.normalize());
    mesh.scale.set(1, length, 1);
  }
}

const ARM_CHAIN = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'];

/** Draw the furniture (table, target object) from the scenario's own geometry. */
function buildScene(info) {
  if (!info) return;
  robotGroup.position.z = info.table_height_m;
  if (!tableMesh) {
    tableMesh = makeBox([0.6, 0.6, 0.04], TABLE_MATERIAL);
    scene.add(tableMesh);
  }
  tableMesh.position.set(
    info.workspace_center[0], info.workspace_center[1], info.table_height_m - 0.02,
  );
  if (!objectMesh) {
    objectMesh = makeBox([0.04, 0.04, 0.04], OBJECT_MATERIAL);
    scene.add(objectMesh);
  }
  objectMesh.position.set(info.object_pose[0], info.object_pose[1], info.object_pose[2]);
}

/** Place the arm from one live joint-state snapshot. */
function applyJoints(angles) {
  if (!kinematics) return;
  const frames = computeFrames(kinematics, angles);

  const points = [new THREE.Vector3(0, 0, 0)];
  for (const name of ARM_CHAIN) {
    const frame = frames.get(name);
    if (frame) points.push(positionOf(frame));
  }
  for (let i = 0; i < bones.length && i + 1 < points.length; i += 1) {
    span(bones[i], points[i], points[i + 1]);
  }
  for (let i = 0; i < jointSpheres.length && i + 1 < points.length; i += 1) {
    jointSpheres[i].position.copy(points[i + 1]);
  }

  ['left_finger', 'right_finger'].forEach((name, index) => {
    const frame = frames.get(name);
    if (frame) fingers[index].position.copy(positionOf(frame));
  });

  updateJointTable(angles);
}

const tableBody = document.querySelector('#joints tbody');
const rows = new Map();

function buildJointTable(names) {
  tableBody.replaceChildren();
  rows.clear();
  for (const name of names) {
    const tr = document.createElement('tr');
    const label = document.createElement('td');
    label.textContent = name;
    const value = document.createElement('td');
    value.className = 'value';
    const barCell = document.createElement('td');
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('span');
    bar.appendChild(fill);
    barCell.appendChild(bar);
    tr.append(label, value, barCell);
    tableBody.appendChild(tr);
    rows.set(name, { value, fill });
  }
}

function updateJointTable(angles) {
  for (const [name, row] of rows) {
    const v = angles[name] ?? 0;
    row.value.textContent = v.toFixed(4);
    // Bars are drawn against each joint's own URDF limits, not a shared scale,
    // so a small finger travel stays legible next to a 3 rad wrist sweep.
    const joint = kinematics?.joints.find((j) => j.name === name);
    const lo = joint ? joint.lower : -3;
    const hi = joint ? joint.upper : 3;
    const span = hi - lo || 1;
    const frac = Math.min(1, Math.max(0, (v - lo) / span));
    row.fill.style.left = '0%';
    row.fill.style.right = `${(1 - frac) * 100}%`;
  }
}

const statusList = document.getElementById('status');

function renderStatus(message) {
  const ctx = message.context;
  const health = message.health;
  const fields = [
    ['simulator', 'Gazebo Harmonic'],
    ['task', ctx.task],
    ['policy', ctx.policy],
    ['seed', ctx.seed],
    ['depth noise', `${ctx.depth_noise_mm} mm`],
    ['condition', ctx.condition],
    ['config hash', (ctx.config_hash || '—').slice(0, 12)],
    ['phase', ctx.phase],
    ['sim time', `${message.sim_time.toFixed(3)} s`],
    ['joint messages', health.joint_messages],
    ['camera frames', health.camera_frames],
    ['joint stream', health.joint_stream ? 'live' : 'waiting'],
    ['camera stream', health.camera_stream ? 'live' : 'waiting'],
  ];
  statusList.replaceChildren(
    ...fields.flatMap(([key, value]) => {
      const dt = document.createElement('dt');
      dt.textContent = key;
      const dd = document.createElement('dd');
      dd.textContent = value;
      return [dt, dd];
    }),
  );
  document.getElementById('sim-clock').textContent = `sim t = ${message.sim_time.toFixed(3)} s`;
}

const badge = document.getElementById('connection');

function connectState() {
  const socket = new WebSocket(`ws://${location.hostname}:${Number(location.port) + 1}/ws/state`);
  socket.onopen = () => {
    badge.textContent = 'live';
    badge.className = 'badge ok';
  };
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    applyJoints(message.joints);
    renderStatus(message);
  };
  socket.onclose = () => {
    badge.textContent = 'disconnected';
    badge.className = 'badge bad';
    setTimeout(connectState, 1000);
  };
}

const cameraImg = document.getElementById('camera');
const cameraHint = document.getElementById('camera-hint');
let previousUrl = null;

function connectCamera() {
  const socket = new WebSocket(`ws://${location.hostname}:${Number(location.port) + 1}/ws/camera`);
  socket.binaryType = 'blob';
  socket.onmessage = (event) => {
    const url = URL.createObjectURL(event.data);
    cameraImg.src = url;
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    previousUrl = url;
    cameraHint.textContent = 'live from Gazebo (rgbd_camera)';
  };
  socket.onclose = () => setTimeout(connectCamera, 1000);
}

function resize() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

fetch('robot.json')
  .then((response) => response.json())
  .then((info) => {
    kinematics = info;
    buildScene(info.scene);
    buildJointTable(info.joints.map((joint) => joint.name));
    applyJoints({});
  })
  .catch((error) => {
    badge.textContent = 'robot.json unavailable';
    badge.className = 'badge bad';
    console.error(error);
  });

resize();
animate();
connectState();
connectCamera();
