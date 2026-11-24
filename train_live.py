#!/usr/bin/env python3
"""
train_live.py
=============
Executable live-training script with a real-time interactive Three.js 3D viewer
in your web browser.

Defaults to HashSCKNetwork (Instant-NGP Triton 3D Hash Grid), trains at 100%
native GPU speed on the main process, and serves live novel-view 360-degree
point cloud renders over a lightweight stdlib HTTP server (zero extra packages,
zero VTK/GIL contention, works locally or over SSH port-forwarding).

Usage:
  ./train_live.py                        # HashSCK, vis every 0.25 epochs, port 8765
  ./train_live.py --vis-every 1.0        # Update once per epoch
  ./train_live.py --vis-every 0.1        # Update 10x per epoch
  ./train_live.py --model fourier        # Compare with Fourier baseline
  ./train_live.py --compile              # Enable torch.compile (~19% faster after JIT)
"""

import argparse
import json
import logging
import struct
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from generate_training_data import load_scene_from_config
from hash_sck_model import HashSCKNetwork, compute_hash_sck_loss
from sck_model import (
    DeeperSCKNetwork,
    GPURayBatcher,
    compute_sck_loss,
    vectorized_gpu_origin_slide,
)
from synthetic_data_generation_utils import LidarProjectionModel

# Suppress benign TorchInductor SM-count warning on mobile GPUs
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)


# ==============================================================================
# 1. Embedded Three.js Interactive 3D Viewer HTML
# ==============================================================================

VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LiZARD Live 3D Training Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; overflow: hidden; background: #0e1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
  #canvas-container { width: 100%; height: 100%; position: relative; }
  #hud {
    position: absolute; top: 14px; left: 14px; width: 310px;
    background: rgba(15, 19, 27, 0.86); backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 10px;
    padding: 14px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    user-select: none; z-index: 10; font-size: 12px;
  }
  .hud-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .title { font-size: 14px; font-weight: 700; letter-spacing: 0.5px; color: #f0f6fc; }
  .badge {
    font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 999px;
    background: rgba(56, 139, 253, 0.22); color: #79c0ff; border: 1px solid rgba(56, 139, 253, 0.4);
  }
  .badge.paused { background: rgba(210, 153, 34, 0.25); color: #e3b341; border-color: rgba(210, 153, 34, 0.45); }
  .badge.done { background: rgba(46, 160, 67, 0.25); color: #56d364; border-color: rgba(46, 160, 67, 0.45); }
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px; margin-bottom: 10px; }
  .stat { background: rgba(255, 255, 255, 0.04); padding: 6px 8px; border-radius: 6px; }
  .stat-label { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.4px; }
  .stat-value { font-size: 13px; font-weight: 600; color: #f0f6fc; margin-top: 2px; }
  #loss-canvas {
    width: 100%; height: 64px; background: rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px;
    display: block; margin-bottom: 10px;
  }
  .controls { display: flex; flex-direction: column; gap: 8px; }
  .btn-row { display: flex; gap: 6px; }
  button {
    flex: 1; padding: 6px 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.07); color: #e6edf3; font-size: 11px; font-weight: 600;
    cursor: pointer; transition: background 0.15s;
  }
  button:hover { background: rgba(255,255,255,0.15); }
  button.primary { background: rgba(56, 139, 253, 0.28); border-color: rgba(56, 139, 253, 0.5); color: #a5d6ff; }
  button.primary:hover { background: rgba(56, 139, 253, 0.42); }
  button.danger { background: rgba(248, 81, 73, 0.2); border-color: rgba(248, 81, 73, 0.4); color: #ffa198; }
  button.danger:hover { background: rgba(248, 81, 73, 0.35); }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .row label { color: #8b949e; font-size: 11px; }
  select, input[type="range"] {
    background: rgba(255,255,255,0.08); color: #e6edf3; border: 1px solid rgba(255,255,255,0.14);
    border-radius: 5px; padding: 3px 6px; font-size: 11px;
  }
  input[type="range"] { padding: 0; width: 120px; accent-color: #58a6ff; }
  .footer-hint { margin-top: 8px; font-size: 10px; color: #6e7681; line-height: 1.4; }
  .cbar { height: 7px; border-radius: 4px; margin-top: 4px; background: linear-gradient(90deg, #0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921); }
  .cbar-labels { display: flex; justify-content: space-between; font-size: 9px; color: #8b949e; margin-top: 2px; }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<div id="canvas-container"></div>
<div id="hud">
  <div class="hud-header">
    <div class="title" id="hud-title">LiZARD Live</div>
    <span class="badge" id="status-badge">Connecting...</span>
  </div>
  <div class="stats-grid">
    <div class="stat"><div class="stat-label">Epoch</div><div class="stat-value" id="st-epoch">0.00 / --</div></div>
    <div class="stat"><div class="stat-label">Loss</div><div class="stat-value" id="st-loss">--</div></div>
    <div class="stat"><div class="stat-label">Throughput</div><div class="stat-value" id="st-rps">--</div></div>
    <div class="stat"><div class="stat-label">Rendered Pts</div><div class="stat-value" id="st-pts">0</div></div>
  </div>
  <canvas id="loss-canvas" width="280" height="64"></canvas>
  <div class="controls">
    <div class="btn-row">
      <button id="btn-pause" class="primary">Pause (Space)</button>
      <button id="btn-reset" class="danger">Reset (R)</button>
    </div>
    <div class="btn-row">
      <button id="btn-mode">Mode: Sample (M)</button>
      <button id="btn-gt">GT Scene: On (G)</button>
    </div>
    <div class="row">
      <label for="sel-color">Color by</label>
      <select id="sel-color">
        <option value="range" selected>Range (m)</option>
        <option value="z">Height Z (m)</option>
        <option value="prob">Return Prob</option>
      </select>
    </div>
    <div class="row">
      <label for="rng-size">Point size (<span id="lbl-size">0.07</span>m)</label>
      <input id="rng-size" type="range" min="0.02" max="0.22" step="0.01" value="0.07" />
    </div>
    <div>
      <div class="cbar"></div>
      <div class="cbar-labels"><span id="cbar-min">0.0</span><span id="cbar-mid">Range (m)</span><span id="cbar-max">8.0</span></div>
    </div>
    <div class="btn-row">
      <button id="btn-cam">Reset Camera</button>
    </div>
  </div>
  <div class="footer-hint">Left-drag: Orbit &bull; Right-drag: Pan &bull; Scroll: Zoom</div>
</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// LiDAR world frame is Z-up
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1117);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.05, 200);
camera.up.set(0, 0, 1);
const DEFAULT_CAM_POS = new THREE.Vector3(10.5, -10.5, 8.8);
const DEFAULT_CAM_TGT = new THREE.Vector3(0, 0, 1.2);
camera.position.copy(DEFAULT_CAM_POS);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.copy(DEFAULT_CAM_TGT);
controls.update();

// Lighting for translucent GT geometry
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(10, -8, 15);
scene.add(dirLight);

// Reference floor grid (rotated into XY plane for Z-up) + axes
const grid = new THREE.GridHelper(16, 16, 0x30363d, 0x1c2128);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.01;
scene.add(grid);
scene.add(new THREE.AxesHelper(1.2));

// Ground-truth scene group + sensor marker
const gtGroup = new THREE.Group();
scene.add(gtGroup);
let gtVisible = true;

const sensorMarker = new THREE.Mesh(
  new THREE.SphereGeometry(0.11, 20, 16),
  new THREE.MeshStandardMaterial({ color: 0xff5a4f, emissive: 0x661510 })
);
scene.add(sensorMarker);

// Live point cloud object (pre-allocated dynamic BufferGeometry)
const cloudGeom = new THREE.BufferGeometry();
const cloudMat = new THREE.PointsMaterial({
  size: 0.07,
  vertexColors: true,
  sizeAttenuation: true,
});
const cloudMesh = new THREE.Points(cloudGeom, cloudMat);
cloudMesh.frustumCulled = false;
scene.add(cloudMesh);

// Plasma colormap LUT (256 entries)
const PLASMA_STOPS = [
  [0.0,  13,   8, 135],
  [0.2, 106,   0, 168],
  [0.4, 177,  42, 144],
  [0.6, 225, 100,  98],
  [0.8, 252, 166,  54],
  [1.0, 240, 249,  33],
];
const PLASMA_LUT = new Float32Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  let s0 = PLASMA_STOPS[0], s1 = PLASMA_STOPS[PLASMA_STOPS.length - 1];
  for (let j = 0; j < PLASMA_STOPS.length - 1; j++) {
    if (t >= PLASMA_STOPS[j][0] && t <= PLASMA_STOPS[j + 1][0]) {
      s0 = PLASMA_STOPS[j]; s1 = PLASMA_STOPS[j + 1]; break;
    }
  }
  const a = (t - s0[0]) / Math.max(1e-6, s1[0] - s0[0]);
  PLASMA_LUT[i * 3 + 0] = (s0[1] + a * (s1[1] - s0[1])) / 255;
  PLASMA_LUT[i * 3 + 1] = (s0[2] + a * (s1[2] - s0[2])) / 255;
  PLASMA_LUT[i * 3 + 2] = (s0[3] + a * (s1[3] - s0[3])) / 255;
}

// State for client-side recoloring
let lastPts = null, lastRanges = null, lastProbs = null;
let currentSMax = 8.0;
let colorMode = 'range';
let lastVersion = -1;

function applyColors() {
  if (!lastPts || lastPts.length === 0) return;
  const n = lastPts.length / 3;
  const colors = new Float32Array(n * 3);
  let vmin = 0.0, vmax = currentSMax, label = 'Range (m)';
  if (colorMode === 'z') { vmin = 0.0; vmax = 3.0; label = 'Height Z (m)'; }
  else if (colorMode === 'prob') { vmin = 0.0; vmax = 1.0; label = 'Return Prob'; }

  document.getElementById('cbar-min').textContent = vmin.toFixed(1);
  document.getElementById('cbar-mid').textContent = label;
  document.getElementById('cbar-max').textContent = vmax.toFixed(1);

  const inv = 1.0 / Math.max(1e-6, vmax - vmin);
  for (let i = 0; i < n; i++) {
    const val = colorMode === 'z' ? lastPts[i * 3 + 2]
              : colorMode === 'prob' ? lastProbs[i]
              : lastRanges[i];
    const idx = Math.max(0, Math.min(255, Math.round((val - vmin) * inv * 255)));
    colors[i * 3 + 0] = PLASMA_LUT[idx * 3 + 0];
    colors[i * 3 + 1] = PLASMA_LUT[idx * 3 + 1];
    colors[i * 3 + 2] = PLASMA_LUT[idx * 3 + 2];
  }
  cloudGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

function drawSparkline(history) {
  const cvs = document.getElementById('loss-canvas');
  const ctx = cvs.getContext('2d');
  const W = cvs.width, H = cvs.height;
  ctx.clearRect(0, 0, W, H);
  if (!history || history.length < 2) {
    ctx.fillStyle = '#6e7681';
    ctx.font = '10px monospace';
    ctx.fillText('Waiting for loss curve...', 10, H / 2 + 3);
    return;
  }
  const vals = history.filter(v => Number.isFinite(v));
  if (vals.length < 2) return;
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const span = Math.max(1e-5, maxV - minV);
  const pad = 8;

  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (let i = 0; i < vals.length; i++) {
    const x = pad + (i / (vals.length - 1)) * (W - 2 * pad);
    const y = (H - pad) - ((vals[i] - minV) / span) * (H - 2 * pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = '#8b949e';
  ctx.font = '9px monospace';
  ctx.fillText(`max ${maxV.toFixed(2)}`, 5, 11);
  ctx.fillText(`min ${minV.toFixed(2)}`, 5, H - 4);
}

// Load scene geometry once from /api/scene
async function loadScene() {
  try {
    const resp = await fetch('/api/scene');
    const data = await resp.json();
    if (data.render_pos) sensorMarker.position.set(...data.render_pos);
    if (data.s_max) currentSMax = data.s_max;
    if (data.model) document.getElementById('hud-title').textContent = `LiZARD Live (${data.model.toUpperCase()} SCK)`;
    if (data.no_gt_scene) {
      gtVisible = false;
      gtGroup.visible = false;
      document.getElementById('btn-gt').textContent = 'GT Scene: Off (G)';
    }
    if (data.color_by) {
      colorMode = data.color_by === 'probability' ? 'prob' : data.color_by;
      document.getElementById('sel-color').value = colorMode;
    }

    const palette = [0x4ea8de, 0x56cfe1, 0x64dfdf, 0x80ffdb, 0xf77f00, 0xfcbf49, 0xe63946, 0xa8dadc];
    (data.scene_objects || []).forEach((obj, idx) => {
      const color = obj.density >= 50 ? 0x3b4859 : palette[idx % palette.length];
      const opacity = obj.density >= 50 ? 0.12 : 0.22;
      const mat = new THREE.MeshStandardMaterial({
        color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide
      });
      const edgeMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 });

      let geom = null;
      if (obj.type === 'box') {
        geom = new THREE.BoxGeometry(obj.size[0], obj.size[1], obj.size[2]);
      } else if (obj.type === 'sphere') {
        geom = new THREE.SphereGeometry(obj.radius, 24, 18);
      } else if (obj.type === 'cylinder') {
        geom = new THREE.CylinderGeometry(obj.radius, obj.radius, obj.height, 24);
      }
      if (!geom) return;

      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(...obj.center);

      if (obj.type === 'cylinder') {
        const axis = new THREE.Vector3(...(obj.axis || [0, 0, 1])).normalize();
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
      } else if (obj.type === 'box' && obj.rotation) {
        const r = obj.rotation;
        const m4 = new THREE.Matrix4().set(
          r[0][0], r[0][1], r[0][2], 0,
          r[1][0], r[1][1], r[1][2], 0,
          r[2][0], r[2][1], r[2][2], 0,
          0, 0, 0, 1
        );
        mesh.quaternion.setFromRotationMatrix(m4);
      }

      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geom), edgeMat);
      mesh.add(edges);
      gtGroup.add(mesh);
    });
  } catch (e) {
    console.error('Failed to load scene:', e);
  }
}

// Poll binary frames from /api/frame
async function pollFrame() {
  try {
    const resp = await fetch(`/api/frame?v=${lastVersion}`, { cache: 'no-store' });
    if (resp.status === 200) {
      const buf = await resp.arrayBuffer();
      const dv = new DataView(buf);
      const metaLen = dv.getUint32(0, true);
      const metaStr = new TextDecoder().decode(new Uint8Array(buf, 4, metaLen));
      const meta = JSON.parse(metaStr);

      let offset = 4 + metaLen;
      offset = (offset + 3) & ~3; // 4-byte alignment
      const n = meta.num_points || 0;
      const pts = new Float32Array(buf, offset, n * 3);
      const rngs = new Float32Array(buf, offset + n * 12, n);
      const probs = new Float32Array(buf, offset + n * 16, n);

      lastVersion = meta.version;
      currentSMax = meta.s_max || currentSMax;
      lastPts = pts;
      lastRanges = rngs;
      lastProbs = probs;

      cloudGeom.setAttribute('position', new THREE.BufferAttribute(pts, 3));
      applyColors();
      cloudGeom.computeBoundingSphere();

      // Update HUD
      document.getElementById('st-epoch').textContent = `${meta.epoch_float.toFixed(2)} / ${meta.total_epochs}`;
      document.getElementById('st-loss').textContent = Number.isFinite(meta.loss) ? meta.loss.toFixed(4) : '--';
      document.getElementById('st-rps').textContent = meta.rays_per_sec > 0 ? `${Math.round(meta.rays_per_sec).toLocaleString()}/s` : '--';
      document.getElementById('st-pts').textContent = `${n.toLocaleString()} (${Math.round(meta.render_ms)}ms)`;

      const badge = document.getElementById('status-badge');
      badge.className = 'badge';
      if (meta.paused) { badge.textContent = 'Paused'; badge.classList.add('paused'); }
      else if ((meta.status || '').startsWith('Complete')) { badge.textContent = 'Complete'; badge.classList.add('done'); }
      else { badge.textContent = meta.status || 'Training'; }

      document.getElementById('btn-pause').textContent = meta.paused ? 'Resume (Space)' : 'Pause (Space)';
      document.getElementById('btn-mode').textContent = `Mode: ${meta.render_method === 'threshold' ? 'Threshold' : 'Sample'} (M)`;
      drawSparkline(meta.loss_history);
    }
  } catch (e) {
    // Server restarting or unreachable
  }
  setTimeout(pollFrame, 100);
}

async function sendControl(action) {
  try {
    await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });
  } catch (e) {}
}

document.getElementById('btn-pause').addEventListener('click', () => sendControl('toggle_pause'));
document.getElementById('btn-reset').addEventListener('click', () => sendControl('reset'));
document.getElementById('btn-mode').addEventListener('click', () => sendControl('toggle_method'));
document.getElementById('btn-gt').addEventListener('click', () => {
  gtVisible = !gtVisible;
  gtGroup.visible = gtVisible;
  document.getElementById('btn-gt').textContent = `GT Scene: ${gtVisible ? 'On' : 'Off'} (G)`;
});
document.getElementById('sel-color').addEventListener('change', (e) => {
  colorMode = e.target.value;
  applyColors();
});
document.getElementById('rng-size').addEventListener('input', (e) => {
  const s = parseFloat(e.target.value);
  cloudMat.size = s;
  document.getElementById('lbl-size').textContent = s.toFixed(2);
});
document.getElementById('btn-cam').addEventListener('click', () => {
  camera.position.copy(DEFAULT_CAM_POS);
  controls.target.copy(DEFAULT_CAM_TGT);
  controls.update();
});

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (k === ' ' || k === 'p') { e.preventDefault(); sendControl('toggle_pause'); }
  else if (k === 'r') sendControl('reset');
  else if (k === 'm') sendControl('toggle_method');
  else if (k === 'g') document.getElementById('btn-gt').click();
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

loadScene();
pollFrame();
animate();
</script>
</body>
</html>
"""


# ==============================================================================
# 2. Data Loading & Fast GPU Novel-View Rendering
# ==============================================================================

def load_training_data(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], LidarProjectionModel]:
    """Loads training rays from <data_dir>/training_scans.npz (or scans/*.npz)."""
    consolidated = data_dir / "training_scans.npz"
    scans_dir = data_dir / "scans"

    if consolidated.exists():
        npz = np.load(consolidated)
        rays_o = npz["rays_o"].astype(np.float32)
        rays_d = npz["rays_d"].astype(np.float32)
        hit_ranges = npz["hit_ranges"].astype(np.float32)
        hit_masks = npz["hit_masks"].astype(np.float32)
    elif scans_dir.exists() and any(scans_dir.glob("scan_*.npz")):
        scan_files = sorted(scans_dir.glob("scan_*.npz"))
        parts_o, parts_d, parts_r, parts_m = [], [], [], []
        for sf in scan_files:
            d = np.load(sf)
            parts_o.append(d["rays_o"].astype(np.float32))
            parts_d.append(d["rays_d"].astype(np.float32))
            parts_r.append(d["hit_ranges"].astype(np.float32))
            parts_m.append(d["hit_masks"].astype(np.float32))
        rays_o = np.concatenate(parts_o, axis=0)
        rays_d = np.concatenate(parts_d, axis=0)
        hit_ranges = np.concatenate(parts_r, axis=0)
        hit_masks = np.concatenate(parts_m, axis=0)
    else:
        raise FileNotFoundError(
            f"No training scans found in '{data_dir.resolve()}'. "
            "Run './generate_training_data.py' first to generate synthetic scans."
        )

    config_path = data_dir / "scene_config.json"
    if config_path.exists():
        _, sensor, cfg = load_scene_from_config(config_path)
    else:
        cfg = {"scene_objects": []}
        sensor = LidarProjectionModel(
            elevation_min=-25.0, elevation_max=15.0, num_detectors=32, num_azimuth=180
        )

    return rays_o, rays_d, hit_ranges, hit_masks, cfg, sensor


@torch.no_grad()
def render_cloud_fast_gpu(
    model: torch.nn.Module,
    render_rays_o: torch.Tensor,
    render_rays_d: torch.Tensor,
    method: str = "sample",
    threshold: float = 0.5,
    non_return_threshold: float = 0.1,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast GPU novel-view point cloud render using pre-cached ray tensors.

    Unwraps torch.compile (_orig_mod) so odd render batch sizes never trigger
    TorchInductor recompilation.
    """
    eval_model = getattr(model, "_orig_mod", model)
    was_training = eval_model.training
    eval_model.eval()

    total_rays = render_rays_o.shape[0]
    pts_list: List[torch.Tensor] = []
    ranges_list: List[torch.Tensor] = []
    probs_list: List[torch.Tensor] = []

    for i in range(0, total_rays, batch_size):
        bo = render_rays_o[i : i + batch_size]
        bd = render_rays_d[i : i + batch_size]
        B = bo.shape[0]

        s_knots, c_knots, _, p_hit = eval_model(bo, bd)
        K = s_knots.shape[1]

        s_ext = F.pad(s_knots, (1, 0), value=0.0)
        c_ext = F.pad(c_knots, (1, 0), value=0.0)
        c_max = c_ext[:, -1]

        p_non_return = (1.0 - p_hit).squeeze(-1)
        ray_hit_mask = p_non_return <= non_return_threshold

        if method == "threshold":
            target = torch.full((B, 1), threshold, device=bo.device)
            valid = ray_hit_mask & (c_max >= threshold)
        else:
            u = torch.rand((B, 1), device=bo.device)
            valid = ray_hit_mask & (u.squeeze(-1) <= c_max)
            target = u

        idx = torch.searchsorted(c_ext.contiguous(), target.contiguous(), right=True)
        idx = torch.clamp(idx, 1, K)
        idx_left = idx - 1

        s_0 = torch.gather(s_ext, 1, idx_left).squeeze(-1)
        s_1 = torch.gather(s_ext, 1, idx).squeeze(-1)
        c_0 = torch.gather(c_ext, 1, idx_left).squeeze(-1)
        c_1 = torch.gather(c_ext, 1, idx).squeeze(-1)

        alpha = torch.clamp((target.squeeze(-1) - c_0) / torch.clamp(c_1 - c_0, min=1e-5), 0.0, 1.0)
        pred_ranges = s_0 + alpha * (s_1 - s_0)
        pts = bo + pred_ranges.unsqueeze(-1) * bd

        if valid.any():
            pts_list.append(pts[valid])
            ranges_list.append(pred_ranges[valid])
            probs_list.append(c_max[valid])

    if was_training:
        eval_model.train()

    if not pts_list:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        torch.cat(pts_list, dim=0).cpu().numpy().astype(np.float32, copy=False),
        torch.cat(ranges_list, dim=0).cpu().numpy().astype(np.float32, copy=False),
        torch.cat(probs_list, dim=0).cpu().numpy().astype(np.float32, copy=False),
    )


# ==============================================================================
# 3. Shared Binary Frame Store + Stdlib HTTP Server
# ==============================================================================

class LiveServerState:
    """Thread-safe container holding the latest pre-packed binary point cloud frame."""

    def __init__(self, scene_payload: Dict[str, Any], args: argparse.Namespace):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.reset_event = threading.Event()
        self.rerender_event = threading.Event()

        self.scene_json = json.dumps(scene_payload).encode("utf-8")
        self.render_method = args.render_method
        self.s_max = float(args.s_max)
        self.version = 0
        self.loss_history: List[float] = []
        self.latest_blob: Optional[bytes] = None
        self.latest_meta: Dict[str, Any] = {
            "version": 0,
            "epoch_float": 0.0,
            "total_epochs": args.epochs,
            "step": 0,
            "loss": None,
            "loss_history": [],
            "rays_per_sec": 0.0,
            "render_ms": 0.0,
            "status": "Initializing...",
            "paused": False,
            "render_method": args.render_method,
            "num_points": 0,
            "s_max": self.s_max,
        }
        self._last_pts = np.empty((0, 3), dtype=np.float32)
        self._last_rngs = np.empty((0,), dtype=np.float32)
        self._last_probs = np.empty((0,), dtype=np.float32)

    def _pack_blob(
        self, meta: Dict[str, Any], pts: np.ndarray, rngs: np.ndarray, probs: np.ndarray
    ) -> bytes:
        meta_bytes = json.dumps(meta).encode("utf-8")
        pad = (-len(meta_bytes)) % 4
        header = struct.pack("<I", len(meta_bytes)) + meta_bytes + (b"\x00" * pad)
        return header + pts.tobytes() + rngs.tobytes() + probs.tobytes()

    def publish(
        self,
        pts: np.ndarray,
        rngs: np.ndarray,
        probs: np.ndarray,
        epoch_float: float,
        total_epochs: int,
        step: int,
        loss_val: float,
        rays_per_sec: float,
        render_ms: float,
        status: str,
        append_loss: bool = True,
    ) -> None:
        with self.lock:
            self.version += 1
            if append_loss and np.isfinite(loss_val):
                self.loss_history.append(float(loss_val))
                if len(self.loss_history) > 300:
                    self.loss_history = self.loss_history[-300:]

            meta = {
                "version": self.version,
                "epoch_float": float(epoch_float),
                "total_epochs": int(total_epochs),
                "step": int(step),
                "loss": float(loss_val) if np.isfinite(loss_val) else None,
                "loss_history": list(self.loss_history),
                "rays_per_sec": float(rays_per_sec),
                "render_ms": float(render_ms),
                "status": status,
                "paused": self.pause_event.is_set(),
                "render_method": self.render_method,
                "num_points": int(len(pts)),
                "s_max": self.s_max,
            }
            self._last_pts = pts
            self._last_rngs = rngs
            self._last_probs = probs
            self.latest_meta = meta
            self.latest_blob = self._pack_blob(meta, pts, rngs, probs)

    def bump_meta_only(self) -> None:
        """Repacks latest cloud with updated pause/method flags so UI reflects clicks instantly."""
        with self.lock:
            if self.latest_blob is None:
                return
            self.version += 1
            meta = dict(self.latest_meta)
            meta["version"] = self.version
            meta["paused"] = self.pause_event.is_set()
            meta["render_method"] = self.render_method
            self.latest_meta = meta
            self.latest_blob = self._pack_blob(meta, self._last_pts, self._last_rngs, self._last_probs)

    def get_frame_if_newer(self, client_version: int) -> Optional[bytes]:
        with self.lock:
            if self.latest_blob is None or self.version == client_version:
                return None
            return self.latest_blob


def start_http_server(state: LiveServerState, host: str, port: int) -> ThreadingHTTPServer:
    html_bytes = VIEWER_HTML.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return  # Keep terminal clean for training logs

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
            elif parsed.path == "/api/scene":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(state.scene_json)))
                self.end_headers()
                self.wfile.write(state.scene_json)
            elif parsed.path == "/api/frame":
                qs = urllib.parse.parse_qs(parsed.query)
                try:
                    client_v = int(qs.get("v", ["-1"])[0])
                except ValueError:
                    client_v = -1
                blob = state.get_frame_if_newer(client_v)
                if blob is None:
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(blob)))
                    self.end_headers()
                    self.wfile.write(blob)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/api/control":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                cmd = json.loads(body.decode("utf-8")).get("action", "")
            except Exception:
                cmd = ""

            if cmd == "toggle_pause":
                if state.pause_event.is_set():
                    state.pause_event.clear()
                else:
                    state.pause_event.set()
                state.bump_meta_only()
            elif cmd == "toggle_method":
                with state.lock:
                    state.render_method = "threshold" if state.render_method == "sample" else "sample"
                state.rerender_event.set()
                state.bump_meta_only()
            elif cmd == "reset":
                state.reset_event.set()

            resp = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ==============================================================================
# 4. Main Training Loop (100% Native Speed on Main Thread)
# ==============================================================================

def make_optimizer_and_scheduler(
    model: torch.nn.Module, args: argparse.Namespace, device: torch.device
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    fused_ok = device.type == "cuda"
    if args.model == "hash":
        orig = getattr(model, "_orig_mod", model)
        optimizer = torch.optim.Adam(
            orig.parameter_groups(lr_mlp=args.lr_mlp, lr_hash=args.lr_hash),
            eps=1e-15,
            fused=fused_ok,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.lr_mlp * 0.05
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_fourier, fused=fused_ok)
        scheduler = None
    return optimizer, scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LiZARD SCK models with a real-time Three.js 3D browser viewer."
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing training_scans.npz / scans/ (default: data)")
    parser.add_argument("--model", type=str, choices=["hash", "fourier"], default="hash", help="Model architecture (default: hash)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument(
        "--vis-every",
        type=float,
        default=0.25,
        help="Epochs between visual updates; supports floats < 1.0 for sub-epoch updates (default: 0.25)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 2048 for hash, 1024 for fourier)")
    parser.add_argument("--s-max", type=float, default=8.0, help="Max ray range in meters (default: 8.0)")
    parser.add_argument("--num-knots", type=int, default=32, help="Number of spline knots K per ray (default: 32)")
    parser.add_argument("--num-rounds", type=int, default=None, help="Self-conditioning rounds R (default: 2 for hash, 3 for fourier)")
    parser.add_argument("--hidden-dim", type=int, default=None, help="MLP hidden dim (default: 64 for hash, 256 for fourier)")
    parser.add_argument("--cdf-mode", type=str, choices=["softmax", "hazard"], default="softmax", help="CDF parameterization (default: softmax)")
    parser.add_argument("--joint-cdf", action="store_true", help="(Hash model) Condition CDF head on ray_feat instead of pure 3D local readout")
    parser.add_argument("--lr-mlp", type=float, default=1e-3, help="(Hash model) MLP learning rate (default: 1e-3)")
    parser.add_argument("--lr-hash", type=float, default=1e-2, help="(Hash model) Hash table learning rate (default: 1e-2)")
    parser.add_argument("--lr-fourier", type=float, default=5e-4, help="(Fourier model) Learning rate (default: 5e-4)")
    parser.add_argument("--lambda-raydrop", type=float, default=0.1, help="Ray-drop BCE loss weight (default: 0.1)")
    parser.add_argument("--lambda-sky-cdf", type=float, default=0.1, help="(Hash model) Sky empty-space carving weight (default: 0.1)")
    parser.add_argument("--origin-slide", action="store_true", help="Enable vectorized GPU origin-sliding augmentation")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (adds one-time JIT compile at Epoch 1, ~19%% faster after)")

    # Viewer & server options
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP server bind address (default: 127.0.0.1; use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser tab")
    parser.add_argument("--exit-on-finish", action="store_true", help="Exit immediately when training finishes instead of keeping viewer alive")
    parser.add_argument("--render-pos", type=float, nargs=3, default=[0.0, 0.0, 1.5], metavar=("X", "Y", "Z"), help="Novel viewpoint position (default: 0.0 0.0 1.5)")
    parser.add_argument("--render-azimuth", type=int, default=360, help="Azimuthal rays for live novel-view render (default: 360)")
    parser.add_argument("--render-method", type=str, choices=["sample", "threshold"], default="sample", help="CDF point extraction mode (default: sample)")
    parser.add_argument("--threshold", type=float, default=0.5, help="CDF threshold when --render-method=threshold (default: 0.5)")
    parser.add_argument("--non-return-threshold", type=float, default=0.1, help="Prune ray if P(non-return) > threshold (default: 0.1)")
    parser.add_argument("--color-by", type=str, choices=["range", "z", "probability"], default="range", help="Initial point color scalar (default: range)")
    parser.add_argument("--no-gt-scene", action="store_true", help="Hide translucent ground-truth room overlay on startup")
    parser.add_argument("--save-checkpoint", type=str, default=None, help="Optional path to save trained model weights (.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vis_every <= 0.0:
        raise ValueError("--vis-every must be strictly positive (e.g. 0.25 or 1.0).")

    if args.batch_size is None:
        args.batch_size = 2048 if args.model == "hash" else 1024
    if args.num_rounds is None:
        args.num_rounds = 2 if args.model == "hash" else 3
    if args.hidden_dim is None:
        args.hidden_dim = 64 if args.model == "hash" else 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    # 1. Load training data from ./data/
    data_dir = Path(args.data_dir)
    rays_o, rays_d, hit_ranges, hit_masks, scene_cfg, train_sensor = load_training_data(data_dir)
    print(
        f"Loaded {len(rays_o):,} training rays from '{data_dir}' "
        f"({(hit_masks > 0).mean() * 100:.1f}% hits)",
        flush=True,
    )

    dataloader = GPURayBatcher(
        rays_o=rays_o,
        rays_d=rays_d,
        hit_ranges=hit_ranges,
        hit_masks=hit_masks,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
    )
    steps_per_epoch = max(1, len(dataloader))
    vis_every_steps = max(1, int(round(args.vis_every * steps_per_epoch)))
    print(
        f"Schedule: {args.epochs} epochs x {steps_per_epoch} batches/epoch (B={args.batch_size}) | "
        f"Visual update every {args.vis_every} epoch(s) (= {vis_every_steps} batches)",
        flush=True,
    )

    # 2. Build model (defaults to HashSCKNetwork)
    if args.model == "hash":
        model: torch.nn.Module = HashSCKNetwork(
            num_knots=args.num_knots,
            s_min=0.1,
            s_max=args.s_max,
            hidden_dim=args.hidden_dim,
            num_hidden_blocks=1,
            num_rounds=args.num_rounds,
            cdf_mode=args.cdf_mode,
            local_cdf=not args.joint_cdf,
            scene_bound=args.s_max,
        ).to(device)
    else:
        model = DeeperSCKNetwork(
            num_knots=args.num_knots,
            s_min=0.1,
            s_max=args.s_max,
            hidden_dim=args.hidden_dim,
            num_hidden_blocks=2,
            num_rounds=args.num_rounds,
            cdf_mode=args.cdf_mode,
        ).to(device)

    if args.compile:
        print("Wrapping model with torch.compile() (first step incurs one-time JIT compile)...", flush=True)
        model = torch.compile(model)

    # 3. Pre-cache novel-view rays on GPU
    render_sensor = LidarProjectionModel(
        elevation_min=train_sensor.elevation_min_deg,
        elevation_max=train_sensor.elevation_max_deg,
        num_detectors=train_sensor.num_detectors,
        num_azimuth=args.render_azimuth,
        max_range=train_sensor.max_range,
    )
    ro_np, rd_np = render_sensor.get_rays(sensor_position=args.render_pos, flatten=True)
    render_rays_o = torch.from_numpy(ro_np).float().to(device)
    render_rays_d = torch.from_numpy(rd_np).float().to(device)
    print(
        f"Live render viewpoint: {args.render_pos} "
        f"({render_sensor.num_detectors}x{render_sensor.num_azimuth} = {len(ro_np):,} rays)",
        flush=True,
    )

    # 4. Start Three.js HTTP server
    scene_payload = {
        "model": args.model,
        "s_max": args.s_max,
        "render_pos": args.render_pos,
        "no_gt_scene": args.no_gt_scene,
        "color_by": args.color_by,
        "scene_objects": scene_cfg.get("scene_objects", []),
    }
    state = LiveServerState(scene_payload=scene_payload, args=args)
    server = start_http_server(state, args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"\n>>> Live 3D Viewer running at: {url}  (Ctrl+C to stop) <<<\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    def push_render(
        epoch_float: float,
        step: int,
        loss_val: float,
        rps: float,
        status: str,
        append_loss: bool = True,
    ) -> None:
        with state.lock:
            method = state.render_method
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_r0 = time.perf_counter()
        pts, rngs, probs = render_cloud_fast_gpu(
            model=model,
            render_rays_o=render_rays_o,
            render_rays_d=render_rays_d,
            method=method,
            threshold=args.threshold,
            non_return_threshold=args.non_return_threshold,
            batch_size=args.batch_size,
        )
        render_ms = (time.perf_counter() - t_r0) * 1000.0
        state.publish(
            pts=pts,
            rngs=rngs,
            probs=probs,
            epoch_float=epoch_float,
            total_epochs=args.epochs,
            step=step,
            loss_val=loss_val,
            rays_per_sec=rps,
            render_ms=render_ms,
            status=status,
            append_loss=append_loss,
        )

    # 5. Train directly on the main thread at 100% native GPU speed
    try:
        while not state.stop_event.is_set():
            optimizer, scheduler = make_optimizer_and_scheduler(model, args, device)
            global_step = 0
            last_loss = float("nan")
            last_rps = 0.0

            with state.lock:
                state.loss_history.clear()
            push_render(0.0, 0, float("nan"), 0.0, "Untrained", append_loss=False)

            restarted = False
            for epoch in range(args.epochs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                epoch_t0 = time.perf_counter()
                window_t0 = epoch_t0
                window_rays = 0
                window_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
                window_batches = 0
                epoch_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)

                model.train()
                for batch in dataloader:
                    if state.reset_event.is_set():
                        state.reset_event.clear()
                        state.pause_event.clear()
                        model.reset_weights()
                        print("\n[Reset] Weights re-initialized in-place — restarting from Epoch 0.", flush=True)
                        restarted = True
                        break

                    while state.pause_event.is_set():
                        if state.reset_event.is_set():
                            break
                        if state.rerender_event.is_set():
                            state.rerender_event.clear()
                            push_render(global_step / steps_per_epoch, global_step, last_loss, last_rps, "Paused", append_loss=False)
                        time.sleep(0.05)

                    if state.reset_event.is_set():
                        state.reset_event.clear()
                        state.pause_event.clear()
                        model.reset_weights()
                        print("\n[Reset] Weights re-initialized in-place — restarting from Epoch 0.", flush=True)
                        restarted = True
                        break

                    optimizer.zero_grad(set_to_none=True)
                    ray_o = batch["ray_o"]
                    ray_d = batch["ray_d"]
                    hit_range = batch["hit_range"]
                    hit_mask = batch["hit_mask"]

                    if args.origin_slide:
                        ray_o, hit_range, _ = vectorized_gpu_origin_slide(
                            ray_o=ray_o,
                            ray_d=ray_d,
                            hit_ranges=hit_range,
                            hit_masks=hit_mask,
                        )

                    _, _, round_outputs, p_hit = model(ray_o, ray_d)

                    if args.model == "hash":
                        loss = compute_hash_sck_loss(
                            round_outputs=round_outputs,
                            hit_range=hit_range,
                            p_hit=p_hit,
                            hit_mask=hit_mask,
                            lambda_raydrop=args.lambda_raydrop,
                            lambda_sky_cdf=args.lambda_sky_cdf,
                            s_max=args.s_max,
                        )
                    else:
                        loss, _ = compute_sck_loss(
                            round_outputs=round_outputs,
                            hit_range=hit_range,
                            p_hit=p_hit,
                            hit_mask=hit_mask,
                            lambda_raydrop=args.lambda_raydrop,
                            s_max=args.s_max,
                            return_metrics=False,
                        )

                    loss.backward()
                    optimizer.step()

                    loss_det = loss.detach()
                    epoch_loss_gpu += loss_det
                    window_loss_gpu += loss_det
                    window_batches += 1
                    window_rays += ray_o.shape[0]
                    global_step += 1

                    if global_step % vis_every_steps == 0 or state.rerender_event.is_set():
                        state.rerender_event.clear()
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        now = time.perf_counter()
                        last_rps = window_rays / max(now - window_t0, 1e-6)
                        last_loss = (window_loss_gpu / max(1, window_batches)).item()
                        push_render(global_step / steps_per_epoch, global_step, last_loss, last_rps, "Training")

                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        window_t0 = time.perf_counter()
                        window_rays = 0
                        window_loss_gpu.zero_()
                        window_batches = 0

                if restarted:
                    break

                if scheduler is not None:
                    scheduler.step()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                ep_time = time.perf_counter() - epoch_t0
                ep_loss = (epoch_loss_gpu / steps_per_epoch).item()
                ep_rps = dataloader.N / max(ep_time, 1e-6)
                last_loss = ep_loss
                last_rps = ep_rps
                print(
                    f"Epoch [{epoch + 1:02d}/{args.epochs:02d}] - "
                    f"Loss: {ep_loss:.6f} | {ep_time:.2f}s ({ep_rps:,.0f} rays/s)",
                    flush=True,
                )

            if restarted:
                continue

            push_render(
                global_step / steps_per_epoch,
                global_step,
                last_loss,
                last_rps,
                "Complete",
                append_loss=False,
            )
            if args.save_checkpoint:
                orig = getattr(model, "_orig_mod", model)
                ckpt_path = Path(args.save_checkpoint)
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(orig.state_dict(), ckpt_path)
                print(f"Saved model weights -> {ckpt_path}", flush=True)

            if args.exit_on_finish:
                break

            print(f"Training complete! Viewer still live at {url} (press R in browser to retrain, Ctrl+C to exit).", flush=True)
            while not state.stop_event.is_set():
                if state.reset_event.is_set():
                    state.reset_event.clear()
                    state.pause_event.clear()
                    model.reset_weights()
                    print("\n[Reset] Weights re-initialized in-place — restarting from Epoch 0.", flush=True)
                    break
                if state.rerender_event.is_set():
                    state.rerender_event.clear()
                    push_render(global_step / steps_per_epoch, global_step, last_loss, last_rps, "Complete", append_loss=False)
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nShutting down live viewer...", flush=True)
    finally:
        state.stop_event.set()
        server.shutdown()


if __name__ == "__main__":
    main()
