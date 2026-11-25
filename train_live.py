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
      <button id="btn-unobs">Unobserved: Off (U)</button>
    </div>
    <div class="btn-row">
      <button id="btn-cam">Reset Cam</button>
    </div>
    <div class="row">
      <label for="sel-geom">Point style (V)</label>
      <select id="sel-geom">
        <option value="points">Flat Points</option>
        <option value="spheres">3D Spheres</option>
        <option value="patches" selected>Normal Patches</option>
        <option value="mesh">3D Mesh (MC)</option>
        <option value="mesh+patches">Mesh + Patches</option>
      </select>
    </div>
    <div id="mesh-controls" style="display: none; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px; margin-top: 2px; flex-direction: column; gap: 6px;">
      <div class="row">
        <label for="rng-iso">MC Iso-level (<span id="lbl-iso">0.45</span>)</label>
        <input id="rng-iso" type="range" min="0.05" max="0.95" step="0.02" value="0.45" />
      </div>
      <div class="row">
        <label for="rng-iso-clump">Clump isolate (<span id="lbl-clump">0.45</span>)</label>
        <input id="rng-iso-clump" type="range" min="0.00" max="1.00" step="0.05" value="0.45" />
      </div>
      <div class="row">
        <label for="chk-mask-unobs">Mask unobserved</label>
        <input id="chk-mask-unobs" type="checkbox" checked style="accent-color: #58a6ff; width: 15px; height: 15px;" />
      </div>
      <div class="row">
        <label for="sel-mres">MC Grid res</label>
        <select id="sel-mres">
          <option value="64">64&sup3; (Fast)</option>
          <option value="96" selected>96&sup3; (Balanced)</option>
          <option value="128">128&sup3; (Fine)</option>
        </select>
      </div>
    </div>
    <div class="row">
      <label for="sel-gt">GT Scene (G)</label>
      <select id="sel-gt">
        <option value="transparent" selected>Transparent</option>
        <option value="both">Transparent + Wire</option>
        <option value="wireframe">Wireframe Only</option>
        <option value="off">Hidden</option>
      </select>
    </div>
    <div class="row">
      <label for="sel-color">Color by</label>
      <select id="sel-color">
        <option value="range" selected>Range (m)</option>
        <option value="z">Height Z (m)</option>
        <option value="prob">Return Prob</option>
        <option value="normal">Surface Normal</option>
        <option value="clump">Clump ID (MC)</option>
      </select>
    </div>
    <div class="row">
      <label for="rng-size">Element size (<span id="lbl-size">0.08</span>m)</label>
      <input id="rng-size" type="range" min="0.02" max="0.25" step="0.01" value="0.08" />
    </div>
    <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px; margin-top: 2px; display: flex; flex-direction: column; gap: 6px;">
      <div class="row">
        <label for="rng-emin">Elev min (<span id="lbl-emin">-25</span>&deg;)</label>
        <input id="rng-emin" type="range" min="-60" max="45" step="1" value="-25" />
      </div>
      <div class="row">
        <label for="rng-emax">Elev max (<span id="lbl-emax">15</span>&deg;)</label>
        <input id="rng-emax" type="range" min="-45" max="60" step="1" value="15" />
      </div>
      <div class="row">
        <label for="rng-ch">Channels (<span id="lbl-ch">32</span>)</label>
        <input id="rng-ch" type="range" min="4" max="128" step="1" value="32" />
      </div>
    </div>
    <div>
      <div class="cbar" id="cbar-grad"></div>
      <div class="cbar-labels"><span id="cbar-min">0.0</span><span id="cbar-mid">Range (m)</span><span id="cbar-max">15.0</span></div>
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
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.copy(DEFAULT_CAM_TGT);
controls.update();

// 3-point studio lighting + soft sensor headlight so spheres & normal patches reflect light nicely
scene.add(new THREE.HemisphereLight(0xdbeafe, 0x1e293b, 0.65));
const keyLight = new THREE.DirectionalLight(0xfff5e4, 1.35);
keyLight.position.set(8, -10, 14);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x93c5fd, 0.75);
rimLight.position.set(-10, 9, 8);
scene.add(rimLight);
const sensorLight = new THREE.PointLight(0xffffff, 1.2, 20, 1.4);
sensorLight.position.set(0, 0, 1.5);
scene.add(sensorLight);

// Reference floor grid (rotated into XY plane for Z-up) + axes
const grid = new THREE.GridHelper(16, 16, 0x30363d, 0x1c2128);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.01;
scene.add(grid);
scene.add(new THREE.AxesHelper(1.2));

// Ground-truth scene groups (separated fill vs wireframe so user can toggle freely)
const gtGroup = new THREE.Group();
const gtFillGroup = new THREE.Group();
const gtWireGroup = new THREE.Group();
gtGroup.add(gtFillGroup);
gtGroup.add(gtWireGroup);
scene.add(gtGroup);
let gtMode = 'transparent';

function applyGtMode() {
  gtGroup.visible = (gtMode !== 'off');
  gtFillGroup.visible = (gtMode === 'transparent' || gtMode === 'both');
  gtWireGroup.visible = (gtMode === 'wireframe' || gtMode === 'both');
  document.getElementById('sel-gt').value = gtMode;
}

// Cartoon Velodyne VLP-32C (Ultra Puck) 3D LiDAR model centered at optical origin
function buildCartoonVLP32C() {
  const root = new THREE.Group();
  // Rotate local Y-up cylinders into scene Z-up so +Y inside body maps to world +Z
  const body = new THREE.Group();
  body.rotation.x = Math.PI / 2;
  root.add(body);

  const R = 0.135; // Stylized cartoon Ultra Puck radius (~2.6x real 51.5mm for clear visibility)
  const aluMat = new THREE.MeshStandardMaterial({ color: 0xcbd5e1, metalness: 0.72, roughness: 0.26 });
  const aluTopMat = new THREE.MeshStandardMaterial({ color: 0xdbe4ee, metalness: 0.70, roughness: 0.24 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x1e2530, metalness: 0.25, roughness: 0.55 });
  const blueRingMat = new THREE.MeshStandardMaterial({ color: 0x1d8cf8, emissive: 0x0a3d7a, roughness: 0.25, metalness: 0.35 });
  const glassMat = new THREE.MeshStandardMaterial({
    color: 0x0b101b,
    roughness: 0.12,
    metalness: 0.45,
  });

  // 1. Rubber mounting pad & lower brushed-aluminum base
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.90, R * 0.86, 0.016, 28), darkMat);
  pad.position.y = -0.115;
  body.add(pad);

  const base = new THREE.Mesh(new THREE.CylinderGeometry(R, R * 0.96, 0.056, 32), aluMat);
  base.position.y = -0.079;
  body.add(base);

  // Subtle green status LED on front bezel (+X)
  const led = new THREE.Mesh(
    new THREE.SphereGeometry(0.0085, 10, 8),
    new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x16a34a, emissiveIntensity: 1.5, roughness: 0.2 })
  );
  led.position.set(R * 0.99, -0.079, 0);
  body.add(led);

  // Rear armored cable boot & short cartoon pigtail (-X)
  const boot = new THREE.Mesh(new THREE.CylinderGeometry(0.017, 0.021, 0.044, 12), darkMat);
  boot.rotation.z = Math.PI / 2;
  boot.position.set(-R - 0.016, -0.082, 0);
  body.add(boot);

  const cableCurve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(-R - 0.036, -0.082, 0),
    new THREE.Vector3(-R - 0.078, -0.088, 0),
    new THREE.Vector3(-R - 0.095, -0.135, 0)
  );
  const cable = new THREE.Mesh(new THREE.TubeGeometry(cableCurve, 10, 0.010, 8, false), darkMat);
  body.add(cable);

  // 2. Signature Velodyne blue accent trim rings framing the optical window
  const ringGeo = new THREE.TorusGeometry(R * 1.005, 0.0055, 10, 32);
  const lowerRing = new THREE.Mesh(ringGeo, blueRingMat);
  lowerRing.rotation.x = Math.PI / 2;
  lowerRing.position.y = -0.051;
  body.add(lowerRing);

  const upperRing = new THREE.Mesh(ringGeo, blueRingMat);
  upperRing.rotation.x = Math.PI / 2;
  upperRing.position.y = 0.051;
  body.add(upperRing);

  // 3. Glossy dark IR optical window band (centered at y=0 optical origin)
  const windowBand = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.97, R * 0.97, 0.102, 32), glassMat);
  body.add(windowBand);

  // 4. Two-tier chamfered aluminum top cap & dark center medallion
  const topRim = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.96, R, 0.044, 32), aluTopMat);
  topRim.position.y = 0.073;
  body.add(topRim);

  const topBevel = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.78, R * 0.96, 0.018, 32), aluTopMat);
  topBevel.position.y = 0.104;
  body.add(topBevel);

  const topBadge = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.48, R * 0.48, 0.006, 24), darkMat);
  topBadge.position.y = 0.114;
  body.add(topBadge);

  const topDot = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.14, R * 0.14, 0.008, 16), blueRingMat);
  topDot.position.y = 0.115;
  body.add(topDot);

  return root;
}

const sensorMarker = buildCartoonVLP32C();
scene.add(sensorMarker);

// --- 3 Rendering Modes: Flat Points, Instanced 3D Spheres, Instanced Normal Patches (Surfels) ---
const cloudGeom = new THREE.BufferGeometry();
const cloudMat = new THREE.PointsMaterial({
  size: 0.08,
  vertexColors: true,
  sizeAttenuation: true,
});
const pointsMesh = new THREE.Points(cloudGeom, cloudMat);
pointsMesh.frustumCulled = false;
scene.add(pointsMesh);

const sphereGeo = new THREE.SphereGeometry(0.5, 10, 8);
const sphereMat = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  roughness: 0.38,
  metalness: 0.12,
});

const patchGeo = new THREE.CircleGeometry(0.55, 10);
const patchMat = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  roughness: 0.32,
  metalness: 0.15,
  side: THREE.DoubleSide,
});

let sphereInstMesh = null;
let patchInstMesh = null;
let instCapacity = 0;

// Live 3D Marching Cubes mesh (extracted directly from HashSCKNetwork's 3D field)
const mcGeom = new THREE.BufferGeometry();
const mcMat = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  vertexColors: true,
  roughness: 0.45,
  metalness: 0.08,
  side: THREE.DoubleSide,
});
const mcMesh = new THREE.Mesh(mcGeom, mcMat);
mcMesh.frustumCulled = false;
mcMesh.visible = false;
scene.add(mcMesh);

let lastMeshVerts = null, lastMeshNorms = null, lastMeshIdx = null, lastMeshRegions = null;
let lastMeshVersion = -1;
let meshTrisCount = 0;
let meshMsLast = 0;

// Translucent 3D mesh highlighting online-tracked UNOBSERVED / blind-zone regions
const unobsGeom = new THREE.BufferGeometry();
const unobsMat = new THREE.MeshStandardMaterial({
  color: 0xef4444,
  emissive: 0x7f1d1d,
  emissiveIntensity: 0.35,
  transparent: true,
  opacity: 0.28,
  roughness: 0.35,
  metalness: 0.05,
  depthWrite: false,
  side: THREE.FrontSide,
});
const unobsMesh = new THREE.Mesh(unobsGeom, unobsMat);
unobsMesh.frustumCulled = false;
unobsMesh.visible = false;
scene.add(unobsMesh);

let showUnobs = false;
let lastUnobsVersion = -1;
let lastObsPct = 0.0;

function updateUnobsButton() {
  const btn = document.getElementById('btn-unobs');
  if (showUnobs) {
    const pctStr = lastObsPct > 0 ? ` (${(100 - lastObsPct).toFixed(0)}% blind)` : '';
    btn.textContent = `Unobserved: On${pctStr} (U)`;
    btn.classList.add('primary');
  } else {
    btn.textContent = 'Unobserved: Off (U)';
    btn.classList.remove('primary');
  }
  unobsMesh.visible = showUnobs && lastUnobsVersion > 0;
}

function ensureInstancedMeshes(count) {
  const needed = Math.max(1, count);
  if (needed <= instCapacity && sphereInstMesh && patchInstMesh) return;
  instCapacity = Math.max(needed, Math.ceil(needed * 1.25));

  if (sphereInstMesh) { scene.remove(sphereInstMesh); sphereInstMesh.dispose(); }
  if (patchInstMesh) { scene.remove(patchInstMesh); patchInstMesh.dispose(); }

  sphereInstMesh = new THREE.InstancedMesh(sphereGeo, sphereMat, instCapacity);
  sphereInstMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  sphereInstMesh.frustumCulled = false;
  sphereInstMesh.count = 0;
  scene.add(sphereInstMesh);

  patchInstMesh = new THREE.InstancedMesh(patchGeo, patchMat, instCapacity);
  patchInstMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  patchInstMesh.frustumCulled = false;
  patchInstMesh.count = 0;
  scene.add(patchInstMesh);
}

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

// Reusable scratch objects (zero GC churn per frame)
const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _norm = new THREE.Vector3();
const _unitZ = new THREE.Vector3(0, 0, 1);
const _idQuat = new THREE.Quaternion();
const _col = new THREE.Color();

// Client-side state
let lastPts = null, lastNorms = null, lastRanges = null, lastProbs = null;
let currentSMax = 15.0;
let colorMode = 'range';
let geomMode = 'patches';
let elemSize = 0.08;
let lastVersion = -1;

function computePointColors() {
  const n = lastPts ? lastPts.length / 3 : 0;
  const colors = new Float32Array(n * 3);
  let vmin = 0.0, vmax = currentSMax, label = 'Range (m)';
  if (colorMode === 'z') { vmin = 0.0; vmax = 3.0; label = 'Height Z (m)'; }
  else if (colorMode === 'prob') { vmin = 0.0; vmax = 1.0; label = 'Return Prob'; }
  else if (colorMode === 'normal') { vmin = -1.0; vmax = 1.0; label = 'Normal XYZ -> RGB'; }

  document.getElementById('cbar-min').textContent = vmin.toFixed(1);
  document.getElementById('cbar-mid').textContent = label;
  document.getElementById('cbar-max').textContent = vmax.toFixed(1);

  if (colorMode === 'normal' && lastNorms) {
    for (let i = 0; i < n; i++) {
      colors[i * 3 + 0] = Math.max(0, Math.min(1, lastNorms[i * 3 + 0] * 0.5 + 0.5));
      colors[i * 3 + 1] = Math.max(0, Math.min(1, lastNorms[i * 3 + 1] * 0.5 + 0.5));
      colors[i * 3 + 2] = Math.max(0, Math.min(1, lastNorms[i * 3 + 2] * 0.5 + 0.5));
    }
    return colors;
  }

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
  return colors;
}

const CLUMP_PALETTE = [
  [0.95, 0.45, 0.35], [0.30, 0.78, 0.95], [0.42, 0.88, 0.52], [0.98, 0.78, 0.25],
  [0.78, 0.45, 0.98], [0.98, 0.55, 0.78], [0.35, 0.92, 0.82], [0.98, 0.62, 0.25],
  [0.58, 0.68, 0.98], [0.85, 0.92, 0.38], [0.92, 0.38, 0.58], [0.45, 0.82, 0.68],
];

function rebuildMeshColors() {
  if (!lastMeshVerts || lastMeshVerts.length === 0) return;
  const n = lastMeshVerts.length / 3;
  const colors = new Float32Array(n * 3);
  if (colorMode === 'clump' && lastMeshRegions) {
    for (let i = 0; i < n; i++) {
      const c = CLUMP_PALETTE[lastMeshRegions[i] % CLUMP_PALETTE.length];
      colors[i * 3 + 0] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
    }
  } else if (colorMode === 'normal' && lastMeshNorms) {
    for (let i = 0; i < n; i++) {
      colors[i * 3 + 0] = Math.max(0, Math.min(1, lastMeshNorms[i * 3 + 0] * 0.5 + 0.5));
      colors[i * 3 + 1] = Math.max(0, Math.min(1, lastMeshNorms[i * 3 + 1] * 0.5 + 0.5));
      colors[i * 3 + 2] = Math.max(0, Math.min(1, lastMeshNorms[i * 3 + 2] * 0.5 + 0.5));
    }
  } else if (colorMode === 'prob') {
    // Neutral smooth ceramic clay so Marching Cubes lighting & contours shine
    for (let i = 0; i < n; i++) {
      colors[i * 3 + 0] = 0.78;
      colors[i * 3 + 1] = 0.84;
      colors[i * 3 + 2] = 0.92;
    }
  } else {
    const vmin = 0.0;
    const vmax = colorMode === 'z' ? 3.0 : currentSMax;
    const inv = 1.0 / Math.max(1e-6, vmax - vmin);
    const sx = sensorMarker.position.x, sy = sensorMarker.position.y, sz = sensorMarker.position.z;
    for (let i = 0; i < n; i++) {
      const vx = lastMeshVerts[i * 3], vy = lastMeshVerts[i * 3 + 1], vz = lastMeshVerts[i * 3 + 2];
      const val = colorMode === 'z' ? vz : Math.hypot(vx - sx, vy - sy, vz - sz);
      const idx = Math.max(0, Math.min(255, Math.round((val - vmin) * inv * 255)));
      colors[i * 3 + 0] = PLASMA_LUT[idx * 3 + 0];
      colors[i * 3 + 1] = PLASMA_LUT[idx * 3 + 1];
      colors[i * 3 + 2] = PLASMA_LUT[idx * 3 + 2];
    }
  }
  mcGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

function isMeshMode() {
  return geomMode === 'mesh' || geomMode === 'mesh+patches';
}

function syncMeshControlsUI() {
  const show = isMeshMode();
  document.getElementById('mesh-controls').style.display = show ? 'flex' : 'none';
  mcMesh.visible = show && lastMeshVerts !== null && lastMeshVerts.length > 0;
}

function rebuildCloudVisual() {
  const n = lastPts ? lastPts.length / 3 : 0;
  const showPatches = (geomMode === 'patches' || geomMode === 'mesh+patches');
  pointsMesh.visible = (geomMode === 'points');
  if (sphereInstMesh) sphereInstMesh.visible = (geomMode === 'spheres');
  if (patchInstMesh) patchInstMesh.visible = showPatches;
  syncMeshControlsUI();
  if (isMeshMode()) rebuildMeshColors();

  if (geomMode === 'mesh') return;

  if (n === 0) {
    cloudGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    if (sphereInstMesh) sphereInstMesh.count = 0;
    if (patchInstMesh) patchInstMesh.count = 0;
    return;
  }

  const colors = computePointColors();

  if (geomMode === 'points') {
    cloudMat.size = elemSize;
    cloudGeom.setAttribute('position', new THREE.BufferAttribute(lastPts, 3));
    cloudGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    cloudGeom.computeBoundingSphere();
  } else {
    ensureInstancedMeshes(n);
    sphereInstMesh.visible = (geomMode === 'spheres');
    patchInstMesh.visible = showPatches;

    if (geomMode === 'spheres') {
      _scale.set(elemSize, elemSize, elemSize);
      sphereInstMesh.count = n;
      for (let i = 0; i < n; i++) {
        _pos.set(lastPts[i * 3], lastPts[i * 3 + 1], lastPts[i * 3 + 2]);
        _mat4.compose(_pos, _idQuat, _scale);
        sphereInstMesh.setMatrixAt(i, _mat4);
        _col.setRGB(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
        sphereInstMesh.setColorAt(i, _col);
      }
      sphereInstMesh.instanceMatrix.needsUpdate = true;
      if (sphereInstMesh.instanceColor) sphereInstMesh.instanceColor.needsUpdate = true;
    } else if (showPatches) {
      const ps = elemSize * 1.65; // slightly larger disc so neighboring patches tile cleanly
      _scale.set(ps, ps, ps);
      patchInstMesh.count = n;
      for (let i = 0; i < n; i++) {
        _pos.set(lastPts[i * 3], lastPts[i * 3 + 1], lastPts[i * 3 + 2]);
        _norm.set(lastNorms[i * 3], lastNorms[i * 3 + 1], lastNorms[i * 3 + 2]);
        if (_norm.lengthSq() < 1e-6) _norm.set(0, 0, 1);
        else _norm.normalize();
        // Handle anti-parallel (0, 0, -1) cleanly
        if (_norm.z < -0.9999) _quat.set(1, 0, 0, 0);
        else _quat.setFromUnitVectors(_unitZ, _norm);
        _mat4.compose(_pos, _quat, _scale);
        patchInstMesh.setMatrixAt(i, _mat4);
        _col.setRGB(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
        patchInstMesh.setColorAt(i, _col);
      }
      patchInstMesh.instanceMatrix.needsUpdate = true;
      if (patchInstMesh.instanceColor) patchInstMesh.instanceColor.needsUpdate = true;
    }
  }
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
    if (data.render_pos) {
      sensorMarker.position.set(...data.render_pos);
      sensorLight.position.set(...data.render_pos);
    }
    if (data.s_max) currentSMax = data.s_max;
    if (data.model) document.getElementById('hud-title').textContent = `LiZARD Live (${data.model.toUpperCase()} SCK)`;
    if (data.no_gt_scene) gtMode = 'off';
    else if (data.gt_style) gtMode = data.gt_style;
    if (data.point_style) {
      geomMode = data.point_style;
      document.getElementById('sel-geom').value = geomMode;
    }
    if (data.mesh_iso !== undefined) {
      document.getElementById('rng-iso').value = data.mesh_iso;
      document.getElementById('lbl-iso').textContent = Number(data.mesh_iso).toFixed(2);
    }
    if (data.clump_isolate !== undefined) {
      document.getElementById('rng-iso-clump').value = data.clump_isolate;
      document.getElementById('lbl-clump').textContent = Number(data.clump_isolate).toFixed(2);
    }
    if (data.mesh_res !== undefined) {
      document.getElementById('sel-mres').value = String(data.mesh_res);
    }
    if (data.mask_unobserved !== undefined) {
      document.getElementById('chk-mask-unobs').checked = Boolean(data.mask_unobserved);
    }
    if (data.show_unobserved !== undefined) {
      showUnobs = Boolean(data.show_unobserved);
      updateUnobsButton();
    }
    syncMeshControlsUI();
    if (data.color_by) {
      colorMode = data.color_by === 'probability' ? 'prob' : data.color_by;
      document.getElementById('sel-color').value = colorMode;
    }
    if (data.render_elev_min !== undefined) {
      document.getElementById('rng-emin').value = data.render_elev_min;
      document.getElementById('lbl-emin').textContent = Math.round(data.render_elev_min);
    }
    if (data.render_elev_max !== undefined) {
      document.getElementById('rng-emax').value = data.render_elev_max;
      document.getElementById('lbl-emax').textContent = Math.round(data.render_elev_max);
    }
    if (data.render_channels !== undefined) {
      document.getElementById('rng-ch').value = data.render_channels;
      document.getElementById('lbl-ch').textContent = data.render_channels;
    }
    applyGtMode();

    const palette = [0x4ea8de, 0x56cfe1, 0x64dfdf, 0x80ffdb, 0xf77f00, 0xfcbf49, 0xe63946, 0xa8dadc];
    (data.scene_objects || []).forEach((obj, idx) => {
      const name = (obj.name || '').toLowerCase();
      const isFoliage =
        name.includes('foliage') || name.includes('crown') || name.includes('leaf') ||
        name.includes('leaves') || name.includes('canopy') || name.includes('bush');
      const isWood = name.includes('trunk') || name.includes('branch') || name.includes('log');
      const isWindow = name.includes('window');
      const isWall = obj.density >= 50 && !isWood;

      let color = palette[idx % palette.length];
      let opacity = 0.28;
      if (isFoliage) { color = 0x22c55e; opacity = 0.26; }
      else if (isWood) { color = 0x8b5a2b; opacity = 0.24; }
      else if (isWindow) { color = 0x38bdf8; opacity = 0.20; }
      else if (isWall) { color = 0x4b5c73; opacity = 0.10; }

      const mat = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity,
        roughness: isWall ? 0.7 : 0.28,
        metalness: 0.05,
        depthWrite: false,
        side: isWall ? THREE.DoubleSide : THREE.FrontSide,
      });
      const edgeMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: isWall ? 0.25 : 0.42 });

      let geom = null;
      if (obj.type === 'box') {
        geom = new THREE.BoxGeometry(obj.size[0], obj.size[1], obj.size[2]);
      } else if (obj.type === 'sphere') {
        geom = new THREE.SphereGeometry(obj.radius, 32, 24);
      } else if (obj.type === 'cylinder') {
        geom = new THREE.CylinderGeometry(obj.radius, obj.radius, obj.height, 32);
      }
      if (!geom) return;

      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(...obj.center);

      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geom, 18), edgeMat);
      edges.position.set(...obj.center);

      if (obj.type === 'cylinder') {
        const axis = new THREE.Vector3(...(obj.axis || [0, 0, 1])).normalize();
        const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
        mesh.quaternion.copy(q);
        edges.quaternion.copy(q);
      } else if (obj.type === 'box' && obj.rotation) {
        const r = obj.rotation;
        const m4 = new THREE.Matrix4().set(
          r[0][0], r[0][1], r[0][2], 0,
          r[1][0], r[1][1], r[1][2], 0,
          r[2][0], r[2][1], r[2][2], 0,
          0, 0, 0, 1
        );
        mesh.quaternion.setFromRotationMatrix(m4);
        edges.quaternion.setFromRotationMatrix(m4);
      }

      gtFillGroup.add(mesh);
      gtWireGroup.add(edges);
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
      const norms = new Float32Array(buf, offset + n * 12, n * 3);
      const rngs = new Float32Array(buf, offset + n * 24, n);
      const probs = new Float32Array(buf, offset + n * 28, n);

      lastVersion = meta.version;
      currentSMax = meta.s_max || currentSMax;
      lastPts = pts;
      lastNorms = norms;
      lastRanges = rngs;
      lastProbs = probs;

      rebuildCloudVisual();

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

// Poll binary 3D Marching Cubes meshes from /api/mesh
async function pollMesh() {
  try {
    if (isMeshMode()) {
      const resp = await fetch(`/api/mesh?v=${lastMeshVersion}`, { cache: 'no-store' });
      if (resp.status === 200) {
        const buf = await resp.arrayBuffer();
        const dv = new DataView(buf);
        const metaLen = dv.getUint32(0, true);
        const metaStr = new TextDecoder().decode(new Uint8Array(buf, 4, metaLen));
        const meta = JSON.parse(metaStr);

        let offset = (4 + metaLen + 3) & ~3;
        const nv = meta.num_verts || 0;
        const nt = meta.num_tris || 0;
        const verts = new Float32Array(buf, offset, nv * 3);
        const norms = new Float32Array(buf, offset + nv * 12, nv * 3);
        const regions = new Uint32Array(buf, offset + nv * 24, nv);
        const indices = new Uint32Array(buf, offset + nv * 28, nt * 3);

        lastMeshVersion = meta.version;
        meshTrisCount = nt;
        meshMsLast = meta.mesh_ms || 0;
        lastMeshVerts = verts;
        lastMeshNorms = norms;
        lastMeshRegions = regions;
        lastMeshIdx = indices;

        mcGeom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        mcGeom.setAttribute('normal', new THREE.BufferAttribute(norms, 3));
        mcGeom.setIndex(new THREE.BufferAttribute(indices, 1));
        rebuildMeshColors();
        mcGeom.computeBoundingSphere();
        syncMeshControlsUI();

        if (geomMode === 'mesh') {
          document.getElementById('st-pts').textContent = `${nt.toLocaleString()} tris (${Math.round(meshMsLast)}ms)`;
        }
      }
    }
  } catch (e) {}
  setTimeout(pollMesh, 120);
}

// Poll binary 3D Unobserved / Blind-Zone mesh from /api/unobs
async function pollUnobs() {
  try {
    if (showUnobs) {
      const resp = await fetch(`/api/unobs?v=${lastUnobsVersion}`, { cache: 'no-store' });
      if (resp.status === 200) {
        const buf = await resp.arrayBuffer();
        const dv = new DataView(buf);
        const metaLen = dv.getUint32(0, true);
        const metaStr = new TextDecoder().decode(new Uint8Array(buf, 4, metaLen));
        const meta = JSON.parse(metaStr);

        let offset = (4 + metaLen + 3) & ~3;
        const nv = meta.num_verts || 0;
        const nt = meta.num_tris || 0;
        const verts = new Float32Array(buf, offset, nv * 3);
        const norms = new Float32Array(buf, offset + nv * 12, nv * 3);
        const indices = new Uint32Array(buf, offset + nv * 24, nt * 3);

        lastUnobsVersion = meta.version;
        lastObsPct = meta.observed_pct || 0.0;

        unobsGeom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        unobsGeom.setAttribute('normal', new THREE.BufferAttribute(norms, 3));
        unobsGeom.setIndex(new THREE.BufferAttribute(indices, 1));
        unobsGeom.computeBoundingSphere();
        updateUnobsButton();
      }
    }
  } catch (e) {}
  setTimeout(pollUnobs, 150);
}

async function sendControl(action, extra = {}) {
  try {
    await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, ...extra }),
    });
  } catch (e) {}
}

let sensorTimer = null;
function scheduleSensorUpdate() {
  const eminEl = document.getElementById('rng-emin');
  const emaxEl = document.getElementById('rng-emax');
  const chEl = document.getElementById('rng-ch');
  let emin = parseFloat(eminEl.value);
  let emax = parseFloat(emaxEl.value);
  const ch = parseInt(chEl.value, 10);

  if (emin > emax - 1) {
    emax = Math.min(60, emin + 1);
    emaxEl.value = emax;
  }
  document.getElementById('lbl-emin').textContent = Math.round(emin);
  document.getElementById('lbl-emax').textContent = Math.round(emax);
  document.getElementById('lbl-ch').textContent = ch;

  clearTimeout(sensorTimer);
  sensorTimer = setTimeout(() => {
    sendControl('set_sensor', { elev_min: emin, elev_max: emax, num_detectors: ch });
  }, 75);
}

document.getElementById('rng-emin').addEventListener('input', () => {
  const eminEl = document.getElementById('rng-emin');
  const emaxEl = document.getElementById('rng-emax');
  if (parseFloat(eminEl.value) >= parseFloat(emaxEl.value)) {
    emaxEl.value = Math.min(60, parseFloat(eminEl.value) + 1);
  }
  scheduleSensorUpdate();
});
document.getElementById('rng-emax').addEventListener('input', () => {
  const eminEl = document.getElementById('rng-emin');
  const emaxEl = document.getElementById('rng-emax');
  if (parseFloat(emaxEl.value) <= parseFloat(eminEl.value)) {
    eminEl.value = Math.max(-60, parseFloat(emaxEl.value) - 1);
  }
  scheduleSensorUpdate();
});
document.getElementById('rng-ch').addEventListener('input', scheduleSensorUpdate);

document.getElementById('btn-pause').addEventListener('click', () => sendControl('toggle_pause'));
document.getElementById('btn-reset').addEventListener('click', () => sendControl('reset'));
document.getElementById('btn-mode').addEventListener('click', () => sendControl('toggle_method'));

let meshTimer = null;
function pushMeshSettings(immediate = false) {
  const iso = parseFloat(document.getElementById('rng-iso').value);
  const clump = parseFloat(document.getElementById('rng-iso-clump').value);
  const res = parseInt(document.getElementById('sel-mres').value, 10);
  const maskUnobs = document.getElementById('chk-mask-unobs').checked;
  document.getElementById('lbl-iso').textContent = iso.toFixed(2);
  document.getElementById('lbl-clump').textContent = clump.toFixed(2);
  clearTimeout(meshTimer);
  const doSend = () => sendControl('set_mesh', {
    enabled: isMeshMode(),
    iso_level: iso,
    clump_isolate: clump,
    grid_res: res,
    mask_unobserved: maskUnobs,
  });
  if (immediate) doSend();
  else meshTimer = setTimeout(doSend, 60);
}

function toggleUnobs() {
  showUnobs = !showUnobs;
  updateUnobsButton();
  sendControl('set_unobs', { enabled: showUnobs });
}

document.getElementById('btn-unobs').addEventListener('click', toggleUnobs);
document.getElementById('rng-iso').addEventListener('input', () => pushMeshSettings(false));
document.getElementById('rng-iso-clump').addEventListener('input', () => pushMeshSettings(false));
document.getElementById('chk-mask-unobs').addEventListener('change', () => pushMeshSettings(true));
document.getElementById('sel-mres').addEventListener('change', () => pushMeshSettings(true));

document.getElementById('sel-geom').addEventListener('change', (e) => {
  geomMode = e.target.value;
  rebuildCloudVisual();
  pushMeshSettings(true);
});
document.getElementById('sel-gt').addEventListener('change', (e) => {
  gtMode = e.target.value;
  applyGtMode();
});
document.getElementById('sel-color').addEventListener('change', (e) => {
  colorMode = e.target.value;
  rebuildCloudVisual();
});
document.getElementById('rng-size').addEventListener('input', (e) => {
  elemSize = parseFloat(e.target.value);
  document.getElementById('lbl-size').textContent = elemSize.toFixed(2);
  rebuildCloudVisual();
});
document.getElementById('btn-cam').addEventListener('click', () => {
  camera.position.copy(DEFAULT_CAM_POS);
  controls.target.copy(DEFAULT_CAM_TGT);
  controls.update();
});

const GEOM_CYCLE = ['patches', 'mesh', 'mesh+patches', 'spheres', 'points'];
const GT_CYCLE = ['transparent', 'both', 'wireframe', 'off'];
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (k === ' ' || k === 'p') { e.preventDefault(); sendControl('toggle_pause'); }
  else if (k === 'r') sendControl('reset');
  else if (k === 'm') sendControl('toggle_method');
  else if (k === 'u') toggleUnobs();
  else if (k === 'v') {
    geomMode = GEOM_CYCLE[(GEOM_CYCLE.indexOf(geomMode) + 1) % GEOM_CYCLE.length];
    document.getElementById('sel-geom').value = geomMode;
    rebuildCloudVisual();
    pushMeshSettings(true);
  }
  else if (k === 'g') {
    gtMode = GT_CYCLE[(GT_CYCLE.indexOf(gtMode) + 1) % GT_CYCLE.length];
    applyGtMode();
  }
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
pollMesh();
pollUnobs();
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
def estimate_grid_normals_gpu(
    pts_grid: torch.Tensor,
    ranges_grid: torch.Tensor,
    valid_grid: torch.Tensor,
    dirs_grid: torch.Tensor,
) -> torch.Tensor:
    """Computes edge-aware surface normals from a structured (H, W, 3) LiDAR range image.

    Uses central differences along azimuth (periodic wrap) and elevation with
    depth-discontinuity fallback to one-sided differences, and falls back to
    -ray_d for isolated volumetric points (e.g. inside porous spheres).
    All normals are oriented to face back toward the sensor origin.
    """
    H, W, _ = pts_grid.shape
    jump_thresh = 0.25 + 0.05 * ranges_grid  # adaptive depth-jump guard (meters)

    # Azimuth neighbors (wrap around 360 degrees)
    p_r = torch.roll(pts_grid, shifts=-1, dims=1)
    r_r = torch.roll(ranges_grid, shifts=-1, dims=1)
    v_r = torch.roll(valid_grid, shifts=-1, dims=1) & ((r_r - ranges_grid).abs() < jump_thresh)

    p_l = torch.roll(pts_grid, shifts=1, dims=1)
    r_l = torch.roll(ranges_grid, shifts=1, dims=1)
    v_l = torch.roll(valid_grid, shifts=1, dims=1) & ((r_l - ranges_grid).abs() < jump_thresh)

    t_az = torch.where(
        (v_r & v_l).unsqueeze(-1),
        p_r - p_l,
        torch.where(
            v_r.unsqueeze(-1),
            p_r - pts_grid,
            torch.where(v_l.unsqueeze(-1), pts_grid - p_l, torch.zeros_like(pts_grid)),
        ),
    )

    # Elevation neighbors (clamped boundaries)
    p_u = torch.cat([pts_grid[1:], pts_grid[-1:]], dim=0)
    r_u = torch.cat([ranges_grid[1:], ranges_grid[-1:]], dim=0)
    v_u = torch.cat([valid_grid[1:], torch.zeros_like(valid_grid[-1:])], dim=0)
    v_u = v_u & ((r_u - ranges_grid).abs() < jump_thresh)

    p_d = torch.cat([pts_grid[:1], pts_grid[:-1]], dim=0)
    r_d = torch.cat([ranges_grid[:1], ranges_grid[:-1]], dim=0)
    v_d = torch.cat([torch.zeros_like(valid_grid[:1]), valid_grid[:-1]], dim=0)
    v_d = v_d & ((r_d - ranges_grid).abs() < jump_thresh)

    t_el = torch.where(
        (v_u & v_d).unsqueeze(-1),
        p_u - p_d,
        torch.where(
            v_u.unsqueeze(-1),
            p_u - pts_grid,
            torch.where(v_d.unsqueeze(-1), pts_grid - p_d, torch.zeros_like(pts_grid)),
        ),
    )

    normals = torch.cross(t_az, t_el, dim=-1)
    norm_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
    has_normal = norm_len > 1e-6

    fallback = -dirs_grid  # point isolated volumetric hits back at sensor
    normals = torch.where(has_normal, normals / torch.clamp(norm_len, min=1e-6), fallback)

    # Orient all normals to face toward the sensor (-dirs_grid)
    dot = (normals * (-dirs_grid)).sum(dim=-1, keepdim=True)
    normals = torch.where(dot < 0.0, -normals, normals)
    return normals


@torch.no_grad()
def render_cloud_fast_gpu(
    model: torch.nn.Module,
    render_rays_o: torch.Tensor,
    render_rays_d: torch.Tensor,
    grid_shape: Tuple[int, int],
    method: str = "sample",
    threshold: float = 0.5,
    non_return_threshold: float = 0.1,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast GPU novel-view point cloud + surface normal render using pre-cached rays.

    Unwraps torch.compile (_orig_mod) so odd render batch sizes never trigger
    TorchInductor recompilation.
    """
    eval_model = getattr(model, "_orig_mod", model)
    was_training = eval_model.training
    eval_model.eval()

    total_rays = render_rays_o.shape[0]
    H, W = grid_shape
    all_pts = torch.empty((total_rays, 3), device=render_rays_o.device, dtype=torch.float32)
    all_med_pts = torch.empty((total_rays, 3), device=render_rays_o.device, dtype=torch.float32)
    all_med_ranges = torch.empty((total_rays,), device=render_rays_o.device, dtype=torch.float32)
    all_ranges = torch.empty((total_rays,), device=render_rays_o.device, dtype=torch.float32)
    all_probs = torch.empty((total_rays,), device=render_rays_o.device, dtype=torch.float32)
    all_valid = torch.empty((total_rays,), device=render_rays_o.device, dtype=torch.bool)
    all_med_valid = torch.empty((total_rays,), device=render_rays_o.device, dtype=torch.bool)

    def _invert_cdf(s_ext, c_ext, target, K):
        idx = torch.searchsorted(c_ext.contiguous(), target.contiguous(), right=True)
        idx = torch.clamp(idx, 1, K)
        idx_left = idx - 1
        s_0 = torch.gather(s_ext, 1, idx_left).squeeze(-1)
        s_1 = torch.gather(s_ext, 1, idx).squeeze(-1)
        c_0 = torch.gather(c_ext, 1, idx_left).squeeze(-1)
        c_1 = torch.gather(c_ext, 1, idx).squeeze(-1)
        alpha = torch.clamp((target.squeeze(-1) - c_0) / torch.clamp(c_1 - c_0, min=1e-5), 0.0, 1.0)
        return s_0 + alpha * (s_1 - s_0)

    for i in range(0, total_rays, batch_size):
        j = min(i + batch_size, total_rays)
        bo = render_rays_o[i:j]
        bd = render_rays_d[i:j]
        B = bo.shape[0]

        s_knots, c_knots, _, p_hit = eval_model(bo, bd)
        K = s_knots.shape[1]

        s_ext = F.pad(s_knots, (1, 0), value=0.0)
        c_ext = F.pad(c_knots, (1, 0), value=0.0)
        c_max = c_ext[:, -1]

        p_non_return = (1.0 - p_hit).squeeze(-1)
        ray_hit_mask = p_non_return <= non_return_threshold

        med_target = torch.full((B, 1), threshold, device=bo.device)
        med_ranges = _invert_cdf(s_ext, c_ext, med_target, K)
        med_valid = ray_hit_mask & (c_max >= threshold)

        if method == "threshold":
            pred_ranges = med_ranges
            valid = med_valid
        else:
            u = torch.rand((B, 1), device=bo.device)
            valid = ray_hit_mask & (u.squeeze(-1) <= c_max)
            pred_ranges = _invert_cdf(s_ext, c_ext, u, K)

        all_pts[i:j] = bo + pred_ranges.unsqueeze(-1) * bd
        all_med_pts[i:j] = bo + med_ranges.unsqueeze(-1) * bd
        all_med_ranges[i:j] = med_ranges
        all_ranges[i:j] = pred_ranges
        all_probs[i:j] = c_max
        all_valid[i:j] = valid
        all_med_valid[i:j] = med_valid

    if was_training:
        eval_model.train()

    if not all_valid.any():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    # Compute noise-free surface normals from the deterministic median range grid (< 0.3 ms)
    normals_grid = estimate_grid_normals_gpu(
        all_med_pts.view(H, W, 3),
        all_med_ranges.view(H, W),
        all_med_valid.view(H, W),
        render_rays_d.view(H, W, 3),
    )
    all_normals = normals_grid.view(total_rays, 3)

    return (
        all_pts[all_valid].cpu().numpy().astype(np.float32, copy=False),
        all_normals[all_valid].cpu().numpy().astype(np.float32, copy=False),
        all_ranges[all_valid].cpu().numpy().astype(np.float32, copy=False),
        all_probs[all_valid].cpu().numpy().astype(np.float32, copy=False),
    )


# ==============================================================================
# 2b. Live 3D Field Marching Cubes (vtkFlyingEdges3D)
# ==============================================================================

def compute_scene_bbox(
    scene_cfg: Dict[str, Any], s_max: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Computes a tight 3D bounding box for Marching Cubes from scene_config.json."""
    objs = scene_cfg.get("scene_objects", [])
    if not objs:
        return (-s_max, -s_max, -0.4), (s_max, s_max, 3.6)

    lo = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    hi = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for o in objs:
        c = np.asarray(o["center"], dtype=np.float64)
        kind = o["type"]
        if kind == "box":
            half = np.asarray(o["size"], dtype=np.float64) * 0.5
        elif kind == "sphere":
            half = np.full(3, float(o["radius"]))
        elif kind == "cylinder":
            r = float(o["radius"])
            h = float(o["height"]) * 0.5
            half = np.array([r, r, h])
        else:
            continue
        lo = np.minimum(lo, c - half)
        hi = np.maximum(hi, c + half)

    # Small padding around geometry while staying close to observed interior
    pad = np.array([0.15, 0.15, 0.15], dtype=np.float64)
    lo = lo - pad
    hi = hi + pad
    return (float(lo[0]), float(lo[1]), float(lo[2])), (float(hi[0]), float(hi[1]), float(hi[2]))


def _gaussian_blur_3d(vol_5d: torch.Tensor, sigma: float, radius: int) -> torch.Tensor:
    """Separable 3D Gaussian blur on GPU for (1, 1, nx, ny, nz) volume."""
    coords = torch.arange(-radius, radius + 1, device=vol_5d.device, dtype=vol_5d.dtype)
    k = torch.exp(-0.5 * (coords / sigma) ** 2)
    k = k / k.sum()
    x = F.conv3d(vol_5d, k.view(1, 1, -1, 1, 1), padding=(radius, 0, 0))
    x = F.conv3d(x, k.view(1, 1, 1, -1, 1), padding=(0, radius, 0))
    x = F.conv3d(x, k.view(1, 1, 1, 1, -1), padding=(0, 0, radius))
    return x


@torch.no_grad()
def eval_hash_3d_volume_gpu(
    model: torch.nn.Module,
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    grid_res: int = 96,
    device: Optional[torch.device] = None,
    chunk_size: int = 65536,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]]:
    """Evaluates HashSCKNetwork's continuous 3D field on a voxel grid.

    Returns (v_smooth, v_wide, origin, spacing) so Marching Cubes can apply
    real-time Difference-of-Gaussians clump isolation without re-querying the GPU.
    """
    orig = getattr(model, "_orig_mod", model)
    if not isinstance(orig, HashSCKNetwork):
        return None
    if device is None:
        device = next(orig.parameters()).device

    was_training = orig.training
    orig.eval()

    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
    max_xy = max(sx, sy, 1e-3)
    nx = max(16, int(round(grid_res * (sx / max_xy))))
    ny = max(16, int(round(grid_res * (sy / max_xy))))
    nz = max(16, int(round(grid_res * (sz / max_xy))))

    xs = torch.linspace(x0, x1, nx, device=device, dtype=torch.float32)
    ys = torch.linspace(y0, y1, ny, device=device, dtype=torch.float32)
    zs = torch.linspace(z0, z1, nz, device=device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (N, 3)

    N = pts.shape[0]
    vals = torch.empty((N,), device=device, dtype=torch.float32)
    bound = orig.hash_grid.bound

    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        p = pts[i:j]
        h = orig.hash_grid(p)
        parts = [h, p / bound]
        if orig.use_viewdir:
            parts.append(torch.zeros_like(p))
        if not orig.local_cdf:
            hidden_dim = orig.ray_proj.out_features
            parts.append(torch.zeros((j - i, hidden_dim), device=device, dtype=torch.float32))
        feat = orig.cdf_mlp(torch.cat(parts, dim=-1))
        raw = orig.cdf_out(feat).squeeze(-1)
        if orig.cdf_mode == "softmax":
            # In softmax mode, learned logits span ~[-2 (air), +4 (porous), +6 (pillar), +11 (wall)].
            # Centering & scaling keeps the full physical dynamic range inside (0, 1) instead of
            # saturating everything above +3 into 0.99+.
            vals[i:j] = torch.sigmoid((raw - 3.5) / 2.5)
        else:
            vals[i:j] = 1.0 - torch.exp(-F.softplus(raw))

    if was_training:
        orig.train()

    vol_5d = vals.view(1, 1, nx, ny, nz)
    v_smooth_t = _gaussian_blur_3d(vol_5d, sigma=0.85, radius=2)
    v_wide_t = _gaussian_blur_3d(vol_5d, sigma=3.0, radius=6)

    v_smooth = v_smooth_t[0, 0].cpu().numpy().astype(np.float32)
    v_wide = v_wide_t[0, 0].cpu().numpy().astype(np.float32)

    origin = (x0, y0, z0)
    spacing = (sx / max(1, nx - 1), sy / max(1, ny - 1), sz / max(1, nz - 1))
    return v_smooth, v_wide, origin, spacing


def run_marching_cubes_vtk(
    vol: np.ndarray,
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    iso_level: float = 0.45,
    min_clump_verts: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts a smooth triangle mesh via VTK Flying Edges + connected-component clump pruning.

    Prunes tiny isolated dust floaters (< min_clump_verts) and returns per-vertex clump IDs.
    """
    import vtk
    from vtkmodules.util import numpy_support

    nx, ny, nz = vol.shape
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetOrigin(*origin)
    img.SetSpacing(*spacing)
    flat_f = np.ascontiguousarray(vol.ravel(order="F"), dtype=np.float32)
    vtk_arr = numpy_support.numpy_to_vtk(flat_f, deep=False, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(vtk_arr)

    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, float(iso_level))
    fe.ComputeNormalsOn()
    fe.Update()

    if fe.GetOutput().GetNumberOfPoints() == 0 or fe.GetOutput().GetNumberOfCells() == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint32),
            np.empty((0,), dtype=np.uint32),
        )

    # Label connected 3D clumps so we can drop tiny floaters & support Clump ID coloring
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputConnection(fe.GetOutputPort())
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()

    poly = conn.GetOutput()
    n_pts = poly.GetNumberOfPoints()
    n_tris = poly.GetNumberOfCells()
    if n_pts == 0 or n_tris == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint32),
            np.empty((0,), dtype=np.uint32),
        )

    verts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32, copy=True)
    norms_vtk = poly.GetPointData().GetNormals()
    norms = (
        numpy_support.vtk_to_numpy(norms_vtk).astype(np.float32, copy=True)
        if norms_vtk is not None
        else np.zeros_like(verts)
    )
    regions_vtk = poly.GetPointData().GetArray("RegionId")
    regions = (
        numpy_support.vtk_to_numpy(regions_vtk).astype(np.uint32, copy=True)
        if regions_vtk is not None
        else np.zeros((n_pts,), dtype=np.uint32)
    )
    indices = numpy_support.vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).astype(np.uint32, copy=True)

    # Prune tiny isolated floater specks (< min_clump_verts)
    if min_clump_verts > 0 and len(regions) > 0:
        counts = np.bincount(regions)
        keep_vert = counts[regions] >= min_clump_verts
        if not np.all(keep_vert):
            if not np.any(keep_vert):
                return (
                    np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.float32),
                    np.empty((0,), dtype=np.uint32),
                    np.empty((0,), dtype=np.uint32),
                )
            tris = indices.reshape(-1, 3)
            keep_tri = keep_vert[tris[:, 0]] & keep_vert[tris[:, 1]] & keep_vert[tris[:, 2]]
            tris = tris[keep_tri]
            remap = np.full(len(verts), -1, dtype=np.int64)
            kept_idx = np.nonzero(keep_vert)[0]
            remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int64)
            verts = np.ascontiguousarray(verts[kept_idx])
            norms = np.ascontiguousarray(norms[kept_idx])
            regions = np.ascontiguousarray(regions[kept_idx])
            indices = np.ascontiguousarray(remap[tris].reshape(-1).astype(np.uint32))

    return verts, norms, regions, indices


class OnlineVisibilityGrid:
    """Online GPU-resident 3D observed/unobserved voxel tracker.

    Incrementally carves observed space directly from incoming LiDAR batches
    (~0.7 ms/batch, zero preprocessing required):
      - Hitting rays mark [s_min, hit_range + margin] as observed (1) and leave
        occluded regions beyond the hit surface unobserved (0).
      - Sky / non-returning rays mark [s_min, s_max] as observed free space (1).
    """

    def __init__(
        self,
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float],
        grid_res: int = 96,
        s_min: float = 0.1,
        s_max: float = 15.0,
        samples_per_ray: int = 48,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x0, y0, z0 = bbox_min
        x1, y1, z1 = bbox_max
        sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
        max_xy = max(sx, sy, 1e-3)
        self.nx = max(16, int(round(grid_res * (sx / max_xy))))
        self.ny = max(16, int(round(grid_res * (sy / max_xy))))
        self.nz = max(16, int(round(grid_res * (sz / max_xy))))
        self.origin = (x0, y0, z0)
        self.spacing = (
            sx / max(1, self.nx - 1),
            sy / max(1, self.ny - 1),
            sz / max(1, self.nz - 1),
        )
        self.voxel_size = max(self.spacing)
        self.hit_margin = 1.5 * self.voxel_size  # include hit surface voxel, not behind it
        self.s_min = float(s_min)
        self.s_max = float(s_max)

        self.bbox_min_t = torch.tensor([x0, y0, z0], device=self.device, dtype=torch.float32)
        dims_t = torch.tensor([self.nx, self.ny, self.nz], device=self.device, dtype=torch.float32)
        span_t = torch.tensor([sx, sy, sz], device=self.device, dtype=torch.float32)
        self.inv_span = (dims_t - 1.0) / torch.clamp(span_t, min=1e-6)
        self.t_frac = torch.linspace(0.0, 1.0, samples_per_ray, device=self.device, dtype=torch.float32)

        self.grid = torch.zeros((self.nx * self.ny * self.nz,), device=self.device, dtype=torch.uint8)

    def reset(self) -> None:
        """Clears observed mask back to 0 (e.g. when resetting weights)."""
        self.grid.zero_()

    @torch.no_grad()
    def update_batch(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        hit_range: torch.Tensor,
        hit_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Marks voxels traversed by the observed segment of each ray in the batch."""
        r = hit_range.view(-1)
        if hit_mask is not None:
            m = hit_mask.view(-1)
            t_end = torch.where(
                m > 0.5,
                torch.clamp(r + self.hit_margin, max=self.s_max),
                torch.full_like(r, self.s_max),
            )
        else:
            t_end = torch.clamp(r + self.hit_margin, max=self.s_max)

        ts = self.s_min + self.t_frac.unsqueeze(0) * torch.clamp(t_end - self.s_min, min=0.0).unsqueeze(1)
        pts = ray_o.unsqueeze(1) + ts.unsqueeze(-1) * ray_d.unsqueeze(1)  # (B, S, 3)
        ijk = torch.round((pts - self.bbox_min_t) * self.inv_span).long()
        inb = (
            (ijk[..., 0] >= 0) & (ijk[..., 0] < self.nx)
            & (ijk[..., 1] >= 0) & (ijk[..., 1] < self.ny)
            & (ijk[..., 2] >= 0) & (ijk[..., 2] < self.nz)
        )
        ijk_v = ijk[inb]
        if ijk_v.numel() > 0:
            flat = (ijk_v[:, 0] * self.ny + ijk_v[:, 1]) * self.nz + ijk_v[:, 2]
            self.grid[flat] = 1

    @torch.no_grad()
    def get_observed_mask_numpy(self, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Returns a float32 [nx, ny, nz] observed mask resampled to target_shape."""
        obs_5d = self.grid.view(1, 1, self.nx, self.ny, self.nz).float()
        if target_shape != (self.nx, self.ny, self.nz):
            obs_5d = F.interpolate(obs_5d, size=target_shape, mode="trilinear", align_corners=True)
        # Small 1-voxel dilation/blur keeps hit boundaries smooth without leaking into shadows
        obs_5d = _gaussian_blur_3d(obs_5d, sigma=0.7, radius=1)
        return (obs_5d[0, 0] > 0.15).float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_unobserved_mesh_vtk(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Extracts a smooth 3D Marching Cubes mesh enclosing all UNOBSERVED voxels."""
        obs_5d = self.grid.view(1, 1, self.nx, self.ny, self.nz).float()
        observed_pct = float(obs_5d.mean().item() * 100.0)
        unobs_5d = 1.0 - obs_5d
        unobs_smooth = _gaussian_blur_3d(unobs_5d, sigma=0.8, radius=2)[0, 0].cpu().numpy().astype(np.float32)

        # Close outer 1-voxel boundary so blind cones & pillar shadows form crisp closed shells
        unobs_smooth[0, :, :] = 0.0
        unobs_smooth[-1, :, :] = 0.0
        unobs_smooth[:, 0, :] = 0.0
        unobs_smooth[:, -1, :] = 0.0
        unobs_smooth[:, :, 0] = 0.0
        unobs_smooth[:, :, -1] = 0.0

        verts, norms, _, indices = run_marching_cubes_vtk(
            unobs_smooth,
            self.origin,
            self.spacing,
            iso_level=0.5,
            min_clump_verts=40,
        )
        return verts, norms, indices, observed_pct


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
        self.sensor_dirty_event = threading.Event()
        self.mesh_dirty_event = threading.Event()
        self.unobs_dirty_event = threading.Event()

        self.scene_json = json.dumps(scene_payload).encode("utf-8")
        self.render_method = args.render_method
        self.render_elev_min = float(scene_payload["render_elev_min"])
        self.render_elev_max = float(scene_payload["render_elev_max"])
        self.render_channels = int(scene_payload["render_channels"])
        self.mesh_enabled = args.point_style in ("mesh", "mesh+patches")
        self.mesh_iso = float(args.mesh_iso)
        self.clump_isolate = float(args.clump_isolate)
        self.mask_unobserved = not args.no_mask_unobserved
        self.mesh_res = int(args.mesh_res)
        self.mesh_version = 0
        self.latest_mesh_blob: Optional[bytes] = None
        self.unobs_enabled = bool(args.show_unobserved)
        self.unobs_version = 0
        self.latest_unobs_blob: Optional[bytes] = None
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
        self._last_norms = np.empty((0, 3), dtype=np.float32)
        self._last_rngs = np.empty((0,), dtype=np.float32)
        self._last_probs = np.empty((0,), dtype=np.float32)

    def _pack_blob(
        self,
        meta: Dict[str, Any],
        pts: np.ndarray,
        norms: np.ndarray,
        rngs: np.ndarray,
        probs: np.ndarray,
    ) -> bytes:
        meta_bytes = json.dumps(meta).encode("utf-8")
        pad = (-len(meta_bytes)) % 4
        header = struct.pack("<I", len(meta_bytes)) + meta_bytes + (b"\x00" * pad)
        return header + pts.tobytes() + norms.tobytes() + rngs.tobytes() + probs.tobytes()

    def publish(
        self,
        pts: np.ndarray,
        norms: np.ndarray,
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
            self._last_norms = norms
            self._last_rngs = rngs
            self._last_probs = probs
            self.latest_meta = meta
            self.latest_blob = self._pack_blob(meta, pts, norms, rngs, probs)

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
            self.latest_blob = self._pack_blob(
                meta, self._last_pts, self._last_norms, self._last_rngs, self._last_probs
            )

    def get_frame_if_newer(self, client_version: int) -> Optional[bytes]:
        with self.lock:
            if self.latest_blob is None or self.version == client_version:
                return None
            return self.latest_blob

    def publish_mesh(
        self,
        verts: np.ndarray,
        norms: np.ndarray,
        regions: np.ndarray,
        indices: np.ndarray,
        mesh_ms: float,
        iso_level: float,
        grid_res: int,
        epoch_float: float,
    ) -> None:
        with self.lock:
            self.mesh_version += 1
            meta = {
                "version": self.mesh_version,
                "num_verts": int(len(verts)),
                "num_tris": int(len(indices) // 3),
                "mesh_ms": float(mesh_ms),
                "iso_level": float(iso_level),
                "grid_res": int(grid_res),
                "epoch_float": float(epoch_float),
            }
            meta_bytes = json.dumps(meta).encode("utf-8")
            pad = (-len(meta_bytes)) % 4
            header = struct.pack("<I", len(meta_bytes)) + meta_bytes + (b"\x00" * pad)
            self.latest_mesh_blob = (
                header + verts.tobytes() + norms.tobytes() + regions.tobytes() + indices.tobytes()
            )

    def get_mesh_if_newer(self, client_version: int) -> Optional[bytes]:
        with self.lock:
            if self.latest_mesh_blob is None or self.mesh_version == client_version:
                return None
            return self.latest_mesh_blob

    def publish_unobs(
        self,
        verts: np.ndarray,
        norms: np.ndarray,
        indices: np.ndarray,
        observed_pct: float,
        epoch_float: float,
    ) -> None:
        with self.lock:
            self.unobs_version += 1
            meta = {
                "version": self.unobs_version,
                "num_verts": int(len(verts)),
                "num_tris": int(len(indices) // 3),
                "observed_pct": float(observed_pct),
                "epoch_float": float(epoch_float),
            }
            meta_bytes = json.dumps(meta).encode("utf-8")
            pad = (-len(meta_bytes)) % 4
            header = struct.pack("<I", len(meta_bytes)) + meta_bytes + (b"\x00" * pad)
            self.latest_unobs_blob = header + verts.tobytes() + norms.tobytes() + indices.tobytes()

    def get_unobs_if_newer(self, client_version: int) -> Optional[bytes]:
        with self.lock:
            if self.latest_unobs_blob is None or self.unobs_version == client_version:
                return None
            return self.latest_unobs_blob


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
            elif parsed.path == "/api/mesh":
                qs = urllib.parse.parse_qs(parsed.query)
                try:
                    client_v = int(qs.get("v", ["-1"])[0])
                except ValueError:
                    client_v = -1
                blob = state.get_mesh_if_newer(client_v)
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
            elif parsed.path == "/api/unobs":
                qs = urllib.parse.parse_qs(parsed.query)
                try:
                    client_v = int(qs.get("v", ["-1"])[0])
                except ValueError:
                    client_v = -1
                blob = state.get_unobs_if_newer(client_v)
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
                payload = json.loads(body.decode("utf-8"))
                cmd = payload.get("action", "")
            except Exception:
                payload = {}
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
            elif cmd == "set_sensor":
                with state.lock:
                    emin = float(payload.get("elev_min", state.render_elev_min))
                    emax = float(payload.get("elev_max", state.render_elev_max))
                    ch = int(payload.get("num_detectors", state.render_channels))
                    emin = max(-80.0, min(79.0, emin))
                    emax = max(emin + 0.5, min(80.0, emax))
                    ch = max(2, min(256, ch))
                    state.render_elev_min = emin
                    state.render_elev_max = emax
                    state.render_channels = ch
                state.sensor_dirty_event.set()
                state.rerender_event.set()
            elif cmd == "set_mesh":
                with state.lock:
                    if "enabled" in payload:
                        state.mesh_enabled = bool(payload["enabled"])
                    if "iso_level" in payload:
                        state.mesh_iso = max(0.01, min(0.99, float(payload["iso_level"])))
                    if "clump_isolate" in payload:
                        state.clump_isolate = max(0.0, min(1.0, float(payload["clump_isolate"])))
                    if "mask_unobserved" in payload:
                        state.mask_unobserved = bool(payload["mask_unobserved"])
                    if "grid_res" in payload:
                        state.mesh_res = max(32, min(160, int(payload["grid_res"])))
                state.mesh_dirty_event.set()
                state.rerender_event.set()
            elif cmd == "set_unobs":
                with state.lock:
                    state.unobs_enabled = bool(payload.get("enabled", False))
                state.unobs_dirty_event.set()
                state.rerender_event.set()

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
    parser.add_argument("--s-max", type=float, default=15.0, help="Max ray range in meters (default: 15.0)")
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
    parser.add_argument("--render-channels", type=int, default=None, help="Initial vertical channels for live render (default: match training sensor)")
    parser.add_argument("--render-elev-min", type=float, default=None, help="Initial min elevation angle in degrees for live render (default: match training sensor)")
    parser.add_argument("--render-elev-max", type=float, default=None, help="Initial max elevation angle in degrees for live render (default: match training sensor)")
    parser.add_argument("--render-method", type=str, choices=["sample", "threshold"], default="sample", help="CDF point extraction mode (default: sample)")
    parser.add_argument("--threshold", type=float, default=0.5, help="CDF threshold when --render-method=threshold (default: 0.5)")
    parser.add_argument("--non-return-threshold", type=float, default=0.1, help="Prune ray if P(non-return) > threshold (default: 0.1)")
    parser.add_argument("--color-by", type=str, choices=["range", "z", "probability", "normal", "clump"], default="range", help="Initial point color scalar (default: range)")
    parser.add_argument("--point-style", type=str, choices=["points", "spheres", "patches", "mesh", "mesh+patches"], default="patches", help="Initial visual style (default: patches)")
    parser.add_argument("--mesh-iso", type=float, default=0.45, help="Initial Marching Cubes iso-level in (0, 1) (default: 0.45)")
    parser.add_argument("--clump-isolate", type=float, default=0.45, help="Difference-of-Gaussians clump isolation strength in [0, 1] (default: 0.45)")
    parser.add_argument("--mesh-res", type=int, default=96, help="Initial Marching Cubes grid resolution (default: 96)")
    parser.add_argument("--show-unobserved", action="store_true", help="Show translucent 3D overlay of unobserved/blind-zone regions on startup")
    parser.add_argument("--no-mask-unobserved", action="store_true", help="Do not mask Marching Cubes mesh to online-observed voxels")
    parser.add_argument("--gt-style", type=str, choices=["transparent", "both", "wireframe", "off"], default="transparent", help="Initial ground-truth scene overlay style (default: transparent)")
    parser.add_argument("--no-gt-scene", action="store_true", help="Hide ground-truth room overlay on startup (alias for --gt-style off)")
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

    # 3. Pre-cache novel-view rays on GPU (dynamically rebuilt when user adjusts HUD sliders)
    init_elev_min = float(args.render_elev_min) if args.render_elev_min is not None else float(train_sensor.elevation_min_deg)
    init_elev_max = float(args.render_elev_max) if args.render_elev_max is not None else float(train_sensor.elevation_max_deg)
    init_channels = int(args.render_channels) if args.render_channels is not None else int(train_sensor.num_detectors)

    def build_render_rays(elev_min: float, elev_max: float, num_channels: int):
        rs = LidarProjectionModel(
            elevation_min=elev_min,
            elevation_max=elev_max,
            num_detectors=num_channels,
            num_azimuth=args.render_azimuth,
            max_range=train_sensor.max_range,
        )
        ro_np, rd_np = rs.get_rays(sensor_position=args.render_pos, flatten=True)
        return (
            torch.from_numpy(ro_np).float().to(device),
            torch.from_numpy(rd_np).float().to(device),
            (rs.num_detectors, rs.num_azimuth),
        )

    render_rays_o, render_rays_d, render_grid_shape = build_render_rays(
        init_elev_min, init_elev_max, init_channels
    )
    print(
        f"Live render viewpoint: {args.render_pos} "
        f"({render_grid_shape[0]}x{render_grid_shape[1]} = {render_rays_o.shape[0]:,} rays, "
        f"elev=[{init_elev_min:.1f}°, {init_elev_max:.1f}°])",
        flush=True,
    )

    # 4. Start Three.js HTTP server + online GPU 3D visibility tracker
    mesh_bbox_min, mesh_bbox_max = compute_scene_bbox(scene_cfg, args.s_max)
    vis_grid = OnlineVisibilityGrid(
        bbox_min=mesh_bbox_min,
        bbox_max=mesh_bbox_max,
        grid_res=96,
        s_min=0.1,
        s_max=args.s_max,
        samples_per_ray=48,
        device=device,
    )
    cached_vol_info: Dict[str, Any] = {
        "v_smooth": None,
        "v_wide": None,
        "origin": None,
        "spacing": None,
        "step": -1,
        "res": -1,
    }

    scene_payload = {
        "model": args.model,
        "s_max": args.s_max,
        "render_pos": args.render_pos,
        "no_gt_scene": args.no_gt_scene,
        "gt_style": args.gt_style,
        "point_style": args.point_style,
        "color_by": args.color_by,
        "render_elev_min": init_elev_min,
        "render_elev_max": init_elev_max,
        "render_channels": init_channels,
        "mesh_iso": args.mesh_iso,
        "clump_isolate": args.clump_isolate,
        "mask_unobserved": not args.no_mask_unobserved,
        "show_unobserved": args.show_unobserved,
        "mesh_res": args.mesh_res,
        "scene_objects": scene_cfg.get("scene_objects", []),
    }
    state = LiveServerState(scene_payload=scene_payload, args=args)
    server = start_http_server(state, args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"\n>>> Live 3D Viewer running at: {url}  (Ctrl+C to stop) <<<\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    def push_unobs(epoch_float: float) -> None:
        with state.lock:
            enabled = state.unobs_enabled
        if not enabled:
            return
        verts, norms, indices, obs_pct = vis_grid.extract_unobserved_mesh_vtk()
        state.publish_unobs(
            verts=verts,
            norms=norms,
            indices=indices,
            observed_pct=obs_pct,
            epoch_float=epoch_float,
        )

    def push_mesh(epoch_float: float, step: int, force_eval: bool = False) -> None:
        with state.lock:
            enabled = state.mesh_enabled
            iso = state.mesh_iso
            clump = state.clump_isolate
            mask_unobs = state.mask_unobserved
            res = state.mesh_res
        if not enabled:
            return
        t_m0 = time.perf_counter()
        need_vol = (
            force_eval
            or cached_vol_info["v_smooth"] is None
            or cached_vol_info["step"] != step
            or cached_vol_info["res"] != res
        )
        if need_vol:
            vol_out = eval_hash_3d_volume_gpu(
                model=model,
                bbox_min=mesh_bbox_min,
                bbox_max=mesh_bbox_max,
                grid_res=res,
                device=device,
            )
            if vol_out is None:
                return
            v_smooth, v_wide, origin, spacing = vol_out
            cached_vol_info["v_smooth"] = v_smooth
            cached_vol_info["v_wide"] = v_wide
            cached_vol_info["origin"] = origin
            cached_vol_info["spacing"] = spacing
            cached_vol_info["step"] = step
            cached_vol_info["res"] = res

        # Difference-of-Gaussians local peak sharpening: suppresses broad low-density
        # shadow bridges / blind-cone humps so foreground objects isolate cleanly from walls
        vol_eff = np.clip(
            cached_vol_info["v_smooth"] - (clump * 0.75) * cached_vol_info["v_wide"],
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        if mask_unobs and step > 0:
            obs_mask = vis_grid.get_observed_mask_numpy(vol_eff.shape)
            vol_eff = vol_eff * obs_mask
        # Zero outer 1-voxel shell so Marching Cubes closes cleanly at bbox walls
        vol_eff[0, :, :] = 0.0
        vol_eff[-1, :, :] = 0.0
        vol_eff[:, 0, :] = 0.0
        vol_eff[:, -1, :] = 0.0
        vol_eff[:, :, 0] = 0.0
        vol_eff[:, :, -1] = 0.0

        verts, norms, regions, indices = run_marching_cubes_vtk(
            vol_eff,
            cached_vol_info["origin"],
            cached_vol_info["spacing"],
            iso_level=iso,
        )
        mesh_ms = (time.perf_counter() - t_m0) * 1000.0
        state.publish_mesh(
            verts=verts,
            norms=norms,
            regions=regions,
            indices=indices,
            mesh_ms=mesh_ms,
            iso_level=iso,
            grid_res=res,
            epoch_float=epoch_float,
        )

    def push_render(
        epoch_float: float,
        step: int,
        loss_val: float,
        rps: float,
        status: str,
        append_loss: bool = True,
    ) -> None:
        nonlocal render_rays_o, render_rays_d, render_grid_shape
        aux_only_dirty = (
            (state.mesh_dirty_event.is_set() or state.unobs_dirty_event.is_set())
            and not state.sensor_dirty_event.is_set()
        )
        if state.mesh_dirty_event.is_set():
            state.mesh_dirty_event.clear()
        if state.unobs_dirty_event.is_set():
            state.unobs_dirty_event.clear()
        if state.sensor_dirty_event.is_set():
            state.sensor_dirty_event.clear()
            with state.lock:
                emin = state.render_elev_min
                emax = state.render_elev_max
                ch = state.render_channels
            render_rays_o, render_rays_d, render_grid_shape = build_render_rays(emin, emax, ch)
        with state.lock:
            method = state.render_method
        if not (aux_only_dirty and not append_loss):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_r0 = time.perf_counter()
            pts, norms, rngs, probs = render_cloud_fast_gpu(
                model=model,
                render_rays_o=render_rays_o,
                render_rays_d=render_rays_d,
                grid_shape=render_grid_shape,
                method=method,
                threshold=args.threshold,
                non_return_threshold=args.non_return_threshold,
                batch_size=args.batch_size,
            )
            render_ms = (time.perf_counter() - t_r0) * 1000.0
            state.publish(
                pts=pts,
                norms=norms,
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
        push_mesh(epoch_float, step)
        push_unobs(epoch_float)

    # 5. Train directly on the main thread at 100% native GPU speed
    try:
        while not state.stop_event.is_set():
            optimizer, scheduler = make_optimizer_and_scheduler(model, args, device)
            global_step = 0
            last_loss = float("nan")
            last_rps = 0.0

            with state.lock:
                state.loss_history.clear()
            vis_grid.reset()
            cached_vol_info["v_smooth"] = None
            cached_vol_info["v_wide"] = None
            cached_vol_info["step"] = -1
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

                    # Online visibility tracking directly from incoming rays (~0.7 ms/batch)
                    vis_grid.update_batch(ray_o, ray_d, hit_range, hit_mask)

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
