https://en.wikipedia.org/wiki/X-ray_transform 
https://en.wikipedia.org/wiki/Radon_transform

plucker vs ray space model
Ray-space: $\Phi(\ell,t)$ with $\ell$ as input
Plucker space: $\Phi(\mathbf d, \mathbf m, t)$ with $(\mathbf d, \mathbf m)$ as input


use accumulation in x and y to move around control points for CDF 

default strategy (grounded output):
(o, d) -> same as normal NeRF

* output CDF would be [0,1] where 1 is the full size of the world

* output grounded in real world but input is ungrounded

* need to have direction as input for training since we need to be able to query this at arbitrary directions at render time

* Self-Conditioned / Recurrent 3D Point-Transformer?


Alternate strategy (grounded input):
(o, p) -> pass in origin and arbitary point of collision

* output is normalized to [0, p]

* grounds us in 3d geometry for the input, output is not very grounded

* augment training data to get (o, 1.1*p), (o, 1.2*p), etc. with identical CDF output to (o,p), except more time spent at 1 at the end 

* also add training data to get (o, 0.9*p) , etc with all zeros in the CDF

* Shared anchor grid??



Consider local batching for more efficient learning of CDFs?
- do experiment with this

---

### Architectural Note: Shared-Weight Knot Flow (PointNet along a 1D Light Ray)

The SCK knot refinement decoder (`DeeperSCKNetwork` in `sck_model.py`) is structurally identical to a **PointNet evaluated along a 1D ray manifold**, plus a PLiNK-style global ray-drop head:

1. **Global Ray Context (PointNet Global Feature)**:
   - Evaluated once per ray: $(\mathbf{o}, \mathbf{d}, \mathbf{m} = \mathbf{o} \times \mathbf{d}) \xrightarrow{\text{PE (10 freqs) + MLP}} \mathbf{f}_{\text{ray}} \in \mathbb{R}^{\text{hidden\_dim}}$.
   - A dedicated `ray_drop_head` evaluates $\mathbf{f}_{\text{ray}} \to p_{\text{hit}} \in [0, 1]$ to classify hitting vs. sky/non-return rays.
2. **Local Knot Features (PointNet Local Point Feature)**:
   - For every knot $k \in \{0, \dots, K-1\}$, 3D point $\mathbf{x}_k = \mathbf{o} + s_k \mathbf{d} \xrightarrow{\text{PE (10 freqs) + MLP}} \mathbf{f}_{\text{knot}, k} \in \mathbb{R}^{\text{hidden\_dim}}$.
3. **Global + Local Concatenation**:
   - Ray context is broadcast to every candidate knot: $[\mathbf{f}_{\text{ray}}, \mathbf{f}_{\text{knot}, k}] \in \mathbb{R}^{2 \cdot \text{hidden\_dim}}$.
4. **Shared Residual MLP Trunk (PointNet 1x1 Shared Convolution)**:
   - Evaluates all knots simultaneously using the **exact same shared weights** across $K$ (`joint_in` + `num_hidden_blocks` residual ReLU blocks + `out_head`):
     $$(u_k, v_k) = \text{MLP}_{\text{shared}}([\mathbf{f}_{\text{ray}}, \mathbf{f}_{\text{knot}, k}])$$
5. **1D Monotonic Knot Flow & CDF Construction**:
   - **Order-Preserving Gap-Bounded Shifts (No Sorting Needed)**: Endpoints stay pinned at $s_0 = s_{\min}$ and $s_{K-1} = s_{\max}$. Each interior knot shifts by $\delta s_k = \tanh(u_k) \cdot \min(0.45 \min(\Delta s_k^{\text{left}}, \Delta s_k^{\text{right}}), d_{\max})$, guaranteeing strict monotonic ordering and preventing knot bunching without sorting. Refined knot positions are detached (`s_knots.detach()`) between rounds.
   - **Default Interval Softmax CDF (`cdf_mode="softmax"`)**: Predicts probability mass per interval $[s_{k-1}, s_k]$ via softmax over $v_1, \dots, v_{K-1}$ plus a fixed `0.0` escape bin logit, cumulatively summed from $C_0 = 0$ at $s_{\min}$. Avoids exponential density saturation and vanishing gradients.
   - **Legacy Beer–Lambert Mode (`cdf_mode != "softmax"`)**: Integrates $\lambda_k = \operatorname{softplus}(v_k)$ via $C_k = 1 - \exp(-\sum \lambda_j \max(\Delta s_j, 10^{-4}))$.

**Key Advantages**:
- **Knot-Count Agnostic ($K$ is not in the weight matrix)**: Linear layers operate strictly on the trailing feature dimension. The model parameter count is completely independent of $K$.
- **True Continuous Field / Spatial Generalization**: The network never memorizes "knot index #5 at 2.0 meters"; it learns a unified shared per-knot update over world coordinates and ray context.
- **Resolution Flexibility**: Can train efficiently with $K=16$ or $K=32$ knots for fast epoch throughput, and evaluate/render with arbitrary $K$ at test time without altering model weights.

---

### Architectural Note: Exact Grid-Free Analytic Loss (Closed-Form Root-CRPS & Wasserstein-1)

Instead of discretizing space with an artificial evaluation grid (e.g. 64 or 256 ticks), `exact_continuous_cdf_loss` computes the continuous CDF error against a LiDAR return at $r^*$ **analytically in closed form** by inserting $(r^*, F(r^*))$ into the piecewise-linear spline knots:
$$\mathcal{L}_1 = \int_{0}^{s_{\max}} \left| F_{\text{spline}}(t) - \mathbf{1}_{t \ge r^*} \right| dt, \qquad \mathcal{L}_2 = \int_{0}^{s_{\max}} \left( F_{\text{spline}}(t) - \mathbf{1}_{t \ge r^*} \right)^2 dt$$

- **Exact Piecewise Integration**: Because $F_{\text{spline}}$ is piecewise-linear, $\mathcal{L}_1$ (Wasserstein-1) is exact via trapezoids $\frac{e_l + e_r}{2} \Delta s$ and $\mathcal{L}_2$ (CRPS) is exact via piecewise-quadratic integrals $\frac{\Delta s}{3}(e_l^2 + e_l e_r + e_r^2)$.
- **Root-CRPS Training Objective**: In grid-free mode, `compute_sck_loss` optimizes **Root-CRPS** $\sqrt{\mathcal{L}_2 + 10^{-6}}$ per round (masked strictly to hitting rays via `hit_mask`, with geometric round weights $0.5^{R - 1 - r}$) plus BCE ray-drop loss on `p_hit`. L1 is logged as a metric only, because optimizing L1 causes the network to skip semi-transparent surfaces.
- **Zero Discretization Error**: Has infinite effective spatial resolution ($0.0\,\text{mm}$ error), eliminating the surface-snapping blind spots caused by discrete grids.
- **Memory & Compute Efficiency**: Eliminates storing or loading pre-computed $(N, M)$ CDF matrices in VRAM and removes `interpolate_cdf_to_grid` from the inner training loop. Origin sliding simply computes $r^* \leftarrow r^* - \Delta$.

---

### Architectural Note: Training Throughput Optimizations (GPU Batcher & Kernel Fusion)

To maximize training throughput on resource-constrained hardware (e.g. NVIDIA GTX 1650 Mobile with 4GB VRAM):

1. **Epoch-Level Permutation & Zero-Copy Slicing (`GPURayBatcher`)**:
   - Rather than launching indexing/gather kernels every single batch (`tensor[batch_indices]`), rays are permuted once at the start of each epoch (`tensor[perm]`).
   - Batches are yielded as contiguous memory views (`tensor[i : i + B]`), eliminating per-batch CUDA memory allocations and dropping batch iteration overhead from $3.74\,\text{ms}$ to $0.55\,\text{ms}$ ($6.8\times$ faster).

2. **Optimal Batch Sizing ($B=1024$)**:
   - Small batch sizes ($B=256$) severely underutilize CUDA streaming multiprocessors ($33.8\,\text{ms/step}$, $\sim 7,560\,\text{rays/sec}$).
   - Increasing batch size to $B=1024$ takes virtually identical execution time per step ($35.8\,\text{ms/step}$) while shoveling $4\times$ more rays per step, pushing throughput to **$>20,000\,\text{rays/sec}$** ($3.8\times$ real throughput gain).

3. **`torch.compile` CUDA Kernel Fusion**:
   - Compiling `DeeperSCKNetwork` via PyTorch 2.x Inductor fuses positional encodings, linear layers, and MLP activations into unified Triton CUDA kernels.
   - Forward pass latency drops from $15.1\,\text{ms}$ to $8.4\,\text{ms}$ ($1.8\times$ faster).

4. **Precision Considerations (FP32 vs FP16 AMP)**:
   - For compact coordinate MLPs, standard FP32 runs faster than FP16 Automatic Mixed Precision (AMP) on mobile GPUs.
   - The kernel launching overhead of dtype casting and loss scaling (`GradScaler`) outweighs tensor-core savings, and pure FP32 preserves gradient precision for high-frequency positional encodings.

---

### Paper Framing Note: Why Decoupled Hash-SCK Matters for Robotics (vs. PLiNK)

Because `HashSCKNetwork` puts self-conditioned knot placement **in front of** a purely 3D-local readout (`local_cdf=True`: `hash_grid(x) -> cdf_mlp -> cdf_out`), training from raw LiDAR rays automatically carves a **standalone, sensor-model-free 3D volumetric hazard field** alongside the fast ray-CDF predictor:

- **No Multi-View Scan Synthesis & Stitching Required:** Unlike workflows that render novel LiDAR point clouds from multiple viewpoints and merge them into a map, you can query $f(\mathbf{x})$ directly at any $(x, y, z)$ in $O(1)$—enabling instant Marching Cubes meshing (`~20-30 ms` for a 96³ room grid), 3D/2D costmap extraction, and microsecond trajectory collision checking.
- **Continuous Traversability (Hard vs. Porous Media):** Preserves volumetric return density so downstream planners can separate impenetrable walls/pillars from penetrable vegetation/foliage and free space.
- **Retains Fast Sensor-Frame Queries:** Still supports 2-pass ray CDF / likelihood evaluation for PLiNK-style LiDAR pose registration and active visibility planning, without PLiNK's expensive dense coarse-to-fine ray marching (10–50× faster training & inference).
- **Important Caveats to Address in Paper / Robotics Stack:**
  1. **Calibration (`hazard` vs. `softmax`):** `--cdf-mode hazard` directly yields physical extinction $\lambda(\mathbf{x}) \ge 0\,[\text{m}^{-1}]$ (`softplus`), whereas `softmax` logits are relative to the fixed `0.0` escape bin and shift upward on solid walls (`+5` to `+11`).
  2. **Epistemic Uncertainty (Observed vs. Unobserved):** Direct 3D Marching Cubes immediately exposes unsupervised values in sensor blind cones and occlusion shadows behind pillars. Pairing the hash grid with a lightweight 3D ray-visitation/observation mask separates *known-empty* from *unobserved* space for safe motion planning.
