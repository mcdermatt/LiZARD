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

The SCK knot refinement decoder is structurally identical to a **PointNet evaluated along a 1D ray manifold**:

1. **Global Ray Context (PointNet Global Feature)**:
   - Evaluated once per ray: $(\mathbf{o}, \mathbf{d}, \mathbf{m} = \mathbf{o} \times \mathbf{d}) \xrightarrow{\text{PE + MLP}} \mathbf{f}_{\text{ray}} \in \mathbb{R}^{\text{hidden\_dim}}$.
2. **Local Knot Features (PointNet Local Point Feature)**:
   - For every knot $k \in \{1, \dots, K\}$, 3D point $\mathbf{x}_k = \mathbf{o} + s_k \mathbf{d} \xrightarrow{\text{PE + MLP}} \mathbf{f}_{\text{knot}, k} \in \mathbb{R}^{\text{hidden\_dim}}$.
3. **Global + Local Concatenation**:
   - Ray context is broadcast to every candidate knot: $[\mathbf{f}_{\text{ray}}, \mathbf{f}_{\text{knot}, k}] \in \mathbb{R}^{2 \cdot \text{hidden\_dim}}$.
4. **Shared MLP Trunk (PointNet 1x1 Shared Convolution)**:
   - Evaluates all knots simultaneously using the **exact same shared weights** across $K$:
     $$(\Delta s_k, \sigma_k) = \text{MLP}_{\text{shared}}([\mathbf{f}_{\text{ray}}, \mathbf{f}_{\text{knot}, k}])$$
5. **1D Light Ray Transmittance Constraint**:
   - Unlike standard PointNet where points are unordered, the knots interact via the physics of light travel:
     - Differentiable 1D spatial sort: $s_{(0)} \le s_{(1)} \le \dots \le s_{(K)}$ (permutes densities accordingly).
     - Monotonic Beer-Lambert integration: $C(s) = 1 - \exp(-\sum \sigma_k \Delta s_k)$.

**Key Advantages**:
- **Knot-Count Agnostic ($K$ is not in the weight matrix)**: Linear layers operate strictly on the trailing feature dimension. The model parameter count is completely independent of $K$.
- **True Continuous Field / Spatial Generalization**: The network never memorizes "knot index #5 at 2.0 meters"; it learns a unified 3D vector field $\vec{v}(x, y, z) = (\text{shift}, \text{density})$.
- **Resolution Flexibility**: Can train efficiently with $K=16$ or $K=32$ knots for fast epoch throughput, and evaluate/render with arbitrary $K$ at test time without altering model weights.

---

### Architectural Note: Exact Grid-Free Analytic Loss (Closed-Form Wasserstein-1 & CRPS)

Instead of discretizing space with an artificial evaluation grid (e.g. 64 or 256 ticks), the continuous CDF loss against a LiDAR return at $r^*$ is computed **analytically in closed form**:
$$\mathcal{L} = \int_{0}^{s_{\max}} \left| F_{\text{spline}}(t) - \mathbf{1}_{t \ge r^*} \right| dt = \int_0^{r^*} F(t) dt + \int_{r^*}^{s_{\max}} (1 - F(t)) dt$$

- **Polygon Trapezoids**: Because $F_{\text{spline}}$ is piecewise-linear, the exact integral over each segment $[s_{k-1}, s_k]$ is given by simple trapezoids $\frac{c_{k-1} + c_k}{2} \Delta s_k$ (and exact quadratic polynomials for continuous $L_2$).
- **Zero Discretization Error**: Has infinite effective spatial resolution ($0.0\,\text{mm}$ error), eliminating the surface-snapping blind spots caused by discrete grids.
- **Memory & Compute Efficiency**: Eliminates storing or loading pre-computed $(N, M)$ CDF matrices in VRAM and removes `interpolate_cdf_to_grid` from the inner training loop. Origin sliding simply computes $r^* \leftarrow r^* - \delta$.

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
   - Compiling `DeeperSCKNetwork` via PyTorch 2.x Inductor fuses positional encodings, layer norms, and MLP activations into unified Triton CUDA kernels.
   - Forward pass latency drops from $15.1\,\text{ms}$ to $8.4\,\text{ms}$ ($1.8\times$ faster).

4. **Precision Considerations (FP32 vs FP16 AMP)**:
   - For compact coordinate networks ($\sim 17\text{k} - 48\text{k}$ parameters), standard FP32 runs faster than FP16 Automatic Mixed Precision (AMP).
   - In micro-networks, the kernel launching overhead of dtype casting and loss scaling (`GradScaler`) outweighs any tensor-core savings. Pure FP32 minimizes latency while preserving gradient precision for positional encodings.
