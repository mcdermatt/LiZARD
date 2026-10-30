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