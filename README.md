# LiZARD
Exploring LiDAR-Based Neural Hazard Fields


## Summary

1. **Exact segment additivity is easy, but it isn't the hard problem.** If $A$ is written as a monotone per-line potential $\Phi(\ell,t)$, you get nonnegativity, monotonicity, $A(\mathbf a,\mathbf a)=0$, additivity and reversal symmetry by construction. No regularization is needed, and the network runs once per ray: every query distance is then closed-form.
2. **None of your three losses touches cross-ray consistency.** The additivity, symmetry and CDF terms each act on points of a single line. A model that satisfies all of them exactly can still give unrelated hazards to the same voxel on two crossing rays. Additivity buys independence from the sensor origin, not independence from direction.
3. **Exact cross-ray consistency is the same thing as being a line integral of a 3D field** (Prop. 2 below). So you have two choices. You can integrate, either numerically or analytically over a basis with closed-form line integrals. Or you can accept approximate consistency, enforced by a regularizer that has to cover 4D line space.
4. **My recommendation is to integrate analytically, with Gaussian hazard primitives.** Hazards add, so there is no sorting and no alpha-compositing approximation: Beer–Lambert holds exactly. Every property you listed is guaranteed, and evaluation needs no MLP. This is probably the fastest consistent option, and you can add view-dependence where it's physically motivated (foliage).

I couldn't read files in the LiZARD workspace (permission denied), so this is based only on your description.

---

## 1. What additivity actually gives you

**Prop. 1 (additivity means a per-line potential).** Assume $A\ge 0$ is continuous and additive on ordered collinear triples. For each line $\ell$, pick a reference point $\mathbf r_\ell$ and arc-length coordinate $t$, and set $\Phi(\ell,t)=\pm A(\mathbf r_\ell,\mathbf x(t))$, signed by side. Then

$$A(\mathbf a,\mathbf b)=\big|\Phi(\ell,t_b)-\Phi(\ell,t_a)\big|,$$

with $\Phi(\ell,\cdot)$ nondecreasing. If $\Phi$ is differentiable in $t$, the implied hazard

$$\hat\lambda(\mathbf x,\mathbf d) := \partial_t\Phi(\ell(\mathbf x,\mathbf d),t)\big|_{t=\mathbf x\cdot\mathbf d}$$

gives $A(\mathbf a,\mathbf b)=\int_{\mathbf a}^{\mathbf b}\hat\lambda(\mathbf p,\hat{\mathbf d})\,d\ell$. This is just the fundamental theorem of calculus on each line. More generally $\Phi$ is a monotone function of bounded variation, so $d\Phi$ can include atoms: perfectly sharp partial reflectors.

Consequences:

- **The additive models are exactly the direction-dependent 5D hazard fields $\hat\lambda(\mathbf x,\pm\mathbf d)$ written through their per-line antiderivatives.** Origin-independence is automatic, because the origin only moves along the line. What's missing is direction-independence.
- Your loss $\mathcal L_{\text{add}}$ spends network capacity learning a functional equation that can be built into the architecture for free.
- **Distinct lines share at most one point, so no two segments on different lines overlap.** "Overlapping rays" in the LiDAR sense means nearby lines crossing the same neighbourhood. Consistency between them is a condition on derivatives, not on segments.

AutoInt (Lindell, Martel & Wetzstein, CVPR 2021) is exactly this construction for volume rendering. It trains a ray-conditioned antiderivative network and differentiates it to get the integrand, and it inherits the same lack of consistency across rays. Light Field Networks (Sitzmann et al., NeurIPS 2021) is the single-evaluation, Plücker-coordinate version and notes the same issue.

### Exact intra-line architecture (no seam)

Use Plücker coordinates $(\mathbf d,\mathbf m=\mathbf o\times\mathbf d)$. An unoriented line is the equivalence $(\mathbf d,\mathbf m)\sim(-\mathbf d,-\mathbf m)$, and $t=\mathbf x\cdot\mathbf d$ flips sign with it.

- Don't canonicalize by hemisphere. Any hemisphere boundary is a great circle, which crosses the horizontal plane, and horizontal is where most LiDAR rays are.
- Instead, feed in the invariant features $\mathbf d\mathbf d^\top,\ \mathbf m\mathbf m^\top,\ \mathbf d\mathbf m^\top$. These are continuous and determine the line uniquely.
- Predict invariant weights $w_k\ge0$, widths $\tau_k>0$ and **3D anchors $\mathbf c_k$**, then set $\mu_k=\mathbf c_k\cdot\mathbf d$, which flips sign with the orientation.

$$\Phi(\ell,t)=\sum_{k=1}^K w_k(\ell)\,\sigma\!\Big(\tfrac{t-\mu_k(\ell)}{\tau_k(\ell)}\Big),\qquad A(\mathbf a,\mathbf b)=|\Phi(t_b)-\Phi(t_a)|$$

Because $\sigma(-z)=1-\sigma(z)$, $A$ is exactly invariant under reversal. The rational-quadratic monotone splines of Durkan et al. (2019) are a more flexible alternative to the logistic mixture.

| Property | This architecture | Your $\lvert\mathbf b-\mathbf a\rvert\,\mathrm{softplus}(f)$ with losses |
|---|---|---|
| $A\ge0$ | exact | exact |
| $A(\mathbf a,\mathbf a)=0$ | exact | exact |
| Monotone along a ray | exact | approximate |
| Segment additivity | exact | approximate |
| Reversal symmetry | exact | approximate |
| Multimodal / partial transparency | yes (up to $K$ modes per line) | yes |
| Network evaluations per ray | **1** (all $s$ queries closed-form) | one per query |
| **Cross-ray consistency** | **not guaranteed** | **not guaranteed** |

---

## 2. Cross-ray consistency: the real constraint

**Prop. 2 (consistency means an X-ray transform).** Under Prop. 1, $\hat\lambda(\mathbf x,\mathbf d)$ is independent of $\mathbf d$ for all $\mathbf x,\mathbf d$ **if and only if** $A=\mathcal X\lambda$, the segment X-ray transform of a 3D field $\lambda$. Equivalently, the full-line integrals $u(\ell)$ satisfy **John's ultrahyperbolic equation** (F. John, 1938). Parameterize lines as $\{(x_1+ty_1,\,x_2+ty_2,\,t)\}$; then

$$u_{x_1y_2}-u_{x_2y_1}=0.$$

So "exact additivity, exact consistency, and no integration" is only possible when $\lambda$ comes from a class whose X-ray transform is closed-form. There is no general escape.

**Why this matters more for LiDAR than it looks.** Lines in $\mathbb R^3$ form a 4D manifold. Each scan, with motion ignored, is the 2D bundle of lines through one origin. Training data is therefore a union of 2D sheets in 4D line space, and a novel pose is a different sheet. A ray-space network's prediction there depends entirely on its inductive bias. The 3D-field assumption is exactly the bias that fills in the missing dimensions, and it's what makes novel-pose prediction and registration meaningful.

### No-go for scalar potentials

Suppose $A(\mathbf a,\mathbf b)=F(\mathbf b)-F(\mathbf a)$ for arbitrary $\mathbf a,\mathbf b$. That is path-independent and antisymmetric, so nonnegativity plus symmetry force $A\equiv0$. Taking $|F(\mathbf b)-F(\mathbf a)|$ instead gives $\hat\lambda=|\nabla F\cdot\mathbf d|$, which vanishes in directions orthogonal to $\nabla F$: a sheet-like anisotropic medium, not an isotropic one. An arc-length integral of a scalar field is path-dependent, so no 3D potential exists.

### The tension: additivity vs. nonnegativity

- Additivity needs the integrand to be a **linear** functional of the parameters. For example, a Fourier field with integrated (sinc) encodings, as in mip-NeRF's IPE, plus a *linear* decoder is exactly additive.
- Nonnegativity normally comes from a **nonlinearity** (softplus), and that breaks the closed form.

The way out is a **nonnegative basis with nonnegative coefficients and closed-form line integrals**:

| Basis | Line integral | Cost per ray |
|---|---|---|
| Anisotropic Gaussians | erf (closed form) | $O(K_\text{ray})$ |
| Kaiser–Bessel "blobs" (Lewitt, 1990 CT literature) | closed form, compact support | $O(K_\text{ray})$, easy to cull |
| Trilinear / B-spline grids with nonneg vertex values | piecewise cubic, exact per cell | $O(\text{cells crossed})$ (DDA) |
| $\lambda=\lvert\sum_k c_k e^{i\mathbf k\cdot\mathbf x}\rvert^2$ | closed form | $O(K^2)$ |
| ReLU MLP | exact between breakpoints | $O(\text{linear regions crossed})$, impractical |

A multiresolution hash grid with a linear nonnegative readout also integrates exactly, but the cost scales with the number of finest-level cells crossed. That's no better than quadrature.

### Multi-origin derivative consistency (a soft regularizer for ray-space models)

Consider a point $\mathbf p$ observed from two origins $\mathbf o_1,\mathbf o_2$.

**Reversal symmetry does not couple the rays.** $A(\mathbf o_1,\mathbf p)=A(\mathbf p,\mathbf o_1)$ and $A(\mathbf o_2,\mathbf p)=A(\mathbf p,\mathbf o_2)$ are each intra-line identities. They contain no cross-line information, and the Plücker/mixture architecture already satisfies them exactly.

**The longitudinal derivative does.** For a consistent field, $A(\mathbf o,\mathbf p)=L\int_0^1\lambda(\mathbf o+t(\mathbf p-\mathbf o))\,dt$ with $L=|\mathbf p-\mathbf o|$ and $\hat{\mathbf d}=(\mathbf p-\mathbf o)/L$. Then

$$\nabla_{\mathbf p}A(\mathbf o,\mathbf p)=\hat{\mathbf d}\!\int_0^1\!\lambda\,dt+L\!\int_0^1\! t\,\nabla\lambda\,dt,$$

and integrating by parts gives

$$\hat{\mathbf d}\cdot\nabla_{\mathbf p}A(\mathbf o,\mathbf p)=\lambda(\mathbf p).$$

The longitudinal component is local: it is independent of the origin and of the path. The transverse component is a path-weighted integral of $\nabla\lambda$, so it is not a local constraint. Consistency at $\mathbf p$ therefore requires

$$\hat\lambda(\mathbf p,\hat{\mathbf d}_1)=\hat{\mathbf d}_1\cdot\nabla_{\mathbf p}A(\mathbf o_1,\mathbf p)\;\overset{!}{=}\;\hat{\mathbf d}_2\cdot\nabla_{\mathbf p}A(\mathbf o_2,\mathbf p)=\hat\lambda(\mathbf p,\hat{\mathbf d}_2).$$

This is $\mathcal L_{\text{iso}}$ restricted to directions that come from real sensor origins. With the mixture architecture, $\hat\lambda=\sum_k w_k\,\sigma'(\cdot)/\tau_k$ is closed-form, so no autodiff is needed and each origin costs one network evaluation.

**Loss:**

$$\mathcal L_{\text{pair}}=\mathbb E_{\mathbf x}\,\operatorname{Var}_{j\in\mathcal O(\mathbf x)}\big[\,g(\hat\lambda(\mathbf x,\hat{\mathbf d}_j))\big],\qquad g(\lambda)=1-e^{-\lambda\delta}$$

- **Points $\mathbf x$:** sample along real training rays, in both free space and near returns. Free space is where "ghost" hazard on one ray contradicts a clear view on another.
- **Origins $\mathcal O(\mathbf x)$:** any training sensor origins. The line $\mathbf o_j\to\mathbf x$ need not be a measured ray, because the model can be evaluated on any line.
- **Occlusion-agnostic:** $\hat\lambda(\mathbf x,\cdot)$ is local, so the constraint holds even when $\mathbf x$ is hidden from $\mathbf o_j$. This carries information from scans that see $\mathbf x$ into scans that don't. Equating cumulative $A$ would wrongly require equal occlusion.
- **Bounded comparison:** $g$ compares local opacity over a length $\delta$ rather than raw $\lambda$, so hard surfaces with very large $\lambda$ don't dominate.
- **Cost:** about $|\mathcal O|$ extra cheap network evaluations per regularization point.

**Limits:**

1. It only enforces consistency among the training directions at each point. Directions from novel poses are still unconstrained, so mix in random $\hat{\mathbf d}\sim S^2$ (the E4 trade-off).
2. Exact equality forbids real anisotropy, such as grazing-angle effects or foliage via the G-function. Either allow a low-order even-SH deviation, or weight the penalty so near-parallel directions count more than very different ones.
3. It is a soft constraint that makes a ray-space model approximately a 3D field. The Gaussian field (§3) satisfies it exactly. Evaluate it as rung 2.3 of the model ladder (E2).

---

## 3. Recommended core: an additive Gaussian hazard field

Take $\lambda(\mathbf x)=\sum_k w_k\exp\!\big(-\tfrac12(\mathbf x-\boldsymbol\mu_k)^\top P_k(\mathbf x-\boldsymbol\mu_k)\big)$. Along $\mathbf x=\mathbf o+s\mathbf d$, with $\mathbf r=\mathbf o-\boldsymbol\mu_k$:

$$a=\mathbf d^\top P\mathbf d,\quad b=\mathbf d^\top P\mathbf r,\quad c=\mathbf r^\top P\mathbf r,\quad s^*=-b/a$$

$$\int_{s_1}^{s_2}\lambda_k\,ds=w_k\,e^{-\frac12(c-b^2/a)}\sqrt{\tfrac{\pi}{2a}}\Big[\operatorname{erf}\!\big(\sqrt{a/2}\,(s_2-s^*)\big)-\operatorname{erf}\!\big(\sqrt{a/2}\,(s_1-s^*)\big)\Big]$$

What this gives you:

- **Everything is rigorous:** nonnegativity, monotonicity, additivity, symmetry and consistency.
- **Hazards commute.** Unlike 3DGS alpha blending, there is no depth sort, no popping and no flat-splat approximation.
- **NLL is closed-form too:** $p(d)=\lambda(\mathbf x_d)\,e^{-A(\mathbf o,\mathbf x_d)}$.
- **Anisotropy is free.** $\mathbf d$ is constant along a ray, so per-Gaussian even spherical harmonics $w_k(\mathbf d)$ keep the closed form. This is physically motivated for foliage: canopy extinction is $u_L(\mathbf x)\,G(\mathbf x,\mathbf d)$, where $G$ is the Ross–Nilson G-function from the leaf-angle distribution.
- **Hard surfaces** become thin, high-weight anisotropic Gaussians. Their crossing hazard scales with $1/|\mathbf d\cdot\mathbf n|$, which is the correct geometric behaviour.

```python
import math, torch

def gaussian_cum_hazard(o, d, s, mu, M, logw):
    """o,d:[R,3]  s:[R,S] query distances  mu:[K,3]
    M:[K,3,3] with precision P = M^T M   logw:[K]
    Returns A(o, o+s*d): [R,S]. Brute force over K; use BVH/binning in practice."""
    r  = o[:, None] - mu[None]                         # [R,K,3]
    Md = torch.einsum('kij,rj->rki', M, d)             # M d
    Mr = torch.einsum('kij,rkj->rki', M, r)            # M r
    a  = (Md * Md).sum(-1).clamp_min(1e-12)            # [R,K]
    b  = (Md * Mr).sum(-1)
    sstar = -b / a
    perp2 = ((Mr + sstar[..., None] * Md) ** 2).sum(-1)   # c - b^2/a, stable form
    amp = torch.exp(logw - 0.5 * perp2) * torch.sqrt(math.pi / (2 * a))
    k = torch.sqrt(a / 2)[:, None]                     # [R,1,K]
    e = torch.erf(k * (s[..., None] - sstar[:, None])) - torch.erf(k * (-sstar[:, None]))
    return (amp[:, None] * e).sum(-1)                  # [R,S]
```

On the loss: with $s\sim\mathcal U[0,s_{\max}]$, your CDF loss is the **CRPS**, a strictly proper scoring rule, so it's sound for multimodal targets. Because queries are closed-form you can afford many $s$ per ray. If you importance-sample near $d$, reweight to keep it proper. Rays with no return contribute $e^{-A(\mathbf o,\mathbf o+s_{\max}\mathbf d)}$ as a censored term.

---

## 4. Answers to your research questions

1. **Exact additivity without integration?** Within each line, yes, via the per-line potential in §1. Across lines it's only possible by being a line integral, and then "no integration" means "closed-form integration over a restricted basis."
2. **Can everything be architectural?** Nonnegativity, monotonicity, zero length, additivity and symmetry can all be guaranteed, and spatial consistency can too, but only through 3D-field structure.
3. **Does a shared hash-grid encoder enforce consistency?** No, it only improves interpolation. With $A=\text{MLP}(h(\mathbf a),h(\mathbf b),\ldots)$, the decoder can still combine features arbitrarily. Consistency comes from a readout that is a linear line functional, not from where the features live.
4. **Approximate with few samples?**
   - Additivity: you need zero samples, because it's architectural.
   - Consistency: the right regularizer is $\mathcal L_{\text{iso}}=\mathbb E_{\mathbf x}\operatorname{Var}_{\mathbf d\sim S^2}[\hat\lambda(\mathbf x,\mathbf d)]$. It's necessary and sufficient by Prop. 2, and $\hat\lambda$ is closed-form here. But each sampled direction is a new line and a new network evaluation, and coverage has to span 4D line space, especially directions you never observed. I expect this cost to eat most of the speed advantage. Measure it (E4 below).
5. **Relevant frameworks:**
   - AutoInt / neural antiderivatives (with the caveat in §1)
   - Light Field Networks and Plücker ray-space models
   - X-ray-transform range conditions (John; Helgason–Ludwig in 2D)
   - Blob-basis tomography (Lewitt)
   - Integrated positional encoding (mip-NeRF)
   - Monotone networks (UMNN, though it reintroduces 1D quadrature; constrained monotone MLPs; monotone splines)
   - Canopy radiative transfer, for principled anisotropy
6. **Measuring consistency:** see E3 below.
7. **Compared with Gaussians:**

| Method | Work per ray | Consistency | Expected novel-pose generalization |
|---|---|---|---|
| Nerfacto (current) | ~$10^2$ hash lookups + tiny MLPs (proposal + main samples) | exact (field) | good |
| Ray-space $\Phi(\ell,t)$ | 1 MLP on 4D line features, then closed-form queries | none / regularized | weak; ray space has 3D occlusion discontinuities inside a 4D domain |
| Gaussian hazard field | $K_\text{ray}$ × ~40 FLOPs + erf, no MLP | exact | good; explicit parameters, fast Adam convergence |

---

## 5. Proposed experiments

- **E0. Profile Nerfacto first.** Check what share of the 2–5 min goes to data loading, proposal sampling, hash lookups and the iteration count needed to converge. Cheap wins: occupancy-grid empty-space skipping, and putting proposal samples near the known range $d$.
- **E1. Synthetic tomography benchmark** with known $\lambda$: thin surfaces, semi-transparent volumes, foliage with G-function anisotropy, and censored no-return rays. Train on $N$ origins and test on held-out origins at increasing distance.
- **E2. Model ladder** (on E1, then real data):
  1. Your unconstrained $A_\theta$ with the three losses
  2. Exact $\Phi(\ell,t)$
  3. Exact $\Phi(\ell,t)$ plus $\mathcal L_{\text{iso}}$
  4. Isotropic Gaussian hazard field
  5. Gaussian field with even-SH anisotropy
  6. Nerfacto

  Report **time-to-accuracy** curves on held-out-pose CRPS/NLL, not time per iteration.
- **E3. Consistency metrics:**
  - Additivity and reversal residuals (relevant for 2.1 only)
  - $\operatorname{CV}_{\mathbf d}[\hat\lambda(\mathbf x,\mathbf d)]$ at sampled points
  - **Tomographic projection residual:** fit a nonnegative voxel $\lambda^*$ by NNLS to the model's predicted $A$ on random lines, and report $\|A_\theta-\mathcal X\lambda^*\|/\|A_\theta\|$. This measures distance from the consistent set.
  - Cross-trajectory tests: train on one street, test on the perpendicular street or a loop closure.
- **E4. Regularization budget:** sweep $(N_{\mathbf x},N_{\mathbf d})$ for $\mathcal L_{\text{iso}}$ against both consistency and wall-clock time. This directly answers whether a few samples are enough.
- **E5. Downstream:** registration accuracy from novel poses using each model's likelihood.

My hypothesis is that 2.1 and 2.2 match the baseline on training poses but degrade on novel poses, 2.3 closes part of that gap at a large regularization cost, and 2.4/2.5 dominate on both accuracy and time. That's a prediction to test, not a result.

I can prototype the Gaussian hazard field (with culling) and the E1 synthetic benchmark in the LiZARD workspace if you'd like. I'd need read access to the repo to match your existing data format.