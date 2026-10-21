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

### Parameterization matters: Plücker vs. endpoint models

The claim that "lines through the same point from different origins are far apart in input space" holds for the Plücker model $\Phi(\ell,t)$, not for an endpoint model $A_\theta(\mathbf a,\mathbf b)$. In the endpoint model, rays from $\mathbf o_1$ and $\mathbf o_2$ to $\mathbf p$ share half their input ($\mathbf a=\mathbf p$), and a shared 3D hash encoding $h(\mathbf p)$ gives the network a natural place to store what is at $\mathbf p$.

**Reversal symmetry still does not force derivative agreement.** Any even, direction-dependent field $\lambda(\mathbf x,\mathbf d)=\lambda(\mathbf x,-\mathbf d)$ yields $A(\mathbf a,\mathbf b)=\int_{\mathbf a}^{\mathbf b}\lambda(\mathbf x,\hat{\mathbf d})\,d\ell$, which is exactly symmetric, additive, nonnegative and zero on zero-length segments, yet gives arbitrarily different hazards at $\mathbf p$ from different origins. Symmetry only ties $\lambda(\mathbf p,\mathbf d)$ to $\lambda(\mathbf p,-\mathbf d)$, i.e. origins collinear with $\mathbf p$ on opposite sides.

In endpoint form, a consistent field satisfies

$$\hat{\mathbf u}_j\cdot\nabla_{\mathbf a}A(\mathbf p,\mathbf o_j)=-\lambda(\mathbf p),\qquad \hat{\mathbf u}_j=\frac{\mathbf o_j-\mathbf p}{|\mathbf o_j-\mathbf p|}\quad\text{for every } j.$$

The longitudinal component must agree across origins; the transverse component legitimately differs because it depends on the whole path. Nothing in the network makes the longitudinal projection constant in $\mathbf b$, so it must be imposed by a loss.

| Model | Intra-line properties (additivity, monotonicity) | Cross-ray prior |
|---|---|---|
| Plücker $\Phi(\ell,t)$ | exact | poor (no shared point in the input) |
| Endpoint $A_\theta(\mathbf a,\mathbf b)$ + shared hash $h(\cdot)$ | approximate | decent (shares $\mathbf p$ in the input) |
| 3D field (Nerfacto / Gaussians) | exact | exact |

The more a model shares features at $\mathbf x$, the closer it gets to a field, and the closer it gets to needing integration.

### Horizon Endpoint Model: Direct Continuous CDF via Control Points

To eliminate the need for querying $A_\theta(\mathbf a, \mathbf b)$ at many discrete distances $s$, the endpoint formulation can be extended to output the entire **continuous CDF curve** for a ray in a single forward pass.

Instead of outputting a scalar hazard at an arbitrary query point, fix the second endpoint at the sensor horizon / far bounding plane:
$$\mathbf a = \mathbf o, \qquad \mathbf b = \mathbf o + s_{\max}\mathbf d$$

The network takes the ray boundary features (e.g. hash encodings $h(\mathbf o)$ and $h(\mathbf b)$, or ray direction $\mathbf d$) and directly outputs the parameters of a continuous 1D monotonic function:
$$\text{MLP}\big(h(\mathbf o), h(\mathbf b), \mathbf d\big) \longrightarrow \boldsymbol{\theta}_{\text{CDF}} = \big\{(w_k, \tau_k, \mu_k)\big\}_{k=1}^K \quad \text{or spline knots } \big\{(s_k, C_k, \delta_k)\big\}$$

Once parameterized, the return CDF $C(s)$ (or cumulative hazard $A(s) = -\ln(1 - C(s))$) is continuous and closed-form across $s \in [0, s_{\max}]$:
$$C(s) = \sum_{k=1}^K w_k \, \sigma\!\left(\frac{s - \mu_k}{\tau_k}\right) \quad \text{or a monotonic rational-quadratic spline (Durkan et al., 2019)}$$

**Key advantages:**
1. **Single Evaluation at Render Time:** The MLP runs **once per ray**. Any subsequent query along the ray (e.g. survival probability at distance $s$, expected range $\mathbb E[s]$, median depth, or multi-echo returns) is evaluated in closed form or via cheap 1D root-finding without further neural network calls.
2. **Continuous Loss Computation:** The measurement loss $\int_0^{s_{\max}} (C(s) - H_d(s))^2 ds$ can be evaluated analytically or via standard low-order Gauss–Legendre quadrature in a few scalar operations per ray.
3. **Guaranteed Monotonicity:** Enforcing positive spline derivatives or mixture weights $w_k \ge 0$ guarantees strictly non-decreasing CDFs $C(s)$ along each ray by design.

**Avoiding Acausal Hazards (The Horizon Standardization):**
If the second point $\mathbf b$ is allowed to vary freely along the ray ($s_1 < s_2$), the prediction on $[0, s_1]$ under $(\mathbf o, \mathbf b_2)$ must match that under $(\mathbf o, \mathbf b_1)$, but the network receives distinct features $h(\mathbf b_1) \neq h(\mathbf b_2)$. An obstacle at $s_2$ could retroactively change the predicted CDF at $s_1$ (acausal hazard). Standardizing $\mathbf b$ to the maximum ray horizon $\mathbf o + s_{\max}\mathbf d$ resolves this by defining a unique, globally consistent domain for the 1D parameterization.

### Anchor-field consistency (consensus regularizer)

**Why a cross-projection trick can't work.** At $\mathbf p$, $\mathbf g_j=\nabla_{\mathbf a}A(\mathbf p,\mathbf o_j)=-\lambda(\mathbf p)\,\hat{\mathbf u}_j+\mathbf t_j$ with $\mathbf t_j\perp\hat{\mathbf u}_j$. Projecting one ray's gradient onto another's direction gives $\mathbf g_1\cdot\hat{\mathbf u}_2=-\lambda\,(\hat{\mathbf u}_1\cdot\hat{\mathbf u}_2)+\mathbf t_1\cdot\hat{\mathbf u}_2$, which is scaled by a cosine and contaminated by the non-local transverse term. More fundamentally, for any fixed vector $\mathbf v$, $\mathbf v\cdot\hat{\mathbf u}$ is linear and odd in $\hat{\mathbf u}$ and cannot be constant over directions. The shared quantity at $\mathbf p$ must be a **scalar**. Equivalently, $\hat\lambda(\mathbf p,\hat{\mathbf u})$ is even on the sphere and expands in even spherical harmonics ($l=0,2,4,\dots$); consistency means every term except $l=0$ vanishes.

**The anchor.** Introduce a small auxiliary 3D field $\mu_\phi(\mathbf x)\ge0$ (e.g. a cheap hash grid) and tie every ray's implied hazard to it:

$$\mathcal L_{\text{anchor}}=\mathbb E_{\mathbf x,\hat{\mathbf d}}\Big[\big(g(\hat\lambda_\theta(\mathbf x,\hat{\mathbf d}))-g(\mu_\phi(\mathbf x))\big)^2\Big],\qquad g(\lambda)=1-e^{-\lambda\delta}$$

$\mu_\phi$ is **only evaluated pointwise and never integrated**. It acts as a shared scalar consensus that every viewing direction must agree with, in the spirit of consensus ADMM.

Compared with $\mathcal L_{\text{pair}}$:

- **One direction per point suffices.** Comparison is against the anchor, not other origins, so the cost is about one ray-network evaluation plus one grid lookup per sample instead of $|\mathcal O|$ evaluations.
- **Novel directions are cheap.** Random $\hat{\mathbf d}\sim S^2$ costs the same as observed directions, which helps novel-pose generalization.
- **Evidence accumulates.** $\mu_\phi$ pools what every scan says about $\mathbf x$, so a direction seen once is still pulled toward the consensus.
- **Anisotropy is a dial.** Let the anchor carry a few low-order even harmonics to allow physical anisotropy (foliage, grazing incidence).

At the optimum it is equivalent to the variance penalty, but cheaper and lower-variance to train.

**Why there is no exact version.** Exactness would require $\partial_s A(\mathbf o,\mathbf o+s\hat{\mathbf d})=\mu(\mathbf o+s\hat{\mathbf d})$ for all $s$ and all rays. The only solution of that ODE is $A=\int\mu$, which is integration again. The anchor keeps the 3D field as a cheap pointwise regularizer while the fast, exactly additive ray model does the prediction.

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
3. **Does a shared hash-grid encoder enforce consistency?** No, it only improves interpolation. With $A=\text{MLP}(h(\mathbf a),h(\mathbf b),\ldots)$, the decoder can still combine features arbitrarily. Consistency comes from a readout that is a linear line functional, not from where the features live. It does give an endpoint model a better cross-ray prior than Plücker inputs (see "Parameterization matters" in §2).
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
| Endpoint $A_\theta(\mathbf a,\mathbf b)$ + hash + $\mathcal L_{\text{anchor}}$ | 2 hash lookups + 1 MLP per query | approximate (anchor-regularized) | unknown; better prior than Plücker, to be measured |
| Horizon Endpoint CDF + hash + $\mathcal L_{\text{anchor}}$ | 2 hash lookups + 1 MLP per ray (closed-form continuous CDF) | approximate (anchor-regularized) | unknown; fast inference + 3D hash prior |
| Gaussian hazard field | $K_\text{ray}$ × ~40 FLOPs + erf, no MLP | exact | good; explicit parameters, fast Adam convergence |

---

## 5. Proposed experiments

- **E0. Profile Nerfacto first.** Check what share of the 2–5 min goes to data loading, proposal sampling, hash lookups and the iteration count needed to converge. Cheap wins: occupancy-grid empty-space skipping, and putting proposal samples near the known range $d$.
- **E1. Synthetic tomography benchmark** with known $\lambda$: thin surfaces, semi-transparent volumes, foliage with G-function anisotropy, and censored no-return rays. Train on $N$ origins and test on held-out origins at increasing distance.
- **E2. Model ladder** (on E1, then real data):
  1. Your unconstrained $A_\theta$ with the three losses
  2. Exact $\Phi(\ell,t)$
  3. Exact $\Phi(\ell,t)$ plus a consistency regularizer ($\mathcal L_{\text{iso}}$, $\mathcal L_{\text{pair}}$ or $\mathcal L_{\text{anchor}}$)
  4. Pointwise Endpoint $A_\theta(\mathbf a,\mathbf b)$ with a shared 3D hash encoding plus $\mathcal L_{\text{anchor}}$
  5. Continuous Horizon Endpoint CDF (spline/mixture control points) plus $\mathcal L_{\text{anchor}}$
  6. Isotropic Gaussian hazard field
  7. Gaussian field with even-SH anisotropy
  8. Nerfacto

  Report **time-to-accuracy** curves on held-out-pose CRPS/NLL, not time per iteration.
- **E3. Consistency metrics:**
  - Additivity and reversal residuals (relevant for 2.1 and 2.4)
  - $\operatorname{CV}_{\mathbf d}[\hat\lambda(\mathbf x,\mathbf d)]$ at sampled points
  - **Tomographic projection residual:** fit a nonnegative voxel $\lambda^*$ by NNLS to the model's predicted $A$ on random lines, and report $\|A_\theta-\mathcal X\lambda^*\|/\|A_\theta\|$. This measures distance from the consistent set.
  - Cross-trajectory tests: train on one street, test on the perpendicular street or a loop closure.
- **E4. Regularization budget:** sweep $(N_{\mathbf x},N_{\mathbf d})$ for $\mathcal L_{\text{iso}}$, $\mathcal L_{\text{pair}}$ and $\mathcal L_{\text{anchor}}$ against both consistency and wall-clock time. This directly answers whether a few samples are enough.
- **E5. Downstream:** registration accuracy from novel poses using each model's likelihood.

My hypothesis is that 2.1 and 2.2 match the baseline on training poses but degrade on novel poses, 2.3 and 2.4 close part of that gap (2.4 more cheaply, thanks to the shared encoding and anchor), and 2.5/2.6 dominate on both accuracy and time. That's a prediction to test, not a result.

I can prototype the Gaussian hazard field (with culling) and the E1 synthetic benchmark in the LiZARD workspace if you'd like. I'd need read access to the repo to match your existing data format.