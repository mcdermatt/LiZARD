# References

Reading list for LiZARD, grouped by how each work relates to the **Self-Conditioned Knot CDF (SCK)** direction in [README.md](README.md) §0. arXiv links were checked against the arXiv API on 2026-10-26.

**Status:** none of the works below combines (1) self-conditioned knot placement, (2) probabilistic CDF return targets and (3) LiDAR. Each covers one or two of these pieces.

Legend: 🎯 = closest to SCK / likely to come up in review.

---

## 1. Our prior work

| Work | Link | Relevance |
|---|---|---|
| McDermott & Rife, **A Probabilistic Formulation of LiDAR Mapping with Neural Radiance Fields** (PLiNK) | [IEEE](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10947591) · [arXiv:2411.01725](https://arxiv.org/abs/2411.01725) · [code](https://github.com/mcdermatt/PLINK) | Baseline. Probabilistic return CDF along each ray with step-function targets; dense coarse-to-fine sampling. SCK aims to keep the formulation and remove the dense sampling. |

## 2. Probabilistic / multi-return LiDAR neural fields

All of these still use volume rendering with dense samples along each ray.

| Work | Link | Relevance |
|---|---|---|
| Huang et al., **Neural LiDAR Fields for Novel View Synthesis** (NFL), ICCV 2023 | [arXiv:2305.01643](https://arxiv.org/abs/2305.01643) | Beam divergence, secondary returns, ray drop. Multi-return via peak detection on volume-rendering weights. |
| Tao et al., **LiDAR-NeRF: Novel LiDAR View Synthesis via Neural Radiance Fields** | [arXiv:2304.10406](https://arxiv.org/abs/2304.10406) | Depth, intensity and ray-drop heads for LiDAR NVS. |
| Zheng et al., **LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis**, CVPR 2024 | [arXiv:2404.02742](https://arxiv.org/abs/2404.02742) | Dynamic scenes; hybrid 4D representation. |

## 3. Single-evaluation ray-space models (one network call per ray)

| Work | Link | Relevance |
|---|---|---|
| 🎯 Liu et al., **RayDF: Neural Ray-surface Distance Fields with Multi-view Consistency**, NeurIPS 2023 | [arXiv:2310.19629](https://arxiv.org/abs/2310.19629) | Closest on *consistency*. Ray → surface distance, plus a dual-ray visibility classifier enforcing multi-view consistency (a learned analogue of local batching / `L_pair`). Deterministic, single surface, not LiDAR-probabilistic. |
| Feng et al., **PRIF: Primary Ray-based Implicit Function**, ECCV 2022 | [arXiv:2208.06143](https://arxiv.org/abs/2208.06143) | Ray → hit point directly, no marching. Deterministic, single surface. |
| Sitzmann et al., **Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering**, NeurIPS 2021 | [arXiv:2106.02634](https://arxiv.org/abs/2106.02634) | Plücker-coordinate ray-space extreme; notes the lack of multi-view consistency. The "pure ray network" baseline (a). |
| Lindell, Martel & Wetzstein, **AutoInt: Automatic Integration for Fast Neural Volume Rendering**, CVPR 2021 | [arXiv:2012.01714](https://arxiv.org/abs/2012.01714) | Ray-conditioned antiderivative network: the per-line potential $\Phi(\ell,t)$ of README §1. Same lack of cross-ray consistency. |

## 4. Learned / iterative sample placement (RGB NeRF)

| Work | Link | Relevance |
|---|---|---|
| 🎯 Attal et al., **HyperReel: High-Fidelity 6-DoF Video with Ray-Conditioned Sampling**, CVPR 2023 | [arXiv:2301.02238](https://arxiv.org/abs/2301.02238) | Closest on *architecture*. Sample-prediction network outputs per-ray sample points, then queries a volume: the "where (ray) / what (field)" split. |
| 🎯 Neff et al., **DONeRF: Towards Real-Time Rendering of Compact Neural Radiance Fields using Depth Oracle Networks**, EGSR 2021 | [arXiv:2103.03231](https://arxiv.org/abs/2103.03231) | Depth oracle predicts a per-ray depth distribution, then few samples. Single pass, not iterative. |
| Kurz et al., **AdaNeRF: Adaptive Sampling for Real-time Rendering of Neural Radiance Fields**, ECCV 2022 | [arXiv:2207.10312](https://arxiv.org/abs/2207.10312) | Learned sampling network with adaptive sample counts. |
| Fang et al., **NeuSample: Neural Sample Field for Efficient View Synthesis** | [arXiv:2111.15552](https://arxiv.org/abs/2111.15552) | Network maps a ray to its sample locations. |
| Barron et al., **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields**, CVPR 2022 | [arXiv:2111.12077](https://arxiv.org/abs/2111.12077) | Proposal networks distilled online: iterative coarse-to-fine histograms. |
| Barron et al., **Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields**, ICCV 2023 | [arXiv:2304.06706](https://arxiv.org/abs/2304.06706) | Proposal sampling with hash grids; state of the art for sampling efficiency. |

## 5. Mechanisms used by SCK

| Work | Link | Used for |
|---|---|---|
| Teed & Deng, **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow**, ECCV 2020 | [arXiv:2003.12039](https://arxiv.org/abs/2003.12039) | Recurrent shared update operator with feature lookups at current estimates; stop-gradient between iterations. |
| Chen, Zhang & Hinton, **Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning**, ICLR 2023 | [arXiv:2208.04202](https://arxiv.org/abs/2208.04202) | Origin of the "self-conditioning" term (feeding previous outputs back as inputs). |
| Durkan et al., **Neural Spline Flows**, NeurIPS 2019 | [arXiv:1906.04032](https://arxiv.org/abs/1906.04032) | Monotone rational-quadratic splines via cumulative softplus/softmax knots. |
| Wehenkel & Louppe, **Unconstrained Monotonic Neural Networks**, NeurIPS 2019 | [arXiv:1908.05164](https://arxiv.org/abs/1908.05164) | Monotone networks via integrating a positive network (reintroduces 1D quadrature). |
| Müller et al., **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**, SIGGRAPH 2022 | [arXiv:2201.05989](https://arxiv.org/abs/2201.05989) | Hash encoding $h(\mathbf x_k)$ at knot locations; the shared spatial memory that gives consistency. |

## 6. Baselines and general background

| Work | Link | Relevance |
|---|---|---|
| Mildenhall et al., **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**, ECCV 2020 | [arXiv:2003.08934](https://arxiv.org/abs/2003.08934) | Consistency through a shared field with no direct ray-to-ray comparison: the mechanism SCK relies on. |
| Tancik et al., **Nerfstudio: A Modular Framework for Neural Radiance Field Development** (Nerfacto), SIGGRAPH 2023 | [arXiv:2302.04264](https://arxiv.org/abs/2302.04264) | Nerfacto baseline; proposal-sampler cost to compare against. |
| Barron et al., **Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields**, ICCV 2021 | [arXiv:2103.13415](https://arxiv.org/abs/2103.13415) | Integrated positional encoding: an example of exact line integrals with a linear decoder (README §2). |
| Kerbl et al., **3D Gaussian Splatting for Real-Time Radiance Field Rendering**, SIGGRAPH 2023 | [arXiv:2308.04079](https://arxiv.org/abs/2308.04079) | Contrast for the additive Gaussian hazard field (README §3): alpha blending vs. exact hazard sums. |

## 7. Tomography and theory (README §1–2)

| Work | Link | Relevance |
|---|---|---|
| X-ray transform | [Wikipedia](https://en.wikipedia.org/wiki/X-ray_transform) | Consistency across rays ⇔ being a line integral of a 3D field (Prop. 2). |
| Radon transform | [Wikipedia](https://en.wikipedia.org/wiki/Radon_transform) | Background for the above. |
| F. John, *The ultrahyperbolic differential equation with four independent variables*, Duke Math. J., 1938 | — | Range condition on 4D line integrals: $u_{x_1y_2}-u_{x_2y_1}=0$. |
| R. M. Lewitt, *Multidimensional digital image representations using generalized Kaiser–Bessel window functions*, JOSA A, 1990 | — | "Blob" basis with closed-form line integrals. |
| Ross–Nilson G-function (canopy radiative transfer; e.g. J. Ross, *The Radiation Regime and Architecture of Plant Stands*, 1981) | — | Physically motivated anisotropy for foliage. |

---

## To check

- [ ] Google Scholar: papers citing RayDF, NFL and PLiNK, 2025–26.
- [ ] arXiv: `LiDAR ("ray-based" OR "single forward") (distribution OR "multiple returns")`.
- [ ] Any LiDAR work using sample-prediction networks (HyperReel / DONeRF-style) for range distributions.
