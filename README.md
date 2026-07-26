# Generative Model Explainer Studio (Flow Matching vs Diffusion SDE)

An interactive, high-fidelity web visualizer demonstrating the mathematical principles and vector field dynamics of **Flow Matching (Rectified Flow)** vs **Diffusion SDE (Score-based SDE)**.

![Generative Visualizer](https://img.shields.io/badge/Generative_AI-Flow_Matching_vs_Diffusion-blueviolet?style=for-the-badge)

## 🌟 Key Features

1. **Velocity Vector Field Arrows ($\vec{v}_t$) Visualization**
   - **Flow Matching**: Straight Optimal Transport (OT) velocity field with 100% constant velocity vectors ($\vec{v} = x_1 - z_0$) and straight trajectories.
   - **Diffusion SDE**: Dynamic score/drift vector field ($\vec{v}_t = \nabla_x \log p_t(x_t)$) with real-time rotating/swinging arrows and step-coherent SDE stochastic drift.

2. **2D Probability Manifold & 20-Species Breed Clusters**
   - Renders overall data distribution $p(x)$ and prompt-conditioned class distributions $p(x | c)$ (Cat, Dog, Car).
   - 20 species per class, each assigned **permanent, dedicated, unique (x, y) manifold coordinates**.

3. **Multi-Breed Feature Blend & Interpolation**
   - Demonstrates how generative models blend features from neighboring manifold clusters (e.g. `v_삼색이 + 샴 냥이 + 러시안블루 (특징 융합)`).

4. **Real-Time Noise-to-Detail Reverse Denoising Preview**
   - At $t=0$, renders heavy Gaussian RGB noise grain.
   - As $t \to 1.0$, progressively sharpens blurry shapes into crisp final images.

5. **HD WebM Video Exporting**
   - Built-in canvas recorder to export high-definition WebM video clips.

---

## 🛠️ Project Structure

- `index.html`: Clean, modern responsive layout with Lucide icons.
- `styles.css`: Dark mode glassmorphism theme and rich UI tokens.
- `app.js`: Complete 2D manifold canvas renderer, vector field calculations, and WebM recorder.

---

## 🚀 How to Run Locally

```bash
# Simply serve using python or any local static web server
python -m http.server 8080
```
Open `http://localhost:8080` in your web browser.
