# TensorForge 📐

> An interactive, zero-dependency geometric linear algebra playground. Built for First-Year B.Tech AI & Machine Learning coursework and designed as an embeddable component for the **Lectern** study platform.

![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)
![Zero Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)
![Pure Vanilla JS](https://img.shields.io/badge/stack-HTML5%20%7C%20CSS%20%7C%20JS-orange)
![60 FPS Canvas](https://img.shields.io/badge/graphics-HTML5%20Canvas-indigo)

---

## ✨ Overview

Linear Algebra is the foundational math of Artificial Intelligence and Machine Learning—from coordinate spaces and matrix transformations to PCA, SVD, and latent embeddings. Yet textbooks often present these concepts as dry algebraic manipulations rather than dynamic spatial operations.

**TensorForge** gives you direct tactile intuition for 2D linear algebra:
1. **Grab and Drag Basis Vectors**: Directly manipulate $\hat{i}$ and $\hat{j}$ on the Cartesian coordinate plane to see space shear, rotate, scale, or collapse.
2. **Watch the Determinant Morph**: The shaded unit square parallelogram dynamically computes $\det(A) = ad - bc$, color-coding area expansion, orientation inversions ($\det < 0$), and dimensional collapse ($\det = 0$).
3. **Visualize Invariant Eigenvectors**: Real-time analytical solver of the characteristic equation $\det(A - \lambda I) = 0$, tracing the invariant span lines where vectors only stretch and never rotate.
4. **Step Through Matrix Multiplication**: Scrub through the composition of transformations to see visually why matrix multiplication is non-commutative ($AB \neq BA$).
5. **Vector Sandbox**: Explore vector addition parallelograms, dot products, and orthogonal projections in real time.

---

## 🎯 Academic Alignment (FY Sem 1 AI & ML)

| Syllabus Concept | How TensorForge Visualizes It |
|---|---|
| **Linear Transformations** | Grid warping showing how basis vectors $\hat{i}$ and $\hat{j}$ determine where every vector in $\mathbb{R}^2$ lands. |
| **Determinants ($\det A$)** | Area scaling factor of the unit square. Signed area represents orientation preservation vs reflection. |
| **Matrix Rank & Singularity** | Rank 2 = 2D plane preserved; Rank 1 = Space collapses into a 1D line ($\det = 0$); Rank 0 = Origin collapse. |
| **Eigenvalues & Eigenvectors** | Invariant direction lines $\text{Span}(\vec{v})$ where $A\vec{v} = \lambda \vec{v}$. Stretches along the line by $\lambda$. |
| **Matrix Multiplication** | Consecutive application of transformations $T(x) = A(Bx)$ with step-by-step interpolation slider. |
| **Dot Product & Projections** | Real-time orthogonal decomposition $\text{proj}_{\vec{v}}(\vec{u}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{v}\|^2}\vec{v}$. |

---

## 🚀 Quick Start (Zero Build, Zero Setup)

TensorForge has **zero npm dependencies** and requires no compilation.

1. Clone or download the folder:
   ```bash
   git clone https://github.com/screen-watcher-elite/tensorforge.git
   ```
2. Double-click `index.html` to open it in any modern browser, or serve with any static server:
   ```bash
   npx serve .
   ```

---

## 🧩 Integration with Lectern (Study Tool)

TensorForge was architected to be completely decoupled and zero-cost, making it effortless to embed into closed-source student tools like **Lectern**:

### Option 1: Direct `<iframe>` Embed
```html
<iframe 
  src="/path/to/tensorforge/index.html" 
  width="100%" 
  height="650px" 
  style="border: none; border-radius: 12px;">
</iframe>
```

### Option 2: Programmatic JavaScript API
The global `TensorForge` object is exposed on `window`:
```javascript
// Programmatically set matrix values
window.TensorForge.setMatrix(2.0, 1.0, 0.0, 2.0);

// Get current matrix
const currentM = window.TensorForge.getMatrix();
console.log(currentM.determinant(), currentM.trace());

// Switch operational mode ('transform' | 'eigen' | 'mult' | 'vectors')
window.TensorForge.setMode('eigen');
```

---

## 📦 File Architecture

```
tensorforge/
├── index.html       # Semantic HTML5 layout with sidebar controls & canvas viewport
├── style.css        # Dark Glassmorphism 2.0 design tokens & responsive grid
├── engine.js        # Pure JS Vector2D, Matrix2x2, EigenSolver, and AnimationTimer
├── app.js           # 60 FPS Canvas coordinate engine, interaction & UI binding
├── README.md        # Syllabus guide, math explanations, integration docs
└── LICENSE          # Apache 2.0 License
```

---

## 📜 License

Distributed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for details.
