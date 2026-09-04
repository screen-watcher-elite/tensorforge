# TensorForge 📐

> An interactive, zero-dependency geometric linear algebra playground. Built for First-Year B.Tech AI & Machine Learning coursework and designed as an embeddable component for the **Lectern** study platform.

![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)
![Zero Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)
![Pure Vanilla JS](https://img.shields.io/badge/stack-HTML5%20%7C%20CSS%20%7C%20JS-orange)
![60 FPS Canvas](https://img.shields.io/badge/graphics-HTML5%20Canvas-indigo)
![Built with Claude Code](https://img.shields.io/badge/built%20with-Claude%20Code-d97706?logo=anthropic&logoColor=white)

---

## ✨ Overview

Linear Algebra is the foundational math of Artificial Intelligence and Machine Learning—from coordinate spaces and matrix transformations to PCA, SVD, loss landscapes, and neural backpropagation. Yet textbooks often present these concepts as dry algebraic manipulations rather than dynamic spatial operations.

**TensorForge** is the **ultimate zero-dependency geometric linear algebra and optimization powerhouse** featuring 9 integrated modules:

1. **🌀 2D Linear Transformations**: Directly manipulate basis vectors $\hat{i}$ and $\hat{j}$ on the Cartesian coordinate plane to see space shear, rotate, scale, or collapse.
2. **⚡ Invariant Eigensystem & Scanner**: Real-time analytical solver for $\det(A - \lambda I) = 0$, invariant span lines, and an interactive unit-vector probe to hunt for collinear directions.
3. **✖ Matrix Composition ($A \times B$)**: Scrub through the composition of transformations to see visually why matrix multiplication is non-commutative ($AB \neq BA$).
4. **🎯 Vector Sandbox**: Explore vector addition parallelograms, Gram-Schmidt orthogonalization ($\vec{v} \perp \vec{u}$), normalization, and live dot product projections.
5. **🧊 3D VectorSpace**: A full 3D transformation engine for $3 \times 3$ matrices! Interactive 3D unit cube, $\hat{i}, \hat{j}, \hat{k}$ basis vectors, mouse orbit camera, and 3D volume determinant $\det(A_{3\times3})$.
6. **📉 LossLab (Optimization Engine)**: Race **SGD**, **Momentum**, **RMSprop**, and **Adam** particles across 2D loss surfaces (Convex Bowl, Saddle Point, Rosenbrock's Banana Valley) with real-time loss tracking.
7. **🕸️ MicroGraph (Autograd & Backprop)**: Interactive Directed Acyclic Graph (DAG) visualizer stepping through forward activations and reverse chain-rule gradient accumulation.
8. **📚 Theory Vault (Simplified Notes)**: Student-first academic lecture notes covering FY Sem 1 Linear Algebra & ML math with real-world AI applications and viva examination tips.
9. **🎓 Viva Exam Quiz**: 10 active-recall multiple-choice viva prep questions with instant feedback, scoring, and academic explanations.

---

## 🎯 Academic Alignment (FY Sem 1 AI & ML)

| Syllabus Concept | How TensorForge Visualizes It |
|---|---|
| **Linear Transformations** | Grid warping showing how basis vectors $\hat{i}$ and $\hat{j}$ determine where every vector in $\mathbb{R}^2$ lands. |
| **Determinants ($\det A$)** | Area/Volume scaling factor. Signed area represents orientation preservation vs reflection. |
| **Matrix Rank & Singularity** | Rank 2 = 2D plane preserved; Rank 1 = Space collapses into a 1D line ($\det = 0$); Rank 0 = Origin collapse. |
| **Eigenvalues & Invariant Lines** | Exact quadratic characteristic equation breakdown with live substituted values and invariant span lines. |
| **Matrix Diagonalization ($A = PDP^{-1}$)** | Shows $P = [\vec{v}_1 | \vec{v}_2]$ and $D = \text{diag}(\lambda_1, \lambda_2)$ explaining why matrix powers $A^k = P D^k P^{-1}$ are trivial. |
| **Singular Value Decomposition (SVD)** | Maps the unit circle into an ellipse with semi-axes representing singular values $\sigma_1$ and $\sigma_2$. |
| **Covariance / PCA Mapping** | Visualizes how a 2D Gaussian data point cloud stretches and rotates under matrix $A$. |
| **3D Transformations & Volume** | Full $3 \times 3$ matrix engine rendering a transformed 3D cube with mouse orbit camera and $\det(A_{3\times3})$ volume. |
| **Loss Landscapes & Optimizers** | Real-time particle descent comparing SGD, Momentum, RMSprop, and Adam on convex, saddle, and Rosenbrock valleys. |
| **Automatic Differentiation (Autograd)** | Directed graph evaluating forward pass values and backward pass chain-rule gradients $\partial L / \partial w$. |
| **Dot Product & Projections** | Real-time orthogonal decomposition $\text{proj}_{\vec{v}}(\vec{u}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{v}\|^2}\vec{v}$ with acute/obtuse classification. |

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
