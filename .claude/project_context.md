# TensorForge — Claude Code Project Memory

## Core Details
- **Project**: TensorForge (Linear Algebra & Deep Learning Optimization Engine)
- **Author / Maintainer**: `screen-watcher-elite`
- **Target Repository**: `https://github.com/screen-watcher-elite/tensorforge`
- **Curriculum Alignment**: First-Year B.Tech Semester 1 Linear Algebra, Vector Calculus, and Introductory Machine Learning.
- **Lectern Integration**: Embeddable via `<iframe>` or controlled via `window.TensorForge` JavaScript API.

## Architectural Foundations
1. **100% Zero Dependencies**: Native HTML5 Canvas 2D context, Vanilla JS (ES5/ES6), and Vanilla CSS. No external script imports, CDN links, or node modules required.
2. **Offline-First**: Can be deployed on GitHub Pages or run locally by opening `index.html`.
3. **Mathematical Separation**:
   - `engine.js`: Pure mathematical classes (`Vector2D`, `Matrix2x2`, `Vector3D`, `Matrix3x3`, `solveEigensystem`, `LossFunctions`, `OptimizerParticle`, `AutogradValue`, `AnimationController`).
   - `app.js`: High-performance 60 FPS Canvas rendering loop, user interaction (mouse dragging, panning, zooming), mode management, and UI telemetry.

## Key Modules
1. **Transform 2D**: Interactive grid warping with direct basis vector dragging ($\hat{i}, \hat{j}$).
2. **Eigen Hunter**: Characteristic polynomial $\det(A - \lambda I) = 0$, real/complex root solver, invariant lines, and unit vector collinearity scanner.
3. **Matrix Mult**: Visual transformation sequencing showing non-commutativity ($AB \neq BA$).
4. **Vector Sandbox**: Live dot product, angle readout, vector projection, and Gram-Schmidt orthogonalization.
5. **3D Space**: Interactive 3D cube transformation with 3-axis Euler rotations, non-uniform scaling, orbit camera, and 3D determinant volume.
6. **LossLab**: Real-time gradient descent comparing SGD, Momentum, RMSprop, and Adam on Convex, Saddle, and Rosenbrock 2D loss landscapes.
7. **MicroGraph**: Step-by-step visual Autograd DAG demonstrating forward pass activations and backward pass reverse-mode chain rule.
8. **Theory Vault**: Comprehensive academic notes on rank, nullspace, eigenvalues, SVD, and gradient descent.
9. **Viva Quiz**: 10 interactive examination questions with instant explanations and scoring.
