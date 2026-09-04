# TensorForge — Project Guidelines for Claude Code

> Comprehensive guidelines, architecture notes, and development conventions for Claude Code when maintaining or extending **TensorForge**.

---

## 🧭 Project Mission & Overview

**TensorForge** is an interactive, zero-dependency geometric linear algebra and optimization playground created for First-Year B.Tech AI & Machine Learning coursework and designed as an embeddable component for the **Lectern** study platform.

It visualizes abstract matrix mathematics as tangible geometric transformations across 9 dedicated modules:
1. **2D Linear Transformations & Basis Vectors** ($\hat{i}, \hat{j}$)
2. **Analytical Eigensystem & Invariant Line Hunter** ($\det(A - \lambda I) = 0$)
3. **Matrix Composition Stepper** ($A \times B$ vs $B \times A$)
4. **Vector Sandbox** (Dot products, Gram-Schmidt orthogonalization, projections)
5. **3D VectorSpace** ($3 \times 3$ matrices, unit cube, orbit camera, 3D determinant)
6. **LossLab Optimization Engine** (SGD, Momentum, RMSprop, Adam on 2D loss surfaces)
7. **MicroGraph Autograd Engine** (DAG computation tracer & reverse-mode backpropagation)
8. **Theory Vault** (Curated FY curriculum lecture notes and exam tips)
9. **Viva Exam Quiz** (10-question active recall test with academic scoring)

---

## 🏗️ Architecture & File Structure

```
tensorforge/
├── CLAUDE.md             # Project guidelines and memory for Claude Code
├── .claude/              # Claude Code project settings and memory
│   ├── settings.json     # Permitted tools and IDE configuration
│   └── project_context.md# High-level architecture memory & mathematical notes
├── index.html            # Semantic HTML5 single-page layout & sidebar panels
├── style.css             # Vanilla CSS design system (Dark Glassmorphism 2.0)
├── engine.js             # Standalone mathematical engine (0 DOM dependencies)
├── app.js                # Canvas coordinate engine, interaction handlers, and UI bindings
├── embed-demo.html       # Reference demonstration for Lectern iframe embedding
├── README.md             # Public documentation, academic syllabus mapping
└── LICENSE               # Apache 2.0 Open Source License
```

---

## 📐 Math Engine Rules (`engine.js`)

1. **Zero External Dependencies**: All vector, matrix, eigensystem, loss, and autograd calculations must remain 100% native JavaScript. Do not introduce external libraries (math.js, three.js, gl-matrix).
2. **Pure Math Isolation**: `engine.js` must NEVER query the DOM or manipulate browser globals. All canvas rendering and event listeners belong exclusively in `app.js`.
3. **Numeric Robustness**:
   - Always guard floating-point comparisons with `Engine.EPSILON` (`1e-7`).
   - Guard against matrix singularity ($\det \approx 0$) before inverting.
   - Use `fmtNum()` to sanitize output strings and prevent scientific notation overflow when displaying roots.
4. **Coordinate Convention**:
   - Mathematical space: Standard Cartesian $(x \in [-10, 10], y \in [-10, 10])$ with origin $(0, 0)$ at center.
   - Canvas space: Top-left origin with inverted Y-axis. All translations between math space and screen space must pass through `toScreenX()`, `toScreenY()`, `toMathX()`, and `toMathY()`.

---

## 🎨 UI & Interaction Guidelines (`app.js` & `style.css`)

1. **Design System**: Dark glassmorphism with high-contrast accent colors:
   - Primary Accent (Cyan): `#38bdf8` / `rgba(56, 189, 248, 0.2)`
   - Secondary Accent (Amber/Copper): `#f59e0b` / `rgba(245, 158, 11, 0.2)`
   - Danger/Inverted (Rose): `#f43f5e`
   - Success/Basis $\hat{i}$ (Emerald): `#10b981`
   - Basis $\hat{j}$ (Violet): `#8b5cf6`
2. **Safe Matrix Power Animations**:
   - Power previews ($A^2, A^4, A^8$) must use `powerAnimController` to preview the transformation without permanently corrupting `state.matrix`.
3. **URL State Synchronization**:
   - Only sync valid, finite numbers with $|v| \le 20$ into the URL hash (`updateUrlHash()`).
   - Automatically discard and sanitize invalid or overflowing URL hashes upon page load (`readUrlHash()`).
4. **Responsive Layout**:
   - Sidebar panel must remain scrollable and never overflow viewport height.
   - Mode switcher supports horizontal touch/mouse scrolling for all 9 tabs.

---

## 🛠️ Commands & Development Workflow

### Local Development Preview
No build step or transpilations required. Run any static HTTP server from root:
```bash
# Option 1: Node.js (npx serve)
npx -y serve .

# Option 2: Python 3 built-in server
python -m http.server 8000

# Option 3: Direct browser execution
# Open index.html directly via file:// URL
```

### Syntax & Code Quality Verification
Verify JavaScript syntax across both core files prior to commits:
```bash
node --check engine.js && node --check app.js
```

### Git Commit Guidelines
- Use conventional, semantic commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `perf:`.
- Ensure all modules remain fully functional offline without network requests.
