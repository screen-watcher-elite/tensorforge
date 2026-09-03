/**
 * TensorForge — Interactive Canvas Renderer & Application Logic
 * Full Linear Algebra Sandbox: 60 FPS Canvas, Touch/Mouse Drag,
 * Eigensystem Scanner, Shape Transformations (Circle/SVD, House, F),
 * Matrix Composition Stepper (AB vs BA), Vector Sandbox & PNG Snapshot.
 */

(function () {
  'use strict';

  var Engine = window.TensorForgeEngine;
  var Vector2D = Engine.Vector2D;
  var Matrix2x2 = Engine.Matrix2x2;

  // ── State ─────────────────────────────────────────────────────────────────

  var state = {
    mode: 'transform', // 'transform' | 'eigen' | 'mult' | 'vectors'
    shape: 'square',   // 'square' | 'circle' | 'house' | 'letterF'

    // Primary Transformation Matrix A
    matrix: new Matrix2x2(1.5, 0.5, 0.5, 1.2),
    // Secondary Matrix B for Composition
    matrixB: new Matrix2x2(0.8, -0.6, 0.6, 0.8),

    multT: 0,
    multOrder: 'AB', // 'AB' or 'BA'
    multPlaying: false,
    multAnimId: null,

    // Custom test vector & Eigen Hunter
    customVec: new Vector2D(1.0, 1.0),
    eigenProbe: new Vector2D(1.0, 0.0), // Unit vector for eigen hunt

    // Display Toggles
    showCustomVec: true,
    showTransformedGrid: true,
    showEigenSpans: true,

    // Vector Sandbox Mode
    vecU: new Vector2D(2.0, 1.0),
    vecV: new Vector2D(1.0, 2.0),

    // Viewport transform
    scale: 65, // Pixels per math unit
    panX: 0,
    panY: 0,

    // Drag Interaction
    draggingTarget: null, // 'i' | 'j' | 'v' | 'u' | 'v_sandbox' | 'probe' | 'pan'
    dragStartMouse: { x: 0, y: 0 },
    dragStartPan: { x: 0, y: 0 },
    hoverTarget: null
  };

  // Fixed 2D Gaussian scatter distribution for PCA / Covariance visualization
  var scatterPoints = [];
  (function initScatter() {
    for (var p = 0; p < 45; p++) {
      var u1 = (p * 17 + 13) % 45 / 45;
      var u2 = (p * 31 + 7) % 45 / 45;
      var r = Math.sqrt(-2 * Math.log(u1 + 0.05)) * 0.55;
      var th = 2 * Math.PI * u2;
      scatterPoints.push(new Vector2D(r * Math.cos(th), r * Math.sin(th) * 0.55));
    }
  })();

  var animController = new Engine.AnimationController(function (currentMatrix) {
    state.matrix = currentMatrix;
    syncMatrixInputs();
    updateTelemetry();
    render();
  });

  // ── DOM References ────────────────────────────────────────────────────────

  var $ = function (id) { return document.getElementById(id); };

  var canvas = $('matrix-canvas');
  var ctx = canvas.getContext('2d');

  // Matrix A Inputs
  var matAInput = $('mat-a');
  var matBInput = $('mat-b');
  var matCInput = $('mat-c');
  var matDInput = $('mat-d');

  // Matrix B Inputs (Mode 3)
  var matBAInput = $('mat-b-a');
  var matBBInput = $('mat-b-b');
  var matBCInput = $('mat-b-c');
  var matBDInput = $('mat-b-d');

  // Telemetry Elements
  var readoutDet = $('telemetry-det');
  var badgeDet = $('badge-det');
  var readoutTrace = $('telemetry-trace');
  var readoutRank = $('telemetry-rank');
  var eigenRow1 = $('eigen-val-1');
  var eigenRow2 = $('eigen-val-2');

  // Eigen Mode Elements
  var eigenFormulaSub = $('eigen-formula-sub');
  var eigenQuadExpanded = $('eigen-quad-expanded');
  var eigenDiscVal = $('eigen-disc-val');
  var badgeDisc = $('badge-disc');
  var valCollinearity = $('val-collinearity');
  var barCollinearity = $('bar-collinearity');
  var eigenFoundAlert = $('eigen-found-alert');

  // Matrix Chain Elements (Mode 3)
  var multDrawer = $('mult-drawer');
  var multSlider = $('slider-mult');
  var multTDisplay = $('val-mult-t');
  var btnSwapMult = $('btn-swap-mult');
  var valProdAB = $('val-prod-ab');
  var valProdBA = $('val-prod-ba');

  // Vector Sandbox Elements (Mode 4)
  var vecUXInput = $('vec-u-x');
  var vecUYInput = $('vec-u-y');
  var vecVXInput = $('vec-v-x');
  var vecVYInput = $('vec-v-y');
  var valDotProduct = $('val-dot-product');
  var badgeDotAngle = $('badge-dot-angle');
  var valVectorAngle = $('val-vector-angle');
  var valMagU = $('val-mag-u');
  var valMagV = $('val-mag-v');

  // Sliders
  var rotationSlider = $('slider-rotation');
  var rotationValue = $('val-rotation');
  var shearSlider = $('slider-shear');
  var shearValue = $('val-shear');

  // HUD & Actions
  var hudCoords = $('hud-coords');
  var modalHelp = $('modal-help');

  // ── Canvas Sizing & Retina Resolution ─────────────────────────────────────

  var dpr = window.devicePixelRatio || 1;
  var viewWidth = 0;
  var viewHeight = 0;

  function resizeCanvas() {
    var rect = canvas.parentElement.getBoundingClientRect();
    viewWidth = rect.width;
    viewHeight = rect.height;

    canvas.width = viewWidth * dpr;
    canvas.height = viewHeight * dpr;

    // Reset and apply DPR scale cleanly
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    render();
  }

  // ── Coordinate Conversions ────────────────────────────────────────────────

  function worldToScreen(wx, wy) {
    var originX = viewWidth / 2 + state.panX;
    var originY = viewHeight / 2 + state.panY;
    return {
      x: originX + wx * state.scale,
      y: originY - wy * state.scale
    };
  }

  function screenToWorld(sx, sy) {
    var originX = viewWidth / 2 + state.panX;
    var originY = viewHeight / 2 + state.panY;
    return {
      x: (sx - originX) / state.scale,
      y: -(sy - originY) / state.scale
    };
  }

  // ── Rendering Engine (60 FPS) ─────────────────────────────────────────────

  function render() {
    ctx.clearRect(0, 0, viewWidth, viewHeight);

    // 1. Static Cartesian background grid
    drawBackgroundGrid();

    if (state.mode === 'vectors') {
      // Vector Sandbox mode
      drawVectorSandbox();
    } else {
      // Linear transformation modes
      var activeMatrix = getActiveMatrixForRender();

      // 2. Transformed coordinate grid (warped lines)
      if (state.showTransformedGrid) {
        drawTransformedGrid(activeMatrix);
      }

      // 3. Render Chosen Shape / Parallelogram
      drawTransformedShape(activeMatrix);

      // 4. Invariant Eigenvector Span lines
      if (state.showEigenSpans && state.mode !== 'mult') {
        drawEigenSpanLines(activeMatrix);
      }

      // 5. Axes
      drawAxes();

      // 6. Basis vectors i-hat and j-hat
      drawBasisVectors(activeMatrix);

      // 7. Custom vector / Eigen Hunter probe
      if (state.mode === 'eigen') {
        drawEigenHunter(activeMatrix);
      } else if (state.showCustomVec) {
        drawCustomVector(activeMatrix);
      }
    }
  }

  function getActiveMatrixForRender() {
    if (state.mode === 'mult') {
      var t = state.multT;
      var M1 = state.multOrder === 'AB' ? state.matrixB : state.matrix;
      var M2 = state.multOrder === 'AB' ? state.matrix : state.matrixB;
      var product = M2.multiply(M1);

      if (t <= 0.5) {
        var localT = t * 2;
        return Matrix2x2.lerp(Matrix2x2.identity(), M1, localT);
      } else {
        var localT = (t - 0.5) * 2;
        return Matrix2x2.lerp(M1, product, localT);
      }
    }
    return state.matrix;
  }

  // ── Draw Background Cartesian Grid ────────────────────────────────────────

  function drawBackgroundGrid() {
    var min = screenToWorld(0, viewHeight);
    var max = screenToWorld(viewWidth, 0);

    var startX = Math.floor(min.x);
    var endX = Math.ceil(max.x);
    var startY = Math.floor(min.y);
    var endY = Math.ceil(max.y);

    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';

    ctx.beginPath();
    for (var x = startX; x <= endX; x++) {
      var p1 = worldToScreen(x, min.y);
      var p2 = worldToScreen(x, max.y);
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
    }
    for (var y = startY; y <= endY; y++) {
      var p1 = worldToScreen(min.x, y);
      var p2 = worldToScreen(max.x, y);
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
    }
    ctx.stroke();
  }

  // ── Draw Transformed Coordinate Grid ──────────────────────────────────────

  function drawTransformedGrid(matrix) {
    var range = 12;
    var col1 = matrix.getCol1();
    var col2 = matrix.getCol2();

    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.12)';

    ctx.beginPath();
    for (var i = -range; i <= range; i++) {
      var base = col1.scale(i);
      var pStart = worldToScreen(base.x - col2.x * range, base.y - col2.y * range);
      var pEnd = worldToScreen(base.x + col2.x * range, base.y + col2.y * range);
      ctx.moveTo(pStart.x, pStart.y);
      ctx.lineTo(pEnd.x, pEnd.y);
    }
    for (var j = -range; j <= range; j++) {
      var base = col2.scale(j);
      var pStart = worldToScreen(base.x - col1.x * range, base.y - col1.y * range);
      var pEnd = worldToScreen(base.x + col1.x * range, base.y + col1.y * range);
      ctx.moveTo(pStart.x, pStart.y);
      ctx.lineTo(pEnd.x, pEnd.y);
    }
    ctx.stroke();
  }

  // ── Draw Shapes (Unit Square, Circle/SVD, House, Letter F) ────────────────

  function drawTransformedShape(matrix) {
    var det = matrix.determinant();
    var fillColor = det >= 0 ? 'rgba(6, 182, 212, 0.14)' : 'rgba(245, 158, 11, 0.16)';
    var strokeColor = det >= 0 ? 'rgba(6, 182, 212, 0.45)' : 'rgba(245, 158, 11, 0.55)';

    if (Math.abs(det) < Engine.EPSILON) {
      fillColor = 'transparent';
      strokeColor = 'rgba(239, 68, 68, 0.8)';
    }

    ctx.save();
    ctx.fillStyle = fillColor;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;

    if (state.shape === 'square') {
      // Unit Square: (0,0) -> i -> i+j -> j
      var o = worldToScreen(0, 0);
      var iPos = worldToScreen(matrix.a, matrix.c);
      var sum = worldToScreen(matrix.a + matrix.b, matrix.c + matrix.d);
      var jPos = worldToScreen(matrix.b, matrix.d);

      ctx.beginPath();
      ctx.moveTo(o.x, o.y);
      ctx.lineTo(iPos.x, iPos.y);
      ctx.lineTo(sum.x, sum.y);
      ctx.lineTo(jPos.x, jPos.y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Determinant area label
      if (Math.abs(det) > 0.05) {
        var centerX = (o.x + iPos.x + sum.x + jPos.x) / 4;
        var centerY = (o.y + iPos.y + sum.y + jPos.y) / 4;
        ctx.font = '600 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
        ctx.fillStyle = det > 0 ? '#67e8f9' : '#fcd34d';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Area: ' + Math.abs(det).toFixed(2), centerX, centerY);
      }

    } else if (state.shape === 'circle') {
      // Unit Circle transformed to Ellipse (SVD Visualization)
      var steps = 64;
      ctx.beginPath();
      for (var s = 0; s <= steps; s++) {
        var theta = (s / steps) * Math.PI * 2;
        var wx = Math.cos(theta);
        var wy = Math.sin(theta);
        var tw = matrix.apply(new Vector2D(wx, wy));
        var sp = worldToScreen(tw.x, tw.y);
        if (s === 0) ctx.moveTo(sp.x, sp.y);
        else ctx.lineTo(sp.x, sp.y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Show SVD singular values on ellipse
      var svd = Engine.computeSVD2x2(matrix);
      var centerScreen = worldToScreen(0, 0);
      ctx.font = '600 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillStyle = '#c7d2fe';
      ctx.fillText('SVD: σ₁=' + svd.sigma1.toFixed(2) + ', σ₂=' + svd.sigma2.toFixed(2), centerScreen.x + 12, centerScreen.y + 16);

    } else if (state.shape === 'house') {
      // Classic Computer Graphics House
      var housePts = [
        new Vector2D(-0.5, 0), new Vector2D(0.5, 0),
        new Vector2D(0.5, 0.8), new Vector2D(0, 1.3),
        new Vector2D(-0.5, 0.8)
      ];
      ctx.beginPath();
      housePts.forEach(function (pt, idx) {
        var t = matrix.apply(pt);
        var sp = worldToScreen(t.x, t.y);
        if (idx === 0) ctx.moveTo(sp.x, sp.y);
        else ctx.lineTo(sp.x, sp.y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

    } else if (state.shape === 'letterF') {
      // Letter F polygon to demonstrate reflection/chirality
      var fPts = [
        new Vector2D(0, 0), new Vector2D(0.25, 0), new Vector2D(0.25, 0.6),
        new Vector2D(0.7, 0.6), new Vector2D(0.7, 0.8), new Vector2D(0.25, 0.8),
        new Vector2D(0.25, 1.1), new Vector2D(0.9, 1.1), new Vector2D(0.9, 1.35),
        new Vector2D(0, 1.35)
      ];
      ctx.beginPath();
      fPts.forEach(function (pt, idx) {
        var t = matrix.apply(pt);
        var sp = worldToScreen(t.x, t.y);
        if (idx === 0) ctx.moveTo(sp.x, sp.y);
        else ctx.lineTo(sp.x, sp.y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

    } else if (state.shape === 'cloud') {
      // 2D Gaussian Data Scatter (Covariance / PCA Transformation)
      ctx.fillStyle = det >= 0 ? 'rgba(99, 102, 241, 0.7)' : 'rgba(245, 158, 11, 0.7)';
      scatterPoints.forEach(function (pt) {
        var t = matrix.apply(pt);
        var sp = worldToScreen(t.x, t.y);
        ctx.beginPath();
        ctx.arc(sp.x, sp.y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });

      var centerScreen = worldToScreen(0, 0);
      ctx.font = '600 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillStyle = '#a5b4fc';
      ctx.fillText('Data Scatter (Covariance mapped by A)', centerScreen.x + 12, centerScreen.y - 14);
    }

    ctx.restore();
  }

  // ── Draw Invariant Eigenvector Span Lines ──────────────────────────────────

  function drawEigenSpanLines(matrix) {
    var eigens = Engine.solveEigensystem(matrix);
    if (!eigens.isReal || eigens.eigenvectors.length === 0) return;

    var colors = ['#f59e0b', '#a855f7'];
    var lineLength = 22;

    eigens.eigenvectors.forEach(function (ev, index) {
      var v = ev.vector;
      var p1 = worldToScreen(-v.x * lineLength, -v.y * lineLength);
      var p2 = worldToScreen(v.x * lineLength, v.y * lineLength);

      ctx.save();
      ctx.strokeStyle = colors[index % colors.length];
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 6]);
      ctx.globalAlpha = 0.65;

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      var labelPos = worldToScreen(v.x * 2.8, v.y * 2.8);
      ctx.fillStyle = colors[index % colors.length];
      ctx.font = '600 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillText('Span(v' + (index + 1) + ') λ=' + ev.lambda.toFixed(2), labelPos.x + 6, labelPos.y - 6);

      ctx.restore();
    });
  }

  // ── Draw Eigen Hunter (Mode 2) ────────────────────────────────────────────

  function drawEigenHunter(matrix) {
    var o = worldToScreen(0, 0);

    // Unit circle track
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    var rScreen = state.scale;
    ctx.arc(o.x, o.y, rScreen, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    // Probe vector x
    var pX = worldToScreen(state.eigenProbe.x, state.eigenProbe.y);
    drawArrow(o.x, o.y, pX.x, pX.y, '#f59e0b', 2.5);
    drawVectorHandle(pX.x, pX.y, '#f59e0b', 'probe', state.hoverTarget === 'probe');
    drawVectorLabel('x', pX.x, pX.y, '#f59e0b');

    // Transformed vector Ax
    var tProbe = matrix.apply(state.eigenProbe);
    var pAx = worldToScreen(tProbe.x, tProbe.y);
    drawArrow(o.x, o.y, pAx.x, pAx.y, '#38bdf8', 2);
    drawVectorLabel('Ax', pAx.x, pAx.y, '#38bdf8');

    // Calculate Collinearity
    var magT = tProbe.magnitude();
    var cosAngle = magT > Engine.EPSILON ? Math.abs(state.eigenProbe.dot(tProbe) / magT) : 1;
    var matchPct = Math.round(cosAngle * 100);

    valCollinearity.textContent = matchPct + '%';
    barCollinearity.style.width = matchPct + '%';

    var isCollinear = cosAngle > 0.99;
    eigenFoundAlert.classList.toggle('hidden', !isCollinear);
  }

  // ── Draw Standard Axes ────────────────────────────────────────────────────

  function drawAxes() {
    var o = worldToScreen(0, 0);

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';

    ctx.beginPath();
    ctx.moveTo(0, o.y);
    ctx.lineTo(viewWidth, o.y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(o.x, 0);
    ctx.lineTo(o.x, viewHeight);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.beginPath();
    ctx.arc(o.x, o.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // ── Draw Basis Vectors ────────────────────────────────────────────────────

  function drawBasisVectors(matrix) {
    var o = worldToScreen(0, 0);
    var iPos = worldToScreen(matrix.a, matrix.c);
    var jPos = worldToScreen(matrix.b, matrix.d);

    // Vector i-hat (Column 1)
    drawArrow(o.x, o.y, iPos.x, iPos.y, '#f43f5e', 2.5);
    drawVectorHandle(iPos.x, iPos.y, '#f43f5e', 'i', state.hoverTarget === 'i');
    drawVectorLabel('î [' + matrix.a.toFixed(1) + ', ' + matrix.c.toFixed(1) + ']', iPos.x, iPos.y, '#f43f5e');

    // Vector j-hat (Column 2)
    drawArrow(o.x, o.y, jPos.x, jPos.y, '#06b6d4', 2.5);
    drawVectorHandle(jPos.x, jPos.y, '#06b6d4', 'j', state.hoverTarget === 'j');
    drawVectorLabel('ĵ [' + matrix.b.toFixed(1) + ', ' + matrix.d.toFixed(1) + ']', jPos.x, jPos.y, '#06b6d4');
  }

  // ── Draw Custom Test Vector ───────────────────────────────────────────────

  function drawCustomVector(matrix) {
    var o = worldToScreen(0, 0);
    var vInput = worldToScreen(state.customVec.x, state.customVec.y);
    var transformed = matrix.apply(state.customVec);
    var vOutput = worldToScreen(transformed.x, transformed.y);

    ctx.save();
    ctx.setLineDash([4, 4]);
    drawArrow(o.x, o.y, vInput.x, vInput.y, 'rgba(16, 185, 129, 0.5)', 1.5);
    ctx.restore();

    drawVectorHandle(vInput.x, vInput.y, '#10b981', 'v', state.hoverTarget === 'v');

    drawArrow(o.x, o.y, vOutput.x, vOutput.y, '#10b981', 2.5);
    drawVectorLabel('T(v) [' + transformed.x.toFixed(1) + ', ' + transformed.y.toFixed(1) + ']', vOutput.x, vOutput.y, '#10b981');
  }

  // ── Draw Vector Sandbox Mode ──────────────────────────────────────────────

  function drawVectorSandbox() {
    var o = worldToScreen(0, 0);
    var uPos = worldToScreen(state.vecU.x, state.vecU.y);
    var vPos = worldToScreen(state.vecV.x, state.vecV.y);

    drawAxes();

    // Vector U
    drawArrow(o.x, o.y, uPos.x, uPos.y, '#f43f5e', 2.5);
    drawVectorHandle(uPos.x, uPos.y, '#f43f5e', 'u', state.hoverTarget === 'u');
    drawVectorLabel('u [' + state.vecU.x.toFixed(1) + ', ' + state.vecU.y.toFixed(1) + ']', uPos.x, uPos.y, '#f43f5e');

    // Vector V
    drawArrow(o.x, o.y, vPos.x, vPos.y, '#06b6d4', 2.5);
    drawVectorHandle(vPos.x, vPos.y, '#06b6d4', 'v_sandbox', state.hoverTarget === 'v_sandbox');
    drawVectorLabel('v [' + state.vecV.x.toFixed(1) + ', ' + state.vecV.y.toFixed(1) + ']', vPos.x, vPos.y, '#06b6d4');

    // Parallelogram sum (U + V)
    var sum = state.vecU.add(state.vecV);
    var sumPos = worldToScreen(sum.x, sum.y);

    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.beginPath();
    ctx.moveTo(uPos.x, uPos.y);
    ctx.lineTo(sumPos.x, sumPos.y);
    ctx.lineTo(vPos.x, vPos.y);
    ctx.stroke();
    ctx.restore();

    drawArrow(o.x, o.y, sumPos.x, sumPos.y, '#10b981', 2);
    drawVectorLabel('u+v [' + sum.x.toFixed(1) + ', ' + sum.y.toFixed(1) + ']', sumPos.x, sumPos.y, '#10b981');

    // Orthogonal Projection of U onto V
    var proj = state.vecU.projectOnto(state.vecV);
    var projPos = worldToScreen(proj.x, proj.y);

    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = '#f59e0b';
    ctx.beginPath();
    ctx.moveTo(uPos.x, uPos.y);
    ctx.lineTo(projPos.x, projPos.y);
    ctx.stroke();

    drawArrow(o.x, o.y, projPos.x, projPos.y, '#f59e0b', 3);
    ctx.restore();

    updateVectorSandboxTelemetry();
  }

  function updateVectorSandboxTelemetry() {
    var dot = state.vecU.dot(state.vecV);
    var magU = state.vecU.magnitude();
    var magV = state.vecV.magnitude();
    var cosTheta = (magU > 0 && magV > 0) ? Math.max(-1, Math.min(1, dot / (magU * magV))) : 1;
    var angleDeg = (Math.acos(cosTheta) * 180) / Math.PI;

    valDotProduct.textContent = dot.toFixed(2);
    valMagU.textContent = magU.toFixed(2);
    valMagV.textContent = magV.toFixed(2);
    valVectorAngle.textContent = angleDeg.toFixed(1) + '°';

    if (Math.abs(dot) < Engine.EPSILON) {
      badgeDotAngle.textContent = 'Orthogonal (90°)';
      badgeDotAngle.className = 'telemetry-badge badge-det-pos';
    } else if (dot > 0) {
      badgeDotAngle.textContent = 'Acute Angle (<90°)';
      badgeDotAngle.className = 'telemetry-badge badge-det-pos';
    } else {
      badgeDotAngle.textContent = 'Obtuse Angle (>90°)';
      badgeDotAngle.className = 'telemetry-badge badge-det-neg';
    }
  }

  // ── Primitive Drawing Helpers ─────────────────────────────────────────────

  function drawArrow(x1, y1, x2, y2, color, width) {
    var headLength = 12;
    var dx = x2 - x1;
    var dy = y2 - y1;
    var angle = Math.atan2(dy, dx);
    var length = Math.sqrt(dx * dx + dy * dy);

    if (length < 2) return;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLength * Math.cos(angle - Math.PI / 6), y2 - headLength * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - headLength * Math.cos(angle + Math.PI / 6), y2 - headLength * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }

  function drawVectorHandle(x, y, color, targetName, isHovered) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, isHovered ? 8 : 5, 0, Math.PI * 2);
    ctx.fill();

    if (isHovered) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawVectorLabel(text, x, y, color) {
    ctx.save();
    ctx.font = '600 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
    ctx.fillStyle = color;
    ctx.fillText(text, x + 8, y - 8);
    ctx.restore();
  }

  // ── Math & Telemetry Updates ──────────────────────────────────────────────

  function updateTelemetry() {
    var m = state.matrix;
    var det = m.determinant();
    var tr = m.trace();
    var rk = m.rank();

    // Determinant
    readoutDet.textContent = det.toFixed(2);
    if (Math.abs(det) < Engine.EPSILON) {
      badgeDet.textContent = 'Dimension Collapsed';
      badgeDet.className = 'telemetry-badge badge-det-zero';
    } else if (det > 0) {
      badgeDet.textContent = 'Orientation Preserved';
      badgeDet.className = 'telemetry-badge badge-det-pos';
    } else {
      badgeDet.textContent = 'Orientation Inverted';
      badgeDet.className = 'telemetry-badge badge-det-neg';
    }

    // Trace & Rank
    readoutTrace.textContent = tr.toFixed(2);
    readoutRank.textContent = rk;

    // Eigensystem
    var eigens = Engine.solveEigensystem(m);
    eigenFormulaSub.textContent = eigens.equationString;
    eigenQuadExpanded.textContent = eigens.discriminantString;
    eigenDiscVal.textContent = 'Δ = ' + eigens.discriminant.toFixed(2);

    if (eigens.isReal) {
      eigenRow1.textContent = 'λ₁ = ' + eigens.eigenvalues[0].value.toFixed(2);
      eigenRow2.textContent = 'λ₂ = ' + eigens.eigenvalues[1].value.toFixed(2);
      badgeDisc.textContent = '2 Distinct Real Eigenvalues';
      badgeDisc.className = 'telemetry-badge badge-det-pos';
    } else {
      var re = eigens.eigenvalues[0].real.toFixed(2);
      var im = eigens.eigenvalues[0].imag.toFixed(2);
      eigenRow1.textContent = 'λ₁ = ' + re + ' + ' + im + 'i';
      eigenRow2.textContent = 'λ₂ = ' + re + ' - ' + im + 'i';
      badgeDisc.textContent = 'Complex Roots (Pure Rotation/Spiral)';
      badgeDisc.className = 'telemetry-badge badge-det-neg';
    }

    // Diagonalization (A = P D P^-1) readout
    var diagText = $('diag-status-text');
    if (diagText) {
      if (eigens.isReal && eigens.eigenvectors.length >= 2 && Math.abs(eigens.eigenvalues[0].value - eigens.eigenvalues[1].value) > Engine.EPSILON) {
        var ev1 = eigens.eigenvectors[0].vector;
        var ev2 = eigens.eigenvectors[1].vector;
        diagText.innerHTML = 'Diagonalizable over ℝ:<br><strong>P</strong> = [ ' + ev1.x.toFixed(2) + ', ' + ev2.x.toFixed(2) + ' ; ' + ev1.y.toFixed(2) + ', ' + ev2.y.toFixed(2) + ' ]<br><strong>D</strong> = diag(' + eigens.eigenvalues[0].value.toFixed(2) + ', ' + eigens.eigenvalues[1].value.toFixed(2) + ')<br>Powers: Aᵏ = P·Dᵏ·P⁻¹';
      } else if (!eigens.isReal) {
        diagText.textContent = 'Cannot be diagonalized over ℝ (no real eigenbasis). A represents a rotation/spiral.';
      } else {
        diagText.textContent = 'Defective or uniform scalar matrix.';
      }
    }

    // Matrix Multiplication Comparison (Mode 3)
    updateMultComparison();
    updateUrlHash();
  }

  function updateMultComparison() {
    var AB = state.matrix.multiply(state.matrixB);
    var BA = state.matrixB.multiply(state.matrix);

    valProdAB.innerHTML = '[ ' + AB.a.toFixed(2) + ', ' + AB.b.toFixed(2) + ' ]<br>[ ' + AB.c.toFixed(2) + ', ' + AB.d.toFixed(2) + ' ]';
    valProdBA.innerHTML = '[ ' + BA.a.toFixed(2) + ', ' + BA.b.toFixed(2) + ' ]<br>[ ' + BA.c.toFixed(2) + ', ' + BA.d.toFixed(2) + ' ]';
  }

  function syncMatrixInputs() {
    matAInput.value = state.matrix.a.toFixed(2);
    matBInput.value = state.matrix.b.toFixed(2);
    matCInput.value = state.matrix.c.toFixed(2);
    matDInput.value = state.matrix.d.toFixed(2);
  }

  function readMatrixInputs() {
    var a = parseFloat(matAInput.value) || 0;
    var b = parseFloat(matBInput.value) || 0;
    var c = parseFloat(matCInput.value) || 0;
    var d = parseFloat(matDInput.value) || 0;

    state.matrix = new Matrix2x2(a, b, c, d);
    updateTelemetry();
    render();
  }

  function readMatrixBInputs() {
    var a = parseFloat(matBAInput.value) || 0;
    var b = parseFloat(matBBInput.value) || 0;
    var c = parseFloat(matBCInput.value) || 0;
    var d = parseFloat(matBDInput.value) || 0;

    state.matrixB = new Matrix2x2(a, b, c, d);
    updateTelemetry();
    render();
  }

  function readVectorInputs() {
    state.vecU.x = parseFloat(vecUXInput.value) || 0;
    state.vecU.y = parseFloat(vecUYInput.value) || 0;
    state.vecV.x = parseFloat(vecVXInput.value) || 0;
    state.vecV.y = parseFloat(vecVYInput.value) || 0;
    render();
  }

  // ── Drag & Touch Event Handling ───────────────────────────────────────────

  function getMousePos(e) {
    var rect = canvas.getBoundingClientRect();
    var clientX = e.clientX;
    var clientY = e.clientY;

    // Support touch
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }

    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  }

  function checkHitTarget(sx, sy) {
    var threshold = 20;

    if (state.mode === 'vectors') {
      var uPos = worldToScreen(state.vecU.x, state.vecU.y);
      var vPos = worldToScreen(state.vecV.x, state.vecV.y);
      if (Math.hypot(sx - uPos.x, sy - uPos.y) < threshold) return 'u';
      if (Math.hypot(sx - vPos.x, sy - vPos.y) < threshold) return 'v_sandbox';
      return null;
    }

    if (state.mode === 'eigen') {
      var pPos = worldToScreen(state.eigenProbe.x, state.eigenProbe.y);
      if (Math.hypot(sx - pPos.x, sy - pPos.y) < threshold) return 'probe';
    }

    var iPos = worldToScreen(state.matrix.a, state.matrix.c);
    var jPos = worldToScreen(state.matrix.b, state.matrix.d);
    var vPos = worldToScreen(state.customVec.x, state.customVec.y);

    if (Math.hypot(sx - iPos.x, sy - iPos.y) < threshold) return 'i';
    if (Math.hypot(sx - jPos.x, sy - jPos.y) < threshold) return 'j';
    if (state.showCustomVec && Math.hypot(sx - vPos.x, sy - vPos.y) < threshold) return 'v';

    return null;
  }

  function onPointerDown(e) {
    var pos = getMousePos(e);
    var hit = checkHitTarget(pos.x, pos.y);

    if (hit) {
      state.draggingTarget = hit;
    } else {
      state.draggingTarget = 'pan';
      state.dragStartMouse = pos;
      state.dragStartPan = { x: state.panX, y: state.panY };
    }
  }

  function onPointerMove(e) {
    var pos = getMousePos(e);
    var world = screenToWorld(pos.x, pos.y);

    hudCoords.innerHTML = 'Cursor: <strong>[' + world.x.toFixed(2) + ', ' + world.y.toFixed(2) + ']</strong>';

    if (!state.draggingTarget) {
      var hit = checkHitTarget(pos.x, pos.y);
      if (hit !== state.hoverTarget) {
        state.hoverTarget = hit;
        canvas.style.cursor = hit ? 'grab' : 'crosshair';
        render();
      }
      return;
    }

    var snap = e.shiftKey ? 0.25 : 0.05;
    var snappedX = Math.round(world.x / snap) * snap;
    var snappedY = Math.round(world.y / snap) * snap;

    if (state.draggingTarget === 'i') {
      state.matrix.a = snappedX;
      state.matrix.c = snappedY;
      syncMatrixInputs();
      updateTelemetry();
      render();
    } else if (state.draggingTarget === 'j') {
      state.matrix.b = snappedX;
      state.matrix.d = snappedY;
      syncMatrixInputs();
      updateTelemetry();
      render();
    } else if (state.draggingTarget === 'v') {
      state.customVec.x = snappedX;
      state.customVec.y = snappedY;
      render();
    } else if (state.draggingTarget === 'probe') {
      // Constrain probe to unit circle
      var angle = Math.atan2(world.y, world.x);
      state.eigenProbe = new Vector2D(Math.cos(angle), Math.sin(angle));
      render();
    } else if (state.draggingTarget === 'u') {
      state.vecU.x = snappedX;
      state.vecU.y = snappedY;
      vecUXInput.value = snappedX.toFixed(1);
      vecUYInput.value = snappedY.toFixed(1);
      render();
    } else if (state.draggingTarget === 'v_sandbox') {
      state.vecV.x = snappedX;
      state.vecV.y = snappedY;
      vecVXInput.value = snappedX.toFixed(1);
      vecVYInput.value = snappedY.toFixed(1);
      render();
    } else if (state.draggingTarget === 'pan') {
      var dx = pos.x - state.dragStartMouse.x;
      var dy = pos.y - state.dragStartMouse.y;
      state.panX = state.dragStartPan.x + dx;
      state.panY = state.dragStartPan.y + dy;
      render();
    }
  }

  function onPointerUp() {
    state.draggingTarget = null;
    canvas.style.cursor = state.hoverTarget ? 'grab' : 'crosshair';
  }

  // Attach Mouse & Touch
  canvas.addEventListener('mousedown', onPointerDown);
  window.addEventListener('mousemove', onPointerMove);
  window.addEventListener('mouseup', onPointerUp);

  canvas.addEventListener('touchstart', function (e) {
    e.preventDefault();
    onPointerDown(e);
  }, { passive: false });

  window.addEventListener('touchmove', function (e) {
    if (state.draggingTarget) e.preventDefault();
    onPointerMove(e);
  }, { passive: false });

  window.addEventListener('touchend', onPointerUp);

  // Wheel to zoom
  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    var zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
    state.scale = Math.min(180, Math.max(25, state.scale * zoomFactor));
    render();
  }, { passive: false });

  // ── Presets & Transformations ─────────────────────────────────────────────

  var PRESETS = {
    identity: Matrix2x2.identity(),
    rotate45: Matrix2x2.rotation(Math.PI / 4),
    rotate90: Matrix2x2.rotation(Math.PI / 2),
    shearX: Matrix2x2.shearX(1.2),
    shearY: Matrix2x2.shearY(1.0),
    scaleUp: Matrix2x2.scale(1.8, 1.4),
    reflectX: new Matrix2x2(1, 0, 0, -1),
    reflectY: new Matrix2x2(-1, 0, 0, 1),
    singular: new Matrix2x2(1, 1, 1, 1)
  };

  function applyPreset(name) {
    var target = PRESETS[name];
    if (!target) return;

    document.querySelectorAll('.btn-preset').forEach(function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-preset') === name);
    });

    animController.start(state.matrix, target, 500);
  }

  // ── Snapshot Export (PNG) ─────────────────────────────────────────────────

  function exportSnapshot() {
    // Render to high-res offscreen canvas
    var offscreen = document.createElement('canvas');
    offscreen.width = 1600;
    offscreen.height = 1200;
    var offCtx = offscreen.getContext('2d');

    // Fill dark background
    offCtx.fillStyle = '#070a13';
    offCtx.fillRect(0, 0, offscreen.width, offscreen.height);

    // Copy from main canvas centered
    offCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, offscreen.width, offscreen.height);

    // Add watermark
    offCtx.font = '700 16px Inter, sans-serif';
    offCtx.fillStyle = '#6366f1';
    offCtx.fillText('📐 TensorForge — Geometric Linear Algebra', 30, 45);

    var link = document.createElement('a');
    link.download = 'tensorforge-matrix-' + Date.now() + '.png';
    link.href = offscreen.toDataURL('image/png');
    link.click();
  }

  // ── Setup UI Event Listeners ──────────────────────────────────────────────

  function initEvents() {
    window.addEventListener('resize', resizeCanvas);

    // Matrix A Inputs
    [matAInput, matBInput, matCInput, matDInput].forEach(function (inp) {
      inp.addEventListener('input', readMatrixInputs);
    });

    // Matrix B Inputs
    [matBAInput, matBBInput, matBCInput, matBDInput].forEach(function (inp) {
      inp.addEventListener('input', readMatrixBInputs);
    });

    // Vector Sandbox Inputs
    [vecUXInput, vecUYInput, vecVXInput, vecVYInput].forEach(function (inp) {
      inp.addEventListener('input', readVectorInputs);
    });

    // Presets
    document.querySelectorAll('.btn-preset[data-preset]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        applyPreset(this.getAttribute('data-preset'));
      });
    });

    // Shape buttons
    document.querySelectorAll('.btn-shape[data-shape]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.btn-shape').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        state.shape = this.getAttribute('data-shape');
        render();
      });
    });

    // Sliders
    rotationSlider.addEventListener('input', function () {
      var deg = parseFloat(this.value);
      rotationValue.textContent = deg + '°';
      var rad = (deg * Math.PI) / 180;
      state.matrix = Matrix2x2.rotation(rad);
      syncMatrixInputs();
      updateTelemetry();
      render();
    });

    shearSlider.addEventListener('input', function () {
      var k = parseFloat(this.value);
      shearValue.textContent = k.toFixed(2);
      state.matrix = Matrix2x2.shearX(k);
      syncMatrixInputs();
      updateTelemetry();
      render();
    });

    // Mode tabs
    document.querySelectorAll('.mode-btn[data-mode]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.mode-btn').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        setMode(this.getAttribute('data-mode'));
      });
    });

    // Zoom & Pan Actions
    $('btn-zoom-in').addEventListener('click', function () {
      state.scale = Math.min(180, state.scale * 1.2);
      render();
    });

    $('btn-zoom-out').addEventListener('click', function () {
      state.scale = Math.max(25, state.scale / 1.2);
      render();
    });

    $('btn-recenter').addEventListener('click', function () {
      state.panX = 0;
      state.panY = 0;
      state.scale = 65;
      render();
    });

    $('btn-reset').addEventListener('click', function () {
      applyPreset('identity');
    });

    // Header Actions
    $('btn-snapshot').addEventListener('click', exportSnapshot);

    $('btn-toggle-grid').addEventListener('click', function () {
      state.showTransformedGrid = !state.showTransformedGrid;
      this.style.color = state.showTransformedGrid ? 'var(--accent-primary)' : 'var(--text-muted)';
      render();
    });

    // Help Modal
    $('btn-help').addEventListener('click', function () { modalHelp.classList.remove('hidden'); });
    $('btn-close-help').addEventListener('click', function () { modalHelp.classList.add('hidden'); });
    modalHelp.addEventListener('click', function (e) { if (e.target === modalHelp) modalHelp.classList.add('hidden'); });

    // Vector Sandbox quick helpers
    var btnNormU = $('btn-normalize-u');
    if (btnNormU) {
      btnNormU.addEventListener('click', function () {
        state.vecU = state.vecU.normalize();
        vecUXInput.value = state.vecU.x.toFixed(1);
        vecUYInput.value = state.vecU.y.toFixed(1);
        render();
      });
    }

    var btnOrthoV = $('btn-orthogonalize-v');
    if (btnOrthoV) {
      btnOrthoV.addEventListener('click', function () {
        var proj = state.vecV.projectOnto(state.vecU);
        state.vecV = state.vecV.sub(proj);
        vecVXInput.value = state.vecV.x.toFixed(1);
        vecVYInput.value = state.vecV.y.toFixed(1);
        render();
      });
    }

    // Matrix multiplication drawer
    multSlider.addEventListener('input', function () {
      state.multT = parseFloat(this.value);
      multTDisplay.textContent = state.multT.toFixed(2);
      render();
    });

    btnSwapMult.addEventListener('click', function () {
      state.multOrder = state.multOrder === 'AB' ? 'BA' : 'AB';
      btnSwapMult.textContent = 'Order: ' + state.multOrder;
      render();
    });

    // Keyboard Shortcuts
    document.addEventListener('keydown', function (e) {
      var tag = document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;

      if (e.key === 'Escape') {
        modalHelp.classList.add('hidden');
        return;
      }
      if (e.key === 'r' || e.key === 'R') { applyPreset('identity'); return; }
      if (e.key === 'c' || e.key === 'C') { $('btn-recenter').click(); return; }
      if (e.key === 'g' || e.key === 'G') { $('btn-toggle-grid').click(); return; }
      if (e.key === '?') { modalHelp.classList.toggle('hidden'); return; }
      if (e.key === '1') { switchModeBtn(0); return; }
      if (e.key === '2') { switchModeBtn(1); return; }
      if (e.key === '3') { switchModeBtn(2); return; }
      if (e.key === '4') { switchModeBtn(3); return; }
    });
  }

  function switchModeBtn(idx) {
    var btns = document.querySelectorAll('.mode-btn');
    if (btns[idx]) btns[idx].click();
  }

  function setMode(newMode) {
    state.mode = newMode;

    // Toggle panels in sidebar
    $('panel-transform').classList.toggle('hidden', newMode !== 'transform');
    $('panel-eigen').classList.toggle('hidden', newMode !== 'eigen');
    $('panel-mult').classList.toggle('hidden', newMode !== 'mult');
    $('panel-vectors').classList.toggle('hidden', newMode !== 'vectors');

    multDrawer.classList.toggle('active', newMode === 'mult');
    render();
  }

  // ── URL State Synchronization ────────────────────────────────────────────

  var hashDebounceTimer = null;
  function updateUrlHash() {
    if (hashDebounceTimer) clearTimeout(hashDebounceTimer);
    hashDebounceTimer = setTimeout(function () {
      var m = state.matrix;
      var hash = 'a=' + m.a.toFixed(2) + '&b=' + m.b.toFixed(2) + '&c=' + m.c.toFixed(2) + '&d=' + m.d.toFixed(2) + '&m=' + state.mode;
      window.location.hash = hash;
    }, 400);
  }

  function readUrlHash() {
    var h = window.location.hash.replace(/^#/, '');
    if (!h) return;
    var params = {};
    h.split('&').forEach(function (part) {
      var pair = part.split('=');
      if (pair.length === 2) params[pair[0]] = pair[1];
    });

    if (params.a && params.b && params.c && params.d) {
      var a = parseFloat(params.a), b = parseFloat(params.b), c = parseFloat(params.c), d = parseFloat(params.d);
      if (!isNaN(a) && !isNaN(b) && !isNaN(c) && !isNaN(d)) {
        state.matrix = new Matrix2x2(a, b, c, d);
      }
    }
    if (params.m && ['transform', 'eigen', 'mult', 'vectors'].indexOf(params.m) !== -1) {
      setMode(params.m);
      document.querySelectorAll('.mode-btn').forEach(function (btn) {
        btn.classList.toggle('active', btn.getAttribute('data-mode') === params.m);
      });
    }
  }

  // ── Initialization ────────────────────────────────────────────────────────

  function init() {
    readUrlHash();
    initEvents();
    syncMatrixInputs();
    updateTelemetry();
    resizeCanvas();
  }

  // ── Public API for Lectern embedding ──────────────────────────────────────

  window.TensorForge = {
    setMatrix: function (a, b, c, d) {
      state.matrix = new Matrix2x2(a, b, c, d);
      syncMatrixInputs();
      updateTelemetry();
      render();
    },
    getMatrix: function () {
      return state.matrix.clone();
    },
    setMode: setMode,
    setShape: function (shapeName) {
      state.shape = shapeName;
      render();
    },
    reset: function () {
      applyPreset('identity');
    }
  };

  init();

})();
