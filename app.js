/**
 * TensorForge — The Ultimate Linear Algebra & Machine Learning Sandbox
 * 100% Client-side, Zero Dependencies.
 *
 * Modules:
 * 1. 2D Linear Transformation & Basis Vectors
 * 2. Analytical Eigensystem & Invariant Line Hunter
 * 3. Matrix Composition & Non-Commutativity Stepper (A × B)
 * 4. Vector Sandbox (Dot Products, Projections, Gram-Schmidt)
 * 5. 3D VectorSpace (3x3 Matrix, Unit Cube, Orbit Camera, 3D Det)
 * 6. LossLab (Interactive 2D Loss Contours, SGD, Momentum, RMSprop, Adam)
 * 7. MicroGraph (Visual Autograd DAG & Backprop Gradient Tracer)
 * 8. Theory Vault (Simplified FY Engineering Lecture Notes)
 * 9. Viva Exam Quiz (Active Recall Preparation with Explanations)
 */

(function () {
  'use strict';

  var Engine = window.TensorForgeEngine;
  var Vector2D = Engine.Vector2D;
  var Matrix2x2 = Engine.Matrix2x2;
  var Vector3D = Engine.Vector3D;
  var Matrix3x3 = Engine.Matrix3x3;
  var project3DTo2D = Engine.project3DTo2D;
  var LossFunctions = Engine.LossFunctions;
  var OptimizerParticle = Engine.OptimizerParticle;

  // ── Application State ─────────────────────────────────────────────────────

  var state = {
    // Mode: 'transform' | 'eigen' | 'mult' | 'vectors' | '3d' | 'loss' | 'autograd' | 'notes' | 'quiz'
    mode: 'transform',
    shape: 'square', // 'square' | 'circle' | 'house' | 'letterF' | 'cloud'

    // 2D Matrices
    matrix: new Matrix2x2(1.5, 0.5, 0.5, 1.2),
    matrixB: new Matrix2x2(0.8, -0.6, 0.6, 0.8),
    multT: 0,
    multOrder: 'AB',

    // 2D Vectors & Probes
    customVec: new Vector2D(1.0, 1.0),
    eigenProbe: new Vector2D(1.0, 0.0),
    vecU: new Vector2D(2.0, 1.0),
    vecV: new Vector2D(1.0, 2.0),

    // 3D Space State
    camYaw: 25,
    camPitch: 20,
    rotX3D: 0,
    rotY3D: 0,
    rotZ3D: 0,
    scaleX3D: 1.0,
    scaleY3D: 1.0,
    scaleZ3D: 1.0,
    mesh3D: 'cube',

    // LossLab State
    lossKey: 'bowl',
    lossRunning: false,
    learningRate: 0.05,
    particles: [],

    // MicroGraph Autograd State
    autogradPreset: 'neuron',
    autogradStep: 'idle', // 'idle' | 'forward' | 'backward'

    // Display Toggles
    showCustomVec: true,
    showTransformedGrid: true,
    showEigenSpans: true,

    // Viewport Transform
    scale: 65,
    panX: 0,
    panY: 0,

    // Drag Interaction
    draggingTarget: null,
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

  // Matrix Inputs
  var matAInput = $('mat-a');
  var matBInput = $('mat-b');
  var matCInput = $('mat-c');
  var matDInput = $('mat-d');

  var matBAInput = $('mat-b-a');
  var matBBInput = $('mat-b-b');
  var matBCInput = $('mat-b-c');
  var matBDInput = $('mat-b-d');

  // Telemetry Readouts
  var readoutDet = $('telemetry-det');
  var badgeDet = $('badge-det');
  var readoutTrace = $('telemetry-trace');
  var readoutRank = $('telemetry-rank');
  var eigenRow1 = $('eigen-val-1');
  var eigenRow2 = $('eigen-val-2');

  // Eigen Mode Readouts
  var eigenFormulaSub = $('eigen-formula-sub');
  var eigenQuadExpanded = $('eigen-quad-expanded');
  var eigenDiscVal = $('eigen-disc-val');
  var badgeDisc = $('badge-disc');
  var valCollinearity = $('val-collinearity');
  var barCollinearity = $('bar-collinearity');
  var eigenFoundAlert = $('eigen-found-alert');

  // Matrix Chain (Mode 3)
  var multDrawer = $('mult-drawer');
  var multSlider = $('slider-mult');
  var multTDisplay = $('val-mult-t');
  var btnSwapMult = $('btn-swap-mult');
  var valProdAB = $('val-prod-ab');
  var valProdBA = $('val-prod-ba');

  // Vector Sandbox (Mode 4)
  var vecUXInput = $('vec-u-x');
  var vecUYInput = $('vec-u-y');
  var vecVXInput = $('vec-v-x');
  var vecVYInput = $('vec-v-y');
  var valDotProduct = $('val-dot-product');
  var badgeDotAngle = $('badge-dot-angle');
  var valVectorAngle = $('val-vector-angle');
  var valMagU = $('val-mag-u');
  var valMagV = $('val-mag-v');

  // 3D Elements
  var sliderYaw3D = $('slider-yaw-3d');
  var valYaw3D = $('val-yaw-3d');
  var sliderPitch3D = $('slider-pitch-3d');
  var valPitch3D = $('val-pitch-3d');
  var sliderRoll3D = $('slider-roll-3d');
  var valRoll3D = $('val-roll-3d');
  var sliderScaleX = $('slider-scale-x');
  var sliderScaleY = $('slider-scale-y');
  var sliderScaleZ = $('slider-scale-z');
  var valDet3D = $('val-det-3d');
  var badgeDet3D = $('badge-det-3d');

  // LossLab Elements
  var sliderLR = $('slider-lr');
  var valLR = $('val-lr');
  var btnStartDescent = $('btn-start-descent');
  var btnResetDescent = $('btn-reset-descent');
  var lossValSGD = $('loss-val-sgd');
  var lossValMom = $('loss-val-mom');
  var lossValRMS = $('loss-val-rms');
  var lossValAdam = $('loss-val-adam');

  // Autograd Elements
  var btnForwardPass = $('btn-forward-pass');
  var btnBackwardPass = $('btn-backward-pass');
  var autogradStatusText = $('autograd-status-text');

  // Sliders & HUD
  var rotationSlider = $('slider-rotation');
  var rotationValue = $('val-rotation');
  var shearSlider = $('slider-shear');
  var shearValue = $('val-shear');
  var hudCoords = $('hud-coords');
  var modalHelp = $('modal-help');

  // ── Canvas Setup & DPI Scaling ────────────────────────────────────────────

  var dpr = window.devicePixelRatio || 1;
  var viewWidth = 0;
  var viewHeight = 0;

  function resizeCanvas() {
    var rect = canvas.parentElement.getBoundingClientRect();
    viewWidth = rect.width;
    viewHeight = rect.height;

    canvas.width = viewWidth * dpr;
    canvas.height = viewHeight * dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    render();
  }

  // ── Coordinate Conversions (2D Viewport) ──────────────────────────────────

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

  // ── Master Render Pipeline ────────────────────────────────────────────────

  function render() {
    ctx.clearRect(0, 0, viewWidth, viewHeight);

    if (state.mode === '3d') {
      render3DSpace();
    } else if (state.mode === 'loss') {
      renderLossLab();
    } else if (state.mode === 'autograd') {
      renderAutogradGraph();
    } else if (state.mode === 'notes') {
      renderNotesCanvas();
    } else if (state.mode === 'quiz') {
      renderQuizCanvas();
    } else if (state.mode === 'vectors') {
      drawVectorSandbox();
    } else {
      // 2D Transformation & Eigen Analysis
      var activeMatrix = getActiveMatrixForRender();
      drawBackgroundGrid();

      if (state.showTransformedGrid) {
        drawTransformedGrid(activeMatrix);
      }

      drawTransformedShape(activeMatrix);

      if (Math.abs(activeMatrix.determinant()) < Engine.EPSILON) {
        drawRankNullityLines(activeMatrix);
      }

      if (state.showEigenSpans && state.mode !== 'mult') {
        drawEigenSpanLines(activeMatrix);
      }

      drawAxes();
      drawBasisVectors(activeMatrix);

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
        return Matrix2x2.lerp(Matrix2x2.identity(), M1, t * 2);
      } else {
        return Matrix2x2.lerp(M1, product, (t - 0.5) * 2);
      }
    }
    return state.matrix;
  }

  // ── 2D Standard Background Grid ───────────────────────────────────────────

  function drawBackgroundGrid() {
    var min = screenToWorld(0, viewHeight);
    var max = screenToWorld(viewWidth, 0);

    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.beginPath();

    for (var x = Math.floor(min.x); x <= Math.ceil(max.x); x++) {
      var p1 = worldToScreen(x, min.y);
      var p2 = worldToScreen(x, max.y);
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
    }
    for (var y = Math.floor(min.y); y <= Math.ceil(max.y); y++) {
      var q1 = worldToScreen(min.x, y);
      var q2 = worldToScreen(max.x, y);
      ctx.moveTo(q1.x, q1.y);
      ctx.lineTo(q2.x, q2.y);
    }
    ctx.stroke();
  }

  // ── 2D Transformed Coordinate Grid ────────────────────────────────────────

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
      var qStart = worldToScreen(base.x - col1.x * range, base.y - col1.y * range);
      var qEnd = worldToScreen(base.x + col1.x * range, base.y + col1.y * range);
      ctx.moveTo(qStart.x, qStart.y);
      ctx.lineTo(qEnd.x, qEnd.y);
    }
    ctx.stroke();
  }

  // ── 2D Shapes (Square, Circle/SVD, House, F, PCA Cloud) ───────────────────

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

      if (Math.abs(det) > 0.05) {
        var cx = (o.x + iPos.x + sum.x + jPos.x) / 4;
        var cy = (o.y + iPos.y + sum.y + jPos.y) / 4;
        ctx.font = '600 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
        ctx.fillStyle = det > 0 ? '#67e8f9' : '#fcd34d';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Area: ' + Math.abs(det).toFixed(2), cx, cy);
      }

    } else if (state.shape === 'circle') {
      var steps = 64;
      ctx.beginPath();
      for (var s = 0; s <= steps; s++) {
        var theta = (s / steps) * Math.PI * 2;
        var tw = matrix.apply(new Vector2D(Math.cos(theta), Math.sin(theta)));
        var sp = worldToScreen(tw.x, tw.y);
        if (s === 0) ctx.moveTo(sp.x, sp.y);
        else ctx.lineTo(sp.x, sp.y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      var svd = Engine.computeSVD2x2(matrix);
      drawTopRightHUDTag('SVD Singular Values: σ₁=' + svd.sigma1.toFixed(2) + ', σ₂=' + svd.sigma2.toFixed(2));

    } else if (state.shape === 'house') {
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
      ctx.fillStyle = det >= 0 ? 'rgba(99, 102, 241, 0.7)' : 'rgba(245, 158, 11, 0.7)';
      scatterPoints.forEach(function (pt) {
        var t = matrix.apply(pt);
        var sp = worldToScreen(t.x, t.y);
        ctx.beginPath();
        ctx.arc(sp.x, sp.y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
      drawTopRightHUDTag('Gaussian Data Scatter Cloud (PCA Covariance Mapping)');
    }

    ctx.restore();
  }

  function drawTopRightHUDTag(text) {
    ctx.save();
    ctx.font = '600 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
    ctx.fillStyle = '#c7d2fe';
    ctx.textAlign = 'right';
    ctx.fillText(text, viewWidth - 24, 32);
    ctx.restore();
  }

  // ── Invariant Eigenvector Span Lines ──────────────────────────────────────

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

  // ── Rank-Nullity Theorem Lines (when det = 0) ─────────────────────────────

  function drawRankNullityLines(matrix) {
    var nullVec = matrix.nullspace();
    var colVec = matrix.columnspace();
    var lineLength = 25;

    ctx.save();
    // Nullspace (ker A) - dashed red line: vectors on this line are compressed to zero
    if (nullVec) {
      var p1 = worldToScreen(-nullVec.x * lineLength, -nullVec.y * lineLength);
      var p2 = worldToScreen(nullVec.x * lineLength, nullVec.y * lineLength);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      var labelPos = worldToScreen(nullVec.x * 2.5, nullVec.y * 2.5);
      ctx.fillStyle = '#f87171';
      ctx.font = '700 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillText('Kernel / Nullspace (Ax = 0)', labelPos.x + 8, labelPos.y - 8);
    }

    // Column Space (im A) - solid blue line
    if (colVec && colVec.magnitudeSq() > Engine.EPSILON) {
      var q1 = worldToScreen(-colVec.x * lineLength, -colVec.y * lineLength);
      var q2 = worldToScreen(colVec.x * lineLength, colVec.y * lineLength);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(q1.x, q1.y);
      ctx.lineTo(q2.x, q2.y);
      ctx.stroke();

      var colLabelPos = worldToScreen(colVec.x * 3.5, colVec.y * 3.5);
      ctx.fillStyle = '#38bdf8';
      ctx.font = '700 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillText('Column Space / Range (All outputs)', colLabelPos.x + 8, colLabelPos.y + 16);
    }
    ctx.restore();
  }

  // ── Eigen Hunter Probe (Mode 2) ───────────────────────────────────────────

  function drawEigenHunter(matrix) {
    var o = worldToScreen(0, 0);

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(o.x, o.y, state.scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    var pX = worldToScreen(state.eigenProbe.x, state.eigenProbe.y);
    drawArrow(o.x, o.y, pX.x, pX.y, '#f59e0b', 2.5);
    drawVectorHandle(pX.x, pX.y, '#f59e0b', 'probe', state.hoverTarget === 'probe');
    drawVectorLabel('x', pX.x, pX.y, '#f59e0b', state.eigenProbe.angle());

    var tProbe = matrix.apply(state.eigenProbe);
    var pAx = worldToScreen(tProbe.x, tProbe.y);
    drawArrow(o.x, o.y, pAx.x, pAx.y, '#38bdf8', 2);
    drawVectorLabel('Ax', pAx.x, pAx.y, '#38bdf8', tProbe.angle());

    var magT = tProbe.magnitude();
    var cosAngle = magT > Engine.EPSILON ? Math.abs(state.eigenProbe.dot(tProbe) / magT) : 1;
    var matchPct = Math.round(cosAngle * 100);

    valCollinearity.textContent = matchPct + '%';
    barCollinearity.style.width = matchPct + '%';

    var isCollinear = cosAngle > 0.99;
    eigenFoundAlert.classList.toggle('hidden', !isCollinear);
  }

  // ── 2D Axes & Basis Vectors ───────────────────────────────────────────────

  function drawAxes() {
    var o = worldToScreen(0, 0);

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';

    ctx.beginPath();
    ctx.moveTo(0, o.y); ctx.lineTo(viewWidth, o.y);
    ctx.moveTo(o.x, 0); ctx.lineTo(o.x, viewHeight);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.beginPath();
    ctx.arc(o.x, o.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawBasisVectors(matrix) {
    var o = worldToScreen(0, 0);
    var iPos = worldToScreen(matrix.a, matrix.c);
    var jPos = worldToScreen(matrix.b, matrix.d);

    var angI = Math.atan2(matrix.c, matrix.a);
    var angJ = Math.atan2(matrix.d, matrix.b);

    drawArrow(o.x, o.y, iPos.x, iPos.y, '#f43f5e', 2.5);
    drawVectorHandle(iPos.x, iPos.y, '#f43f5e', 'i', state.hoverTarget === 'i');
    drawVectorLabel('î [' + matrix.a.toFixed(1) + ', ' + matrix.c.toFixed(1) + ']', iPos.x, iPos.y, '#f43f5e', angI);

    drawArrow(o.x, o.y, jPos.x, jPos.y, '#06b6d4', 2.5);
    drawVectorHandle(jPos.x, jPos.y, '#06b6d4', 'j', state.hoverTarget === 'j');
    drawVectorLabel('ĵ [' + matrix.b.toFixed(1) + ', ' + matrix.d.toFixed(1) + ']', jPos.x, jPos.y, '#06b6d4', angJ);
  }

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
    drawVectorLabel('T(v) [' + transformed.x.toFixed(1) + ', ' + transformed.y.toFixed(1) + ']', vOutput.x, vOutput.y, '#10b981', transformed.angle());
  }

  // ── Vector Sandbox (Mode 4) ───────────────────────────────────────────────

  function drawVectorSandbox() {
    var o = worldToScreen(0, 0);
    var uPos = worldToScreen(state.vecU.x, state.vecU.y);
    var vPos = worldToScreen(state.vecV.x, state.vecV.y);

    drawAxes();

    drawArrow(o.x, o.y, uPos.x, uPos.y, '#f43f5e', 2.5);
    drawVectorHandle(uPos.x, uPos.y, '#f43f5e', 'u', state.hoverTarget === 'u');
    drawVectorLabel('u [' + state.vecU.x.toFixed(1) + ', ' + state.vecU.y.toFixed(1) + ']', uPos.x, uPos.y, '#f43f5e', state.vecU.angle());

    drawArrow(o.x, o.y, vPos.x, vPos.y, '#06b6d4', 2.5);
    drawVectorHandle(vPos.x, vPos.y, '#06b6d4', 'v_sandbox', state.hoverTarget === 'v_sandbox');
    drawVectorLabel('v [' + state.vecV.x.toFixed(1) + ', ' + state.vecV.y.toFixed(1) + ']', vPos.x, vPos.y, '#06b6d4', state.vecV.angle());

    var sum = state.vecU.add(state.vecV);
    var sumPos = worldToScreen(sum.x, sum.y);

    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.beginPath();
    ctx.moveTo(uPos.x, uPos.y); ctx.lineTo(sumPos.x, sumPos.y); ctx.lineTo(vPos.x, vPos.y);
    ctx.stroke();
    ctx.restore();

    drawArrow(o.x, o.y, sumPos.x, sumPos.y, '#10b981', 2);
    drawVectorLabel('u+v [' + sum.x.toFixed(1) + ', ' + sum.y.toFixed(1) + ']', sumPos.x, sumPos.y, '#10b981', sum.angle());

    var proj = state.vecU.projectOnto(state.vecV);
    var projPos = worldToScreen(proj.x, proj.y);

    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = '#f59e0b';
    ctx.beginPath();
    ctx.moveTo(uPos.x, uPos.y); ctx.lineTo(projPos.x, projPos.y);
    ctx.stroke();
    drawArrow(o.x, o.y, projPos.x, projPos.y, '#f59e0b', 3);
    ctx.restore();

    var angleU = state.vecU.angle();
    var angleV = state.vecV.angle();
    var diff = angleV - angleU;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    var arcRad = 28;

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(o.x, o.y, arcRad, -angleU, -(angleU + diff), diff < 0);
    ctx.stroke();
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

  // ── 3D VectorSpace Renderer (Mode 5) ──────────────────────────────────────

  function render3DSpace() {
    var yawRad = (state.camYaw * Math.PI) / 180;
    var pitchRad = (state.camPitch * Math.PI) / 180;
    var fov = 750;
    var originX = viewWidth / 2;
    var originY = viewHeight / 2;

    // Build 3D Transformation Matrix A
    var rotX = Matrix3x3.rotationX((state.rotX3D * Math.PI) / 180);
    var rotY = Matrix3x3.rotationY((state.rotY3D * Math.PI) / 180);
    var rotZ = Matrix3x3.rotationZ((state.rotZ3D * Math.PI) / 180);
    var scaleM = new Matrix3x3([
      state.scaleX3D, 0, 0,
      0, state.scaleY3D, 0,
      0, 0, state.scaleZ3D
    ]);

    var transform3D = rotZ.multiply(rotY).multiply(rotX).multiply(scaleM);

    // Compute 3D Volume (Determinant)
    var det3D = transform3D.determinant();
    valDet3D.textContent = det3D.toFixed(2);
    if (Math.abs(det3D) < 0.01) {
      badgeDet3D.textContent = 'Planar Collapse (det=0)';
      badgeDet3D.className = 'telemetry-badge badge-det-zero';
    } else {
      badgeDet3D.textContent = 'Volume: ' + Math.abs(det3D).toFixed(2);
      badgeDet3D.className = 'telemetry-badge badge-det-pos';
    }

    // 1. Draw 3D Axes
    var axes3D = [
      { v: new Vector3D(2.5, 0, 0), col: '#f43f5e', name: 'X (î)' },
      { v: new Vector3D(0, 2.5, 0), col: '#10b981', name: 'Y (ĵ)' },
      { v: new Vector3D(0, 0, 2.5), col: '#06b6d4', name: 'Z (k̂)' }
    ];

    var o2D = project3DTo2D(new Vector3D(0, 0, 0), pitchRad, yawRad, fov);
    axes3D.forEach(function (ax) {
      var p = project3DTo2D(ax.v, pitchRad, yawRad, fov);
      ctx.beginPath();
      ctx.strokeStyle = ax.col;
      ctx.lineWidth = 2;
      ctx.moveTo(originX + o2D.x, originY + o2D.y);
      ctx.lineTo(originX + p.x, originY + p.y);
      ctx.stroke();

      ctx.font = '700 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillStyle = ax.col;
      ctx.fillText(ax.name, originX + p.x + 8, originY + p.y - 4);
    });

    // 2. Render 3D Mesh (Cube)
    var rawCubeVertices = [
      new Vector3D(-1, -1, -1), new Vector3D(1, -1, -1),
      new Vector3D(1, 1, -1),   new Vector3D(-1, 1, -1),
      new Vector3D(-1, -1, 1),  new Vector3D(1, -1, 1),
      new Vector3D(1, 1, 1),    new Vector3D(-1, 1, 1)
    ];

    var cubeFaces = [
      [0, 1, 2, 3], // Back
      [4, 5, 6, 7], // Front
      [0, 1, 5, 4], // Bottom
      [2, 3, 7, 6], // Top
      [0, 3, 7, 4], // Left
      [1, 2, 6, 5]  // Right
    ];

    // Transform vertices
    var transformedPts = rawCubeVertices.map(function (v) {
      return transform3D.apply(v);
    });

    // Project and calculate depth
    var projectedPts = transformedPts.map(function (v) {
      var proj = project3DTo2D(v, pitchRad, yawRad, fov);
      return { x: originX + proj.x, y: originY + proj.y, depth: proj.depth };
    });

    // Sort faces by depth for painter's algorithm
    var faceList = cubeFaces.map(function (faceIndices) {
      var avgZ = (projectedPts[faceIndices[0]].depth + projectedPts[faceIndices[1]].depth + projectedPts[faceIndices[2]].depth + projectedPts[faceIndices[3]].depth) / 4;
      return { indices: faceIndices, depth: avgZ };
    });
    faceList.sort(function (a, b) { return a.depth - b.depth; });

    // Draw shaded faces
    faceList.forEach(function (f) {
      var p0 = projectedPts[f.indices[0]];
      var p1 = projectedPts[f.indices[1]];
      var p2 = projectedPts[f.indices[2]];
      var p3 = projectedPts[f.indices[3]];

      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.lineTo(p3.x, p3.y);
      ctx.closePath();

      ctx.fillStyle = det3D >= 0 ? 'rgba(99, 102, 241, 0.22)' : 'rgba(245, 158, 11, 0.24)';
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });

    drawTopRightHUDTag('3D Space: Drag mouse to Orbit Camera (Yaw: ' + state.camYaw + '°, Pitch: ' + state.camPitch + '°)');
  }

  // ── LossLab Optimization Sandbox (Mode 6) ─────────────────────────────────

  function initLossParticles() {
    var startX = state.lossKey === 'rosenbrock' ? -1.2 : -2.0;
    var startY = state.lossKey === 'rosenbrock' ? 1.5 : 2.0;

    state.particles = [
      new OptimizerParticle('sgd', startX, startY, '#ef4444'),
      new OptimizerParticle('momentum', startX, startY, '#f59e0b'),
      new OptimizerParticle('rmsprop', startX, startY, '#06b6d4'),
      new OptimizerParticle('adam', startX, startY, '#10b981')
    ];
  }

  function renderLossLab() {
    var lossFn = LossFunctions[state.lossKey] || LossFunctions.bowl;
    var domain = lossFn.domain;

    // Draw Contour Lines on canvas
    var gridN = 45;
    var stepW = viewWidth / gridN;
    var stepH = viewHeight / gridN;

    ctx.save();
    for (var i = 0; i < gridN; i += 2) {
      for (var j = 0; j < gridN; j += 2) {
        var wx = ((i / gridN) - 0.5) * domain * 2;
        var wy = -((j / gridN) - 0.5) * domain * 2;
        var z = lossFn.evaluate(wx, wy);

        // Normalize color
        var intensity = Math.min(1, Math.max(0, Math.log(Math.abs(z) + 1) / 3));
        ctx.fillStyle = 'rgba(' + Math.round(intensity * 120 + 20) + ', 40, ' + Math.round((1 - intensity) * 140 + 40) + ', 0.18)';
        ctx.fillRect(i * stepW, j * stepH, stepW * 2, stepH * 2);
      }
    }
    ctx.restore();

    drawAxes();

    // If running, step optimizers
    if (state.lossRunning) {
      state.particles.forEach(function (p) {
        p.step(lossFn, state.learningRate);
      });
    }

    // Draw Particles & Trajectories
    state.particles.forEach(function (p) {
      // Trajectory line
      if (p.history.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = p.color;
        ctx.lineWidth = 2;
        p.history.forEach(function (pt, idx) {
          var sp = worldToScreen(pt.x, pt.y);
          if (idx === 0) ctx.moveTo(sp.x, sp.y);
          else ctx.lineTo(sp.x, sp.y);
        });
        ctx.stroke();
      }

      // Ball
      var curSp = worldToScreen(p.x, p.y);
      ctx.beginPath();
      ctx.fillStyle = p.color;
      ctx.arc(curSp.x, curSp.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Update Telemetry Values
    if (state.particles.length >= 4) {
      lossValSGD.textContent = lossFn.evaluate(state.particles[0].x, state.particles[0].y).toFixed(3);
      lossValMom.textContent = lossFn.evaluate(state.particles[1].x, state.particles[1].y).toFixed(3);
      lossValRMS.textContent = lossFn.evaluate(state.particles[2].x, state.particles[2].y).toFixed(3);
      lossValAdam.textContent = lossFn.evaluate(state.particles[3].x, state.particles[3].y).toFixed(3);
    }

    drawTopRightHUDTag('LossLab: ' + lossFn.name + ' (LR: ' + state.learningRate + ')');

    if (state.lossRunning) {
      requestAnimationFrame(render);
    }
  }

  // ── MicroGraph Autograd DAG Renderer (Mode 7) ─────────────────────────────

  function renderAutogradGraph() {
    var nodes = [];

    if (state.autogradPreset === 'neuron') {
      // y = ReLU(w1*x1 + w2*x2 + b)
      var x1 = 1.5, w1 = 0.8;
      var x2 = -1.0, w2 = 1.2;
      var b = 0.3;

      var p1 = x1 * w1; // 1.2
      var p2 = x2 * w2; // -1.2
      var sum = p1 + p2 + b; // 0.3
      var reluOut = sum > 0 ? sum : 0; // 0.3

      // Backward gradients
      var dRelu = 1.0;
      var dSum = reluOut > 0 ? 1.0 : 0.0;
      var dW1 = dSum * x1;
      var dW2 = dSum * x2;

      nodes = [
        { label: 'x₁', val: x1, grad: 0.0, x: 120, y: 160, type: 'in' },
        { label: 'w₁', val: w1, grad: dW1, x: 120, y: 240, type: 'param' },
        { label: 'x₂', val: x2, grad: 0.0, x: 120, y: 340, type: 'in' },
        { label: 'w₂', val: w2, grad: dW2, x: 120, y: 420, type: 'param' },
        { label: 'b', val: b, grad: dSum, x: 260, y: 480, type: 'param' },
        { label: 'w₁·x₁', val: p1, grad: dSum, x: 260, y: 200, type: 'op' },
        { label: 'w₂·x₂', val: p2, grad: dSum, x: 260, y: 380, type: 'op' },
        { label: 'Σ (+)', val: sum, grad: dSum, x: 420, y: 290, type: 'op' },
        { label: 'ReLU', val: reluOut, grad: dRelu, x: 580, y: 290, type: 'out' }
      ];
    } else {
      // MSE Loss: L = (y_pred - y_true)^2
      nodes = [
        { label: 'y_pred', val: 0.85, grad: 0.70, x: 160, y: 220, type: 'in' },
        { label: 'y_true', val: 0.50, grad: -0.70, x: 160, y: 360, type: 'in' },
        { label: 'diff (-)', val: 0.35, grad: 0.70, x: 340, y: 290, type: 'op' },
        { label: 'MSE (²)', val: 0.12, grad: 1.00, x: 520, y: 290, type: 'out' }
      ];
    }

    // Draw Connectors
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1.5;
    for (var i = 0; i < nodes.length - 1; i++) {
      for (var j = i + 1; j < nodes.length; j++) {
        if (nodes[j].x > nodes[i].x && nodes[j].x - nodes[i].x < 200) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x + 45, nodes[i].y);
          ctx.lineTo(nodes[j].x - 45, nodes[j].y);
          ctx.stroke();
        }
      }
    }
    ctx.restore();

    // Draw Nodes
    nodes.forEach(function (n) {
      ctx.save();
      var isForward = state.autogradStep === 'forward' || state.autogradStep === 'backward';
      var isBackward = state.autogradStep === 'backward';

      ctx.fillStyle = n.type === 'out' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(15, 23, 42, 0.85)';
      ctx.strokeStyle = isBackward && n.type === 'param' ? '#f59e0b' : 'rgba(99, 102, 241, 0.4)';
      ctx.lineWidth = 2;

      // Rounded box
      var bw = 85, bh = 50;
      ctx.beginPath();
      ctx.roundRect(n.x - bw / 2, n.y - bh / 2, bw, bh, 8);
      ctx.fill();
      ctx.stroke();

      // Label & Value
      ctx.font = '700 12px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillStyle = '#f8fafc';
      ctx.textAlign = 'center';
      ctx.fillText(n.label, n.x, n.y - 6);

      ctx.font = '600 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillStyle = isForward ? '#38bdf8' : '#94a3b8';
      ctx.fillText('v: ' + n.val.toFixed(2), n.x, n.y + 8);

      if (isBackward) {
        ctx.fillStyle = '#f59e0b';
        ctx.fillText('∇: ' + n.grad.toFixed(2), n.x, n.y + 20);
      }
      ctx.restore();
    });

    drawTopRightHUDTag('MicroGraph Autograd: Click Forward Pass then Backprop to see chain rule!');
  }

  // ── Notes & Quiz Canvas Placeholders ──────────────────────────────────────

  function renderNotesCanvas() {
    drawTopRightHUDTag('Theory Vault: Select any lecture topic on the left sidebar!');
  }

  function renderQuizCanvas() {
    drawTopRightHUDTag('Viva Quiz: Answer the viva practice questions on the left!');
  }

  // ── Vector Helpers & Primitive Drawing ────────────────────────────────────

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
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
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

  function drawVectorLabel(text, x, y, color, angle) {
    ctx.save();
    ctx.font = '600 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
    ctx.fillStyle = color;

    var rad = typeof angle === 'number' ? angle : 0;
    var offX = Math.cos(rad) * 14;
    var offY = -Math.sin(rad) * 14;

    ctx.textAlign = offX >= 0 ? 'left' : 'right';
    ctx.textBaseline = offY <= 0 ? 'bottom' : 'top';
    ctx.fillText(text, x + (offX >= 0 ? 8 : -8), y + (offY <= 0 ? -6 : 6));
    ctx.restore();
  }

  // ── Math Telemetry Updates ────────────────────────────────────────────────

  function updateTelemetry() {
    var m = state.matrix;
    var det = m.determinant();
    var tr = m.trace();
    var rk = m.rank();

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

    readoutTrace.textContent = tr.toFixed(2);
    readoutRank.textContent = rk;

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

    updateMultComparison();

    var nullCard = $('nullspace-card');
    if (nullCard) {
      var isSingular = Math.abs(det) < Engine.EPSILON;
      nullCard.classList.toggle('hidden', !isSingular);
      if (isSingular) {
        var nVec = m.nullspace();
        var cVec = m.columnspace();
        var nText = nVec ? '[' + nVec.x.toFixed(2) + ', ' + nVec.y.toFixed(2) + ']ᵀ' : '[0, 0]ᵀ';
        var cText = cVec ? '[' + cVec.x.toFixed(2) + ', ' + cVec.y.toFixed(2) + ']ᵀ' : '[0, 0]ᵀ';
        $('nullspace-basis-text').innerHTML = '<strong>Kernel ker(A):</strong> Basis = ' + nText + ' (compressed to origin)';
        $('columnspace-basis-text').innerHTML = '<strong>Range im(A):</strong> Basis = ' + cText + ' (1D line)';
      }
    }

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
    state.matrix = new Matrix2x2(
      parseFloat(matAInput.value) || 0,
      parseFloat(matBInput.value) || 0,
      parseFloat(matCInput.value) || 0,
      parseFloat(matDInput.value) || 0
    );
    updateTelemetry();
    render();
  }

  function readMatrixBInputs() {
    state.matrixB = new Matrix2x2(
      parseFloat(matBAInput.value) || 0,
      parseFloat(matBBInput.value) || 0,
      parseFloat(matBCInput.value) || 0,
      parseFloat(matBDInput.value) || 0
    );
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

  // ── Drag & Touch Interactions ─────────────────────────────────────────────

  function getMousePos(e) {
    var rect = canvas.getBoundingClientRect();
    var clientX = e.clientX;
    var clientY = e.clientY;
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }
    return { x: clientX - rect.left, y: clientY - rect.top };
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

    if (state.mode === '3d') {
      state.draggingTarget = '3d_orbit';
      state.dragStartMouse = pos;
      return;
    }

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

    if (state.draggingTarget === '3d_orbit') {
      var dx = pos.x - state.dragStartMouse.x;
      var dy = pos.y - state.dragStartMouse.y;
      state.camYaw += dx * 0.5;
      state.camPitch = Math.max(-85, Math.min(85, state.camPitch - dy * 0.5));
      state.dragStartMouse = pos;
      sliderYaw3D.value = Math.round(state.camYaw);
      valYaw3D.textContent = Math.round(state.camYaw) + '°';
      sliderPitch3D.value = Math.round(state.camPitch);
      valPitch3D.textContent = Math.round(state.camPitch) + '°';
      render();
      return;
    }

    var snap = e.shiftKey ? 0.25 : 0.05;
    var sx = Math.round(world.x / snap) * snap;
    var sy = Math.round(world.y / snap) * snap;

    if (state.draggingTarget === 'i') {
      state.matrix.a = sx; state.matrix.c = sy;
      syncMatrixInputs(); updateTelemetry(); render();
    } else if (state.draggingTarget === 'j') {
      state.matrix.b = sx; state.matrix.d = sy;
      syncMatrixInputs(); updateTelemetry(); render();
    } else if (state.draggingTarget === 'v') {
      state.customVec.x = sx; state.customVec.y = sy; render();
    } else if (state.draggingTarget === 'probe') {
      var ang = Math.atan2(world.y, world.x);
      state.eigenProbe = new Vector2D(Math.cos(ang), Math.sin(ang)); render();
    } else if (state.draggingTarget === 'u') {
      state.vecU.x = sx; state.vecU.y = sy;
      vecUXInput.value = sx.toFixed(1); vecUYInput.value = sy.toFixed(1); render();
    } else if (state.draggingTarget === 'v_sandbox') {
      state.vecV.x = sx; state.vecV.y = sy;
      vecVXInput.value = sx.toFixed(1); vecVYInput.value = sy.toFixed(1); render();
    } else if (state.draggingTarget === 'pan') {
      state.panX = state.dragStartPan.x + (pos.x - state.dragStartMouse.x);
      state.panY = state.dragStartPan.y + (pos.y - state.dragStartMouse.y);
      render();
    }
  }

  function onPointerUp() {
    state.draggingTarget = null;
    canvas.style.cursor = state.hoverTarget ? 'grab' : 'crosshair';
  }

  canvas.addEventListener('mousedown', onPointerDown);
  window.addEventListener('mousemove', onPointerMove);
  window.addEventListener('mouseup', onPointerUp);

  canvas.addEventListener('touchstart', function (e) {
    e.preventDefault(); onPointerDown(e);
  }, { passive: false });

  window.addEventListener('touchmove', function (e) {
    if (state.draggingTarget) e.preventDefault();
    onPointerMove(e);
  }, { passive: false });

  window.addEventListener('touchend', onPointerUp);

  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    var zFactor = e.deltaY < 0 ? 1.08 : 0.92;
    state.scale = Math.min(180, Math.max(25, state.scale * zFactor));
    render();
  }, { passive: false });

  // ── Theory Vault Content Engine ───────────────────────────────────────────

  var LECTURE_NOTES = {
    'matrix-transform': {
      title: '1. Matrices as Space Transformations',
      html: `
        <h4>Core Intuition</h4>
        <p>A matrix is not just a spreadsheet of numbers—it is a <strong>spatial function</strong> that takes any vector in $\\mathbb{R}^2$ and maps it to a new location while preserving two rules:</p>
        <ul>
          <li>The origin $(0, 0)$ remains fixed.</li>
          <li>All grid lines remain straight, parallel, and evenly spaced.</li>
        </ul>
        <div class="theory-highlight">The columns of A tell you where the standard basis vectors land:<br>Col 1 = î destination [a, c]ᵀ<br>Col 2 = ĵ destination [b, d]ᵀ</div>
        <h4>Real-world AI Application</h4>
        <p>In neural networks, fully connected layers compute $y = Wx + b$. The weight matrix $W$ rotates, shears, and stretches the feature space to make data linearly separable!</p>
        <div class="viva-tip">🎓 Viva Question: "If a matrix turns basis vector î into [2, 0] and ĵ into [0, 3], what is the matrix?"<br>Answer: [2, 0; 0, 3] (a non-uniform scaling matrix).</div>
      `
    },
    'determinant': {
      title: '2. The Determinant & Volume Scaling',
      html: `
        <h4>Geometric Meaning</h4>
        <p>The determinant $\\det(A) = ad - bc$ measures the factor by which areas (in 2D) or volumes (in 3D) are multiplied when space is transformed.</p>
        <ul>
          <li><strong>det(A) > 0:</strong> Space scales by |det| and orientation (chirality) is preserved.</li>
          <li><strong>det(A) < 0:</strong> Space is flipped or mirrored (like turning a sheet of paper upside down).</li>
          <li><strong>det(A) = 0:</strong> Space is squished into a lower dimension (2D plane collapses into a 1D line or point). Information is lost forever—the matrix has <strong>no inverse</strong>!</li>
        </ul>
        <div class="theory-highlight">Area(A·Shape) = |det(A)| · Area(Original Shape)</div>
        <div class="viva-tip">🎓 Viva Question: "Why does det(A) = 0 imply non-invertibility?"<br>Answer: Because multiple distinct points are mapped to the exact same output, making it impossible to uniquely reverse the operation.</div>
      `
    },
    'eigenvalues': {
      title: '3. Eigenvalues, Eigenvectors & Resonance',
      html: `
        <h4>What are Eigenvectors?</h4>
        <p>Most vectors get knocked off their span line when transformed by $A$. An <strong>eigenvector</strong> is a special, rare vector that does NOT rotate—it only scales along its original line!</p>
        <div class="theory-highlight">Ax = λx  (where λ is the eigenvalue scalar)</div>
        <ul>
          <li><strong>λ > 1:</strong> Vector stretches outwards.</li>
          <li><strong>0 < λ < 1:</strong> Vector contracts inwards.</li>
          <li><strong>λ < 0:</strong> Vector flips to the opposite direction along the same span.</li>
          <li><strong>Complex λ:</strong> Pure rotation or spiral—no real vectors stay on their line!</li>
        </ul>
        <div class="viva-tip">🎓 Viva Question: "How do you find eigenvalues mathematically?"<br>Answer: By setting $\\det(A - \\lambda I) = 0$ (the characteristic equation) and solving for roots $\\lambda$.</div>
      `
    },
    'diagonalization': {
      title: '4. Diagonalization & Matrix Powers (A = PDP⁻¹)',
      html: `
        <h4>Why Diagonalize?</h4>
        <p>If a matrix has $n$ linearly independent eigenvectors, we can change our coordinate system to the <strong>eigenbasis</strong>. In this coordinate system, matrix operations become completely uncoupled!</p>
        <div class="theory-highlight">A = P · D · P⁻¹  (where D is diagonal with eigenvalues)</div>
        <p>Computing $A^{100}$ directly requires 100 expensive matrix multiplications. With diagonalization, it takes 1 step:</p>
        <div class="theory-highlight">Aᵏ = P · Dᵏ · P⁻¹ = P · diag(λ₁ᵏ, λ₂ᵏ) · P⁻¹</div>
        <div class="viva-tip">🎓 Viva Question: "When is an n×n matrix diagonalizable?"<br>Answer: When it has $n$ linearly independent eigenvectors (always true if it has $n$ distinct eigenvalues).</div>
      `
    },
    'svd': {
      title: '5. Singular Value Decomposition (SVD)',
      html: `
        <h4>The Master Matrix Factorization</h4>
        <p>Eigenvalues only work for square matrices ($n \\times n$). <strong>SVD</strong> works for *any* matrix of any shape ($m \\times n$), making it the heart of modern Data Science and ML!</p>
        <div class="theory-highlight">A = U · Σ · Vᵀ<br>Rotate (Vᵀ) → Stretch (Σ) → Rotate (U)</div>
        <p>Geometrically, SVD proves that <strong>any linear transformation maps a unit sphere into a hyper-ellipse</strong>. The lengths of the semi-axes of this ellipse are the singular values $\\sigma_1, \\sigma_2, \\dots$!</p>
        <div class="viva-tip">🎓 Viva Question: "What are singular values of A?"<br>Answer: The square roots of the eigenvalues of the symmetric matrix $A^T A$.</div>
      `
    },
    'rank-nullity': {
      title: '6. Rank-Nullity Theorem & Dimensional Collapse',
      html: `
        <h4>The Fundamental Theorem</h4>
        <p>When a matrix transforms space, every input vector either lands somewhere in the <strong>Column Space (Image)</strong> or gets squashed to the zero vector $(0, 0)$ in the <strong>Nullspace (Kernel)</strong>.</p>
        <div class="theory-highlight">dim(ker A) + dim(im A) = n  (Rank-Nullity Theorem)</div>
        <p>For a $2 \\times 2$ matrix with $\\det = 0$:</p>
        <ul>
          <li>Rank = 1 (Range is a 1D line)</li>
          <li>Nullity = 1 (Kernel is a 1D line that gets compressed to origin)</li>
          <li>$1 + 1 = 2$ dimensions accounted for!</li>
        </ul>
      `
    },
    'optimizers': {
      title: '7. Gradient Descent & Loss Landscapes',
      html: `
        <h4>How AI Learns</h4>
        <p>Training a neural network means finding weights $(w_1, w_2)$ that minimize a Loss function $L(w)$. The gradient $\\nabla L$ points in the direction of steepest ascent, so we take steps in the negative gradient direction:</p>
        <div class="theory-highlight">w_{t+1} = w_t - α · ∇L(w_t)</div>
        <ul>
          <li><strong>SGD:</strong> Takes raw gradient steps; oscillates wildly in narrow ravines.</li>
          <li><strong>Momentum:</strong> Adds inertia (velocity $v$) like a heavy bowling ball to blow past local saddle points.</li>
          <li><strong>RMSprop:</strong> Normalizes steps by running variance of gradients to equalize slow and fast directions.</li>
          <li><strong>Adam:</strong> Combines Momentum (1st moment) + RMSprop (2nd moment) with bias correction—the gold standard of Deep Learning!</li>
        </ul>
      `
    },
    'backprop': {
      title: '8. Autograd & The Chain Rule',
      html: `
        <h4>Reverse-Mode Automatic Differentiation</h4>
        <p>To train a network with billions of parameters, calculating numerical derivatives $\\frac{f(x+h) - f(x)}{h}$ would require billions of forward passes. Backpropagation solves this in <strong>one single pass</strong> using the chain rule on a Directed Acyclic Graph (DAG):</p>
        <div class="theory-highlight">∂L / ∂w = (∂L / ∂y) · (∂y / ∂w)</div>
        <p>During the forward pass, values are evaluated from left to right. During the backward pass, gradients are accumulated from right to left!</p>
      `
    }
  };

  function updateNotesTopic(key) {
    var item = LECTURE_NOTES[key] || LECTURE_NOTES['matrix-transform'];
    var container = $('notes-content-container');
    if (container) {
      container.innerHTML = '<h3>' + item.title + '</h3>' + item.html;
    }
  }

  // ── Viva Prep Quiz Engine ─────────────────────────────────────────────────

  var QUIZ_QUESTIONS = [
    {
      q: '1. Geometrically, what does it mean if the determinant of a 2x2 matrix is negative (det(A) < 0)?',
      opts: [
        'A) Space expands infinitely',
        'B) Space has collapsed into a point',
        'C) Space is inverted / mirrored (orientation reversed)',
        'D) The matrix is an identity matrix'
      ],
      correct: 2,
      exp: 'Correct! A negative determinant indicates that orientation (chirality) has been flipped, like reflecting a sheet of paper.'
    },
    {
      q: '2. If an eigenvalue λ = 0 for a matrix A, what does this guarantee?',
      opts: [
        'A) The matrix is symmetric',
        'B) The matrix is singular (det(A) = 0 and has no inverse)',
        'C) The matrix is an orthogonal rotation',
        'D) All eigenvalues must be zero'
      ],
      correct: 1,
      exp: 'Correct! Because det(A) equals the product of all eigenvalues (det = λ₁ · λ₂). If any eigenvalue is 0, det(A) = 0 and A is singular.'
    },
    {
      q: '3. Why is matrix multiplication generally non-commutative (AB ≠ BA)?',
      opts: [
        'A) Rounding errors in floating point arithmetic',
        'B) Because consecutive spatial transformations depend on order (e.g. rotate then shear ≠ shear then rotate)',
        'C) Because vectors do not have inverses',
        'D) It is actually commutative for all matrices'
      ],
      correct: 1,
      exp: 'Correct! The sequence of geometric transformations matters—rotating then shearing leaves points in a completely different spot than shearing then rotating.'
    },
    {
      q: '4. What shape does a 2D unit circle always transform into under any linear transformation?',
      opts: [
        'A) A triangle',
        'B) An ellipse (or collapsed line segment)',
        'C) A square',
        'D) A parabola'
      ],
      correct: 1,
      exp: 'Correct! Under SVD, any linear transformation maps the unit circle to an ellipse whose semi-major and semi-minor axes equal the singular values σ₁ and σ₂.'
    },
    {
      q: '5. In the diagonalization formula A = PDP⁻¹, what do the columns of matrix P represent?',
      opts: [
        'A) The gradient vectors',
        'B) The linearly independent eigenvectors of A',
        'C) The inverse determinant',
        'D) The standard basis vectors'
      ],
      correct: 1,
      exp: 'Correct! Matrix P is constructed by placing the eigenvectors as its column vectors: P = [v₁ | v₂ | ... | vₙ].'
    },
    {
      q: '6. According to the Rank-Nullity Theorem, if a 2x2 matrix has rank 1, what is the dimension of its nullspace (kernel)?',
      opts: [
        'A) 0',
        'B) 1',
        'C) 2',
        'D) Infinity'
      ],
      correct: 1,
      exp: 'Correct! Rank + Nullity = n. Here 1 + Nullity = 2 => Nullity = 1 (a 1D line compressed to zero).'
    },
    {
      q: '7. What does the dot product u · v = 0 indicate about two non-zero vectors?',
      opts: [
        'A) They are parallel',
        'B) They are orthogonal (perpendicular, angle 90°)',
        'C) One vector is the zero vector',
        'D) They have equal magnitude'
      ],
      correct: 1,
      exp: 'Correct! u · v = ||u|| ||v|| cos(θ). If u · v = 0 and neither is zero, cos(θ) = 0, meaning θ = 90°.'
    },
    {
      q: '8. In Deep Learning, why is the Adam optimizer preferred over standard SGD on complex loss surfaces?',
      opts: [
        'A) It does not require calculating gradients',
        'B) It combines Momentum (velocity) with RMSprop (adaptive per-parameter learning rate)',
        'C) It only works on convex quadratic functions',
        'D) It is slower and requires more memory'
      ],
      correct: 1,
      exp: 'Correct! Adam uses running 1st moments (momentum) to overcome local plateaus and 2nd moments (adaptive scale) to equalize steep and shallow directions.'
    },
    {
      q: '9. What does the Power Iteration algorithm compute by repeatedly multiplying a vector by matrix A (Aᵏx)?',
      opts: [
        'A) The smallest eigenvalue',
        'B) The dominant eigenvector (corresponding to the eigenvalue with largest magnitude)',
        'C) The determinant of A',
        'D) The trace of A'
      ],
      correct: 1,
      exp: 'Correct! Since the component along the largest eigenvalue grows as λ₁ᵏ, repeated multiplication naturally aligns any random vector with the dominant eigenvector.'
    },
    {
      q: '10. What is the fundamental mathematical tool used by Backpropagation to compute derivatives?',
      opts: [
        'A) Simpson’s Rule',
        'B) The Chain Rule of calculus applied in reverse topological order',
        'C) Monte Carlo integration',
        'D) Cauchy-Schwarz Inequality'
      ],
      correct: 1,
      exp: 'Correct! Backprop computes ∂L/∂w = (∂L/∂y) · (∂y/∂w) by flowing gradients backwards along the computational graph.'
    }
  ];

  var currentQuizIdx = 0;
  var quizScore = 0;
  var quizAnswered = false;

  function renderQuizQuestion() {
    var q = QUIZ_QUESTIONS[currentQuizIdx];
    var container = $('quiz-question-card');
    if (!container) return;

    quizAnswered = false;
    var html = '<div class="quiz-prompt">' + q.q + '</div><div class="quiz-options-list">';
    q.opts.forEach(function (opt, idx) {
      html += '<div class="quiz-option" data-idx="' + idx + '">' + opt + '</div>';
    });
    html += '</div><div id="quiz-feedback-box" class="quiz-explanation hidden"></div>';
    container.innerHTML = html;

    $('quiz-score-badge').textContent = 'Score: ' + quizScore + '/' + QUIZ_QUESTIONS.length;

    // Attach option clicks
    container.querySelectorAll('.quiz-option').forEach(function (el) {
      el.addEventListener('click', function () {
        if (quizAnswered) return;
        var selectedIdx = parseInt(this.getAttribute('data-idx'), 10);
        checkQuizAnswer(selectedIdx, q);
      });
    });
  }

  function checkQuizAnswer(selectedIdx, q) {
    quizAnswered = true;
    var container = $('quiz-question-card');
    var opts = container.querySelectorAll('.quiz-option');
    var feedbackBox = $('quiz-feedback-box');

    if (selectedIdx === q.correct) {
      opts[selectedIdx].classList.add('correct');
      quizScore++;
      feedbackBox.innerHTML = '<strong style="color:#6ee7b7">✓ Correct!</strong> ' + q.exp;
    } else {
      opts[selectedIdx].classList.add('incorrect');
      opts[q.correct].classList.add('correct');
      feedbackBox.innerHTML = '<strong style="color:#fca5a5">✗ Incorrect.</strong> ' + q.exp;
    }
    feedbackBox.classList.remove('hidden');
    $('quiz-score-badge').textContent = 'Score: ' + quizScore + '/' + QUIZ_QUESTIONS.length;
  }

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

  // ── Snapshot Export ───────────────────────────────────────────────────────

  function exportSnapshot() {
    var offscreen = document.createElement('canvas');
    offscreen.width = 1600;
    offscreen.height = 1200;
    var offCtx = offscreen.getContext('2d');

    offCtx.fillStyle = '#070a13';
    offCtx.fillRect(0, 0, offscreen.width, offscreen.height);
    offCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, offscreen.width, offscreen.height);

    offCtx.font = '700 18px Inter, sans-serif';
    offCtx.fillStyle = '#6366f1';
    offCtx.fillText('📐 TensorForge — Geometric Linear Algebra & ML Suite', 32, 48);

    var link = document.createElement('a');
    link.download = 'tensorforge-export-' + Date.now() + '.png';
    link.href = offscreen.toDataURL('image/png');
    link.click();
  }

  // ── UI Events Setup ───────────────────────────────────────────────────────

  function initEvents() {
    window.addEventListener('resize', resizeCanvas);

    // Matrix inputs
    [matAInput, matBInput, matCInput, matDInput].forEach(function (inp) {
      inp.addEventListener('input', readMatrixInputs);
    });
    [matBAInput, matBBInput, matBCInput, matBDInput].forEach(function (inp) {
      inp.addEventListener('input', readMatrixBInputs);
    });
    [vecUXInput, vecUYInput, vecVXInput, vecVYInput].forEach(function (inp) {
      inp.addEventListener('input', readVectorInputs);
    });

    // 2D Presets
    document.querySelectorAll('.btn-preset[data-preset]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        applyPreset(this.getAttribute('data-preset'));
      });
    });

    // Matrix powers stepper
    document.querySelectorAll('.btn-power[data-power]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var p = parseInt(this.getAttribute('data-power'), 10);
        document.querySelectorAll('.btn-power').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        animController.start(state.matrix, state.matrix.power(p), 500);
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

    // 2D Continuous Sliders
    rotationSlider.addEventListener('input', function () {
      var deg = parseFloat(this.value);
      rotationValue.textContent = deg + '°';
      state.matrix = Matrix2x2.rotation((deg * Math.PI) / 180);
      syncMatrixInputs(); updateTelemetry(); render();
    });

    shearSlider.addEventListener('input', function () {
      var k = parseFloat(this.value);
      shearValue.textContent = k.toFixed(2);
      state.matrix = Matrix2x2.shearX(k);
      syncMatrixInputs(); updateTelemetry(); render();
    });

    // Master Mode Navigation Tabs
    document.querySelectorAll('.mode-btn[data-mode]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.mode-btn').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        setMode(this.getAttribute('data-mode'));
      });
    });

    // 3D Sliders
    sliderYaw3D.addEventListener('input', function () {
      state.camYaw = parseFloat(this.value);
      valYaw3D.textContent = state.camYaw + '°';
      render();
    });
    sliderPitch3D.addEventListener('input', function () {
      state.camPitch = parseFloat(this.value);
      valPitch3D.textContent = state.camPitch + '°';
      render();
    });
    sliderRoll3D.addEventListener('input', function () {
      state.rotZ3D = parseFloat(this.value);
      valRoll3D.textContent = state.rotZ3D + '°';
      render();
    });

    sliderScaleX.addEventListener('input', function () { state.scaleX3D = parseFloat(this.value); $('val-scale-x').textContent = state.scaleX3D.toFixed(1); render(); });
    sliderScaleY.addEventListener('input', function () { state.scaleY3D = parseFloat(this.value); $('val-scale-y').textContent = state.scaleY3D.toFixed(1); render(); });
    sliderScaleZ.addEventListener('input', function () { state.scaleZ3D = parseFloat(this.value); $('val-scale-z').textContent = state.scaleZ3D.toFixed(1); render(); });

    // 3D Presets
    document.querySelectorAll('.btn-preset[data-preset-3d]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var p = this.getAttribute('data-preset-3d');
        if (p === 'cube') {
          state.scaleX3D = 1; state.scaleY3D = 1; state.scaleZ3D = 1; state.rotX3D = 0; state.rotY3D = 0; state.rotZ3D = 0;
        } else if (p === 'collapse3d') {
          state.scaleZ3D = 0; // Det = 0 planar collapse!
        }
        render();
      });
    });

    // LossLab Controls
    document.querySelectorAll('.btn-loss[data-loss]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.btn-loss').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        state.lossKey = this.getAttribute('data-loss');
        initLossParticles();
        render();
      });
    });

    sliderLR.addEventListener('input', function () {
      state.learningRate = parseFloat(this.value);
      valLR.textContent = state.learningRate.toFixed(3);
    });

    btnStartDescent.addEventListener('click', function () {
      state.lossRunning = !state.lossRunning;
      this.textContent = state.lossRunning ? '⏸ Pause' : '▶ Run Race';
      if (state.lossRunning) render();
    });

    btnResetDescent.addEventListener('click', function () {
      state.lossRunning = false;
      btnStartDescent.textContent = '▶ Run Race';
      initLossParticles();
      render();
    });

    // Autograd Controls
    document.querySelectorAll('.btn-autograd-preset[data-graph]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.btn-autograd-preset').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        state.autogradPreset = this.getAttribute('data-graph');
        state.autogradStep = 'idle';
        autogradStatusText.textContent = 'Preset loaded. Click Forward Pass!';
        render();
      });
    });

    btnForwardPass.addEventListener('click', function () {
      state.autogradStep = 'forward';
      autogradStatusText.textContent = 'Forward pass complete: intermediate activation values evaluated!';
      render();
    });

    btnBackwardPass.addEventListener('click', function () {
      state.autogradStep = 'backward';
      autogradStatusText.textContent = 'Backprop complete: ∂L/∂w gradients accumulated via chain rule!';
      render();
    });

    // Notes Topic Selector
    var notesSelect = $('notes-topic-select');
    if (notesSelect) {
      notesSelect.addEventListener('change', function () {
        updateNotesTopic(this.value);
      });
    }

    // Viva Quiz Next Button
    var btnQuizNext = $('btn-quiz-next');
    if (btnQuizNext) {
      btnQuizNext.addEventListener('click', function () {
        currentQuizIdx = (currentQuizIdx + 1) % QUIZ_QUESTIONS.length;
        renderQuizQuestion();
      });
    }

    // Vector Sandbox quick helpers
    var btnNormU = $('btn-normalize-u');
    if (btnNormU) {
      btnNormU.addEventListener('click', function () {
        state.vecU = state.vecU.normalize();
        vecUXInput.value = state.vecU.x.toFixed(1); vecUYInput.value = state.vecU.y.toFixed(1);
        render();
      });
    }

    var btnOrthoV = $('btn-orthogonalize-v');
    if (btnOrthoV) {
      btnOrthoV.addEventListener('click', function () {
        state.vecV = state.vecV.sub(state.vecV.projectOnto(state.vecU));
        vecVXInput.value = state.vecV.x.toFixed(1); vecVYInput.value = state.vecV.y.toFixed(1);
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

    // Zoom & Pan Actions
    $('btn-zoom-in').addEventListener('click', function () { state.scale = Math.min(180, state.scale * 1.2); render(); });
    $('btn-zoom-out').addEventListener('click', function () { state.scale = Math.max(25, state.scale / 1.2); render(); });
    $('btn-recenter').addEventListener('click', function () { state.panX = 0; state.panY = 0; state.scale = 65; render(); });
    $('btn-reset').addEventListener('click', function () { applyPreset('identity'); });
    $('btn-snapshot').addEventListener('click', exportSnapshot);

    $('btn-toggle-grid').addEventListener('click', function () {
      state.showTransformedGrid = !state.showTransformedGrid;
      this.style.color = state.showTransformedGrid ? 'var(--accent-primary)' : 'var(--text-muted)';
      render();
    });

    $('btn-help').addEventListener('click', function () { modalHelp.classList.remove('hidden'); });
    $('btn-close-help').addEventListener('click', function () { modalHelp.classList.add('hidden'); });
    modalHelp.addEventListener('click', function (e) { if (e.target === modalHelp) modalHelp.classList.add('hidden'); });

    // Keyboard Shortcuts
    document.addEventListener('keydown', function (e) {
      var tag = document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if (e.key === 'Escape') { modalHelp.classList.add('hidden'); return; }
      if (e.key === 'r' || e.key === 'R') { applyPreset('identity'); return; }
      if (e.key === 'c' || e.key === 'C') { $('btn-recenter').click(); return; }
      if (e.key === 'g' || e.key === 'G') { $('btn-toggle-grid').click(); return; }
      if (e.key === '?') { modalHelp.classList.toggle('hidden'); return; }
      if (e.key >= '1' && e.key <= '9') {
        var idx = parseInt(e.key, 10) - 1;
        var btns = document.querySelectorAll('.mode-btn');
        if (btns[idx]) btns[idx].click();
      }
    });
  }

  function setMode(newMode) {
    state.mode = newMode;

    // Toggle panels in sidebar
    var panels = ['transform', 'eigen', 'mult', 'vectors', '3d', 'loss', 'autograd', 'notes', 'quiz'];
    panels.forEach(function (p) {
      var el = $('panel-' + p);
      if (el) el.classList.toggle('hidden', newMode !== p);
    });

    multDrawer.classList.toggle('active', newMode === 'mult');

    if (newMode === 'loss') {
      initLossParticles();
    } else if (newMode === 'notes') {
      updateNotesTopic($('notes-topic-select').value);
    } else if (newMode === 'quiz') {
      renderQuizQuestion();
    }

    render();
  }

  // ── URL State Sync ────────────────────────────────────────────────────────

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
    if (params.m) {
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
    initLossParticles();
    updateNotesTopic('matrix-transform');
    renderQuizQuestion();
    resizeCanvas();
  }

  // ── Public API for Lectern ────────────────────────────────────────────────

  window.TensorForge = {
    setMatrix: function (a, b, c, d) {
      state.matrix = new Matrix2x2(a, b, c, d);
      syncMatrixInputs(); updateTelemetry(); render();
    },
    getMatrix: function () { return state.matrix.clone(); },
    setMode: setMode,
    setShape: function (shapeName) { state.shape = shapeName; render(); },
    reset: function () { applyPreset('identity'); }
  };

  init();

})();
