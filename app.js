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
    shape: 'square', // 'square' | 'circle' | 'house' | 'letterF' | 'polygon' | 'cloud'

    // 2D Matrices
    matrix: new Matrix2x2(1.5, 0.5, 0.5, 1.2),
    matrixB: new Matrix2x2(0.8, -0.6, 0.6, 0.8),
    multT: 0,
    multOrder: 'AB',

    // Continuous Morph Timeline (I -> A)
    morphT: 1.0,
    morphPlaying: false,

    // 2D Vectors, Solver & Probes
    customVec: new Vector2D(1.0, 1.0),
    solveB: new Vector2D(2.0, 1.5),
    showSolver: true,
    eigenProbe: new Vector2D(1.0, 0.0),
    showSpiral: true,
    showGershgorin: true,
    magneticSnap: true,
    vecU: new Vector2D(2.0, 1.0),
    vecV: new Vector2D(1.0, 2.0),
    showDecomp: true,
    showParallelogram: true,
    snapToGrid: true,

    // 3D Space State
    camYaw: 25,
    camPitch: 20,
    rotX3D: 0,
    rotY3D: 0,
    rotZ3D: 0,
    scaleX3D: 1.0,
    scaleY3D: 1.0,
    scaleZ3D: 1.0,
    shearXY3D: 0,
    camOrthographic: false,
    autoOrbit3D: false,
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

  // baseMatrix stores the user's un-powered matrix so power-stepper can be purely visual
  var state_basePowerMatrix = null;

  // Main animation controller (for preset transitions and user-driven animations)
  var animController = new Engine.AnimationController(function (currentMatrix, easeT) {
    state.matrix = currentMatrix;
    syncMatrixInputs();
    updateTelemetry();
    render();
  });

  // Dedicated power-preview controller: renders A^k but NEVER writes to state.matrix
  var powerAnimController = new Engine.AnimationController(
    function (currentMatrix) {
      // Render using temporary matrix but don't update state, inputs, or hash
      var savedMatrix = state.matrix;
      state.matrix = currentMatrix;
      render();
      state.matrix = savedMatrix;
    },
    function (finalMatrix) {
      // Animation done — stay at base matrix, reset button to A¹
      state_basePowerMatrix = null;
      render();
      document.querySelectorAll('.btn-power').forEach(function (b) {
        b.classList.toggle('active', b.getAttribute('data-power') === '1');
      });
    }
  );

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

      if (state.mode === 'mult' && state.multT > 0.45) {
        var firstMat = state.multOrder === 'AB' ? state.matrixB : state.matrix;
        var firstLabel = state.multOrder === 'AB' ? 'Step 1: B·x' : 'Step 1: A·x';
        drawGhostShape(firstMat, firstLabel);
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
      } else {
        if (state.showCustomVec) {
          drawCustomVector(activeMatrix);
        }
        if (state.showSolver && state.mode === 'transform') {
          drawLinearSolver(activeMatrix);
        }
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
    if (state.mode === 'transform' && typeof state.morphT === 'number' && state.morphT < 1.0) {
      return Matrix2x2.lerp(Matrix2x2.identity(), state.matrix, state.morphT);
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

    } else if (state.shape === 'polygon') {
      var starPts = [];
      for (var p = 0; p < 10; p++) {
        var r = p % 2 === 0 ? 1.0 : 0.42;
        var a = (p * Math.PI) / 5 - Math.PI / 2;
        starPts.push(new Vector2D(r * Math.cos(a), r * Math.sin(a)));
      }
      ctx.beginPath();
      starPts.forEach(function (pt, idx) {
        var t = matrix.apply(pt);
        var sp = worldToScreen(t.x, t.y);
        if (idx === 0) ctx.moveTo(sp.x, sp.y);
        else ctx.lineTo(sp.x, sp.y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      drawTopRightHUDTag('Custom 5-Pointed Star Polygon Transformation');

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

  // ── Linear System Solver Renderer (Ax = b) ────────────────────────────────

  function drawLinearSolver(matrix) {
    if (!state.showSolver) return;
    var o = worldToScreen(0, 0);

    // Target vector b (amber / gold)
    var bPos = worldToScreen(state.solveB.x, state.solveB.y);
    drawArrow(o.x, o.y, bPos.x, bPos.y, '#f59e0b', 2.5);
    drawVectorHandle(bPos.x, bPos.y, '#f59e0b', 'solveB', state.hoverTarget === 'solveB');
    drawVectorLabel('Target b [' + state.solveB.x.toFixed(1) + ', ' + state.solveB.y.toFixed(1) + ']', bPos.x, bPos.y, '#f59e0b', state.solveB.angle());

    // Preimage x = A^-1 b (purple)
    var det = matrix.determinant();
    if (Math.abs(det) > Engine.EPSILON) {
      var xVec = matrix.solve(state.solveB);
      if (xVec) {
        var xPos = worldToScreen(xVec.x, xVec.y);
        ctx.save();
        ctx.setLineDash([4, 4]);
        drawArrow(o.x, o.y, xPos.x, xPos.y, '#c084fc', 2);
        ctx.restore();
        drawVectorLabel('x = A⁻¹b [' + xVec.x.toFixed(1) + ', ' + xVec.y.toFixed(1) + ']', xPos.x, xPos.y, '#c084fc', xVec.angle());
      }
    }
  }

  function drawTopRightHUDTag(text) {
    ctx.save();
    ctx.font = '600 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
    var w = ctx.measureText(text).width;
    var x = viewWidth - 20;
    var y = 28;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    var pad = 10;
    var rectX = x - w - pad * 2;
    var rectY = y - 12;
    var rectW = w + pad * 2;
    var rectH = 24;
    if (ctx.roundRect) {
      ctx.roundRect(rectX, rectY, rectW, rectH, 6);
    } else {
      ctx.rect(rectX, rectY, rectW, rectH);
    }
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#c7d2fe';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x - pad, y);
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
      var slope = Math.abs(v.x) > 1e-4 ? (v.y / v.x).toFixed(2) : '∞';
      ctx.fillStyle = colors[index % colors.length];
      ctx.font = '600 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillText('Span(v' + (index + 1) + ') λ=' + ev.lambda.toFixed(2) + ' [y=' + slope + 'x]', labelPos.x + 6, labelPos.y - 6);
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

    // 1. Gershgorin Discs Overlay (if enabled)
    if (state.showGershgorin) {
      var g = matrix.gershgorinDiscs();
      var d1Center = worldToScreen(g.disc1.center, 0);
      var d1Rad = g.disc1.radius * state.scale;
      var d2Center = worldToScreen(g.disc2.center, 0);
      var d2Rad = g.disc2.radius * state.scale;

      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.2;

      // Disc 1
      ctx.fillStyle = 'rgba(245, 158, 11, 0.08)';
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
      ctx.beginPath();
      ctx.arc(d1Center.x, d1Center.y, Math.max(3, d1Rad), 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();

      // Disc 2
      ctx.fillStyle = 'rgba(168, 85, 247, 0.08)';
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.5)';
      ctx.beginPath();
      ctx.arc(d2Center.x, d2Center.y, Math.max(3, d2Rad), 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      ctx.restore();
    }

    // 2. Complex Spiral Phase Portrait (when discriminant < 0)
    var eigens = Engine.solveEigensystem(matrix);
    if (!eigens.isReal && state.showSpiral) {
      ctx.save();
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();

      var currV = state.eigenProbe.clone();
      var p0 = worldToScreen(currV.x, currV.y);
      ctx.moveTo(p0.x, p0.y);

      for (var s = 1; s <= 32; s++) {
        currV = matrix.apply(currV);
        if (currV.magnitude() > 14) break;
        var pNext = worldToScreen(currV.x, currV.y);
        ctx.lineTo(pNext.x, pNext.y);
      }
      ctx.stroke();

      // Orbit milestone dots
      currV = state.eigenProbe.clone();
      for (var s2 = 1; s2 <= 8; s2++) {
        currV = matrix.apply(currV);
        if (currV.magnitude() > 14) break;
        var pDot = worldToScreen(currV.x, currV.y);
        ctx.fillStyle = s2 % 2 === 0 ? '#38bdf8' : '#c084fc';
        ctx.beginPath();
        ctx.arc(pDot.x, pDot.y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
      drawTopRightHUDTag('Complex Spiral Phase Portrait (r e^{iθ})');
    }

    // 3. Unit circle guide
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
    drawBackgroundGrid();
    drawAxes();

    var o = worldToScreen(0, 0);
    var uPos = worldToScreen(state.vecU.x, state.vecU.y);
    var vPos = worldToScreen(state.vecV.x, state.vecV.y);

    // 1. Shaded Parallelogram (Span & 2D Cross Product Area)
    if (state.showParallelogram) {
      var crossVal = state.vecU.cross(state.vecV);
      var sumVec = state.vecU.add(state.vecV);
      var sPos = worldToScreen(sumVec.x, sumVec.y);

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(o.x, o.y);
      ctx.lineTo(uPos.x, uPos.y);
      ctx.lineTo(sPos.x, sPos.y);
      ctx.lineTo(vPos.x, vPos.y);
      ctx.closePath();
      ctx.fillStyle = crossVal >= 0 ? 'rgba(16, 185, 129, 0.12)' : 'rgba(168, 85, 247, 0.12)';
      ctx.fill();
      ctx.strokeStyle = crossVal >= 0 ? 'rgba(16, 185, 129, 0.35)' : 'rgba(168, 85, 247, 0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.stroke();

      // Parallelogram center area text badge
      if (Math.abs(crossVal) > 0.1) {
        var cX = (o.x + sPos.x) / 2;
        var cY = (o.y + sPos.y) / 2;
        ctx.fillStyle = crossVal >= 0 ? '#34d399' : '#c084fc';
        ctx.font = '600 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Area = ' + Math.abs(crossVal).toFixed(2), cX, cY);
      }
      ctx.restore();
    }

    // 2. Translucent Sector Wedge & Arc for Angle
    var angleU = state.vecU.angle();
    var angleV = state.vecV.angle();
    var diff = angleV - angleU;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    var arcRad = 36;

    ctx.save();
    ctx.fillStyle = 'rgba(56, 189, 248, 0.08)';
    ctx.beginPath();
    ctx.moveTo(o.x, o.y);
    ctx.arc(o.x, o.y, arcRad, -angleU, -(angleU + diff), diff < 0);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(o.x, o.y, arcRad, -angleU, -(angleU + diff), diff < 0);
    ctx.stroke();

    var midAngle = -angleU - diff / 2;
    var tx = o.x + Math.cos(midAngle) * (arcRad + 14);
    var ty = o.y + Math.sin(midAngle) * (arcRad + 14);
    ctx.fillStyle = '#38bdf8';
    ctx.font = '600 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText((Math.abs(diff * 180 / Math.PI)).toFixed(1) + '°', tx, ty);
    ctx.restore();

    // 3. Orthogonal Decomposition (v = v∥ + v⊥)
    if (state.showDecomp) {
      var projParallel = state.vecV.projectOnto(state.vecU);
      var projPerp = state.vecV.rejectFrom(state.vecU);
      var pPos = worldToScreen(projParallel.x, projParallel.y);

      // Parallel projection arrow v∥ along u
      if (projParallel.magnitude() > 0.05) {
        drawArrow(o.x, o.y, pPos.x, pPos.y, '#f59e0b', 3.2);
        drawVectorLabel('v∥ [' + projParallel.x.toFixed(1) + ', ' + projParallel.y.toFixed(1) + ']', pPos.x, pPos.y, '#f59e0b', projParallel.angle());
      }

      // Orthogonal rejection line v⊥ from pPos to vPos
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(pPos.x, pPos.y);
      ctx.lineTo(vPos.x, vPos.y);
      ctx.stroke();

      // Right angle marker
      if (projParallel.magnitude() > 0.4 && projPerp.magnitude() > 0.4) {
        var uDir = state.vecU.normalize();
        var perpDir = projPerp.normalize();
        var mSize = 8;
        var m1x = pPos.x + perpDir.x * mSize;
        var m1y = pPos.y - perpDir.y * mSize;
        var m2x = m1x - uDir.x * mSize;
        var m2y = m1y + uDir.y * mSize;
        var m3x = pPos.x - uDir.x * mSize;
        var m3y = pPos.y + uDir.y * mSize;
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(m1x, m1y);
        ctx.lineTo(m2x, m2y);
        ctx.lineTo(m3x, m3y);
        ctx.stroke();
      }
      ctx.restore();
    }

    // 4. Parallelogram Addition Dashed Guides & Sum Vector u + v
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

    // 5. Primary Vectors u & v
    drawArrow(o.x, o.y, uPos.x, uPos.y, '#f43f5e', 2.8);
    drawVectorHandle(uPos.x, uPos.y, '#f43f5e', 'u', state.hoverTarget === 'u');
    drawVectorLabel('u [' + state.vecU.x.toFixed(1) + ', ' + state.vecU.y.toFixed(1) + ']', uPos.x, uPos.y, '#f43f5e', state.vecU.angle());

    drawArrow(o.x, o.y, vPos.x, vPos.y, '#06b6d4', 2.8);
    drawVectorHandle(vPos.x, vPos.y, '#06b6d4', 'v_sandbox', state.hoverTarget === 'v_sandbox');
    drawVectorLabel('v [' + state.vecV.x.toFixed(1) + ', ' + state.vecV.y.toFixed(1) + ']', vPos.x, vPos.y, '#06b6d4', state.vecV.angle());

    updateVectorSandboxTelemetry();
  }

  function updateVectorSandboxTelemetry() {
    var dot = state.vecU.dot(state.vecV);
    var magU = state.vecU.magnitude();
    var magV = state.vecV.magnitude();
    var cosTheta = (magU > 0 && magV > 0) ? Math.max(-1, Math.min(1, dot / (magU * magV))) : 1;
    var angleDeg = (Math.acos(cosTheta) * 180) / Math.PI;

    // Vector Arithmetic sync
    var sum = state.vecU.add(state.vecV);
    var diff = state.vecU.sub(state.vecV);
    var valVecSum = $('val-vec-sum');
    var valVecSumMag = $('val-vec-sum-mag');
    var valVecDiff = $('val-vec-diff');
    var valVecDiffMag = $('val-vec-diff-mag');
    if (valVecSum) valVecSum.textContent = '[ ' + sum.x.toFixed(2) + ', ' + sum.y.toFixed(2) + ' ]';
    if (valVecSumMag) valVecSumMag.textContent = '||u+v|| = ' + sum.magnitude().toFixed(2);
    if (valVecDiff) valVecDiff.textContent = '[ ' + diff.x.toFixed(2) + ', ' + diff.y.toFixed(2) + ' ]';
    if (valVecDiffMag) valVecDiffMag.textContent = '||u-v|| = ' + diff.magnitude().toFixed(2);

    // Polar coordinates
    var polU = state.vecU.polar();
    var polV = state.vecV.polar();
    var elPolU = $('val-polar-u');
    var elPolV = $('val-polar-v');
    if (elPolU) elPolU.textContent = 'r=' + polU.r.toFixed(2) + ', θ=' + polU.thetaDeg.toFixed(1) + '°';
    if (elPolV) elPolV.textContent = 'r=' + polV.r.toFixed(2) + ', θ=' + polV.thetaDeg.toFixed(1) + '°';

    // Dot product & Angle
    if (valDotProduct) valDotProduct.textContent = dot.toFixed(2);
    if (valMagU) valMagU.textContent = magU.toFixed(2);
    if (valMagV) valMagV.textContent = magV.toFixed(2);
    if (valVectorAngle) valVectorAngle.textContent = angleDeg.toFixed(1) + '°';
    var badgeCos = $('badge-vector-cos');
    if (badgeCos) badgeCos.textContent = 'cos θ = ' + cosTheta.toFixed(2);

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

    // Gram-Schmidt Orthogonal Decomposition
    var projParallel = state.vecV.projectOnto(state.vecU);
    var projPerp = state.vecV.rejectFrom(state.vecU);
    var elProjPar = $('val-vec-proj-parallel');
    var elProjPerp = $('val-vec-proj-perp');
    var elProjDot = $('val-vec-proj-dot');
    if (elProjPar) elProjPar.textContent = '[ ' + projParallel.x.toFixed(2) + ', ' + projParallel.y.toFixed(2) + ' ]';
    if (elProjPerp) elProjPerp.textContent = '[ ' + projPerp.x.toFixed(2) + ', ' + projPerp.y.toFixed(2) + ' ]';
    var decompDot = projParallel.dot(projPerp);
    if (elProjDot) elProjDot.textContent = Math.abs(decompDot) < 1e-6 ? '0.00 ✓ (Orthogonal)' : decompDot.toFixed(2);

    // Fundamental Inequalities
    // 1. Cauchy-Schwarz: |u·v| <= ||u|| ||v||
    var absDot = Math.abs(dot);
    var prodMag = magU * magV;
    var elCS = $('val-cauchy-schwarz');
    var badgeCS = $('badge-cauchy');
    var ratioCS = $('val-cauchy-ratio');
    if (elCS) elCS.textContent = absDot.toFixed(2) + ' ≤ ' + prodMag.toFixed(2);
    var satRatio = prodMag > 0 ? (absDot / prodMag) * 100 : 100;
    if (ratioCS) ratioCS.textContent = satRatio.toFixed(1) + '% Saturation';
    if (badgeCS) {
      if (Math.abs(absDot - prodMag) < 0.05) {
        badgeCS.textContent = 'Collinear Equality (= 100%)';
        badgeCS.className = 'telemetry-badge badge-det-pos';
      } else {
        badgeCS.textContent = 'Strict Inequality (<)';
        badgeCS.className = 'telemetry-badge badge-det-pos';
      }
    }

    // 2. Triangle Inequality: ||u+v|| <= ||u|| + ||v||
    var sumMag = sum.magnitude();
    var sumIndiv = magU + magV;
    var elTri = $('val-triangle-ineq');
    var badgeTri = $('badge-triangle');
    var diffTri = $('val-triangle-diff');
    if (elTri) elTri.textContent = sumMag.toFixed(2) + ' ≤ ' + sumIndiv.toFixed(2);
    if (diffTri) diffTri.textContent = 'Deficit: ' + (sumIndiv - sumMag).toFixed(2);
    if (badgeTri) {
      if (Math.abs(sumMag - sumIndiv) < 0.05) {
        badgeTri.textContent = 'Degenerate Equality (=)';
        badgeTri.className = 'telemetry-badge badge-det-pos';
      } else {
        badgeTri.textContent = 'Triangle Inequality Holds';
        badgeTri.className = 'telemetry-badge badge-det-pos';
      }
    }

    // 2D Cross Product & Span Area
    var cross = state.vecU.cross(state.vecV);
    var elCross = $('val-vec-cross');
    var badgeOrient = $('badge-vec-orientation');
    var elSpanDim = $('val-vec-span-dim');
    var badgeIndep = $('badge-vec-independent');
    if (elCross) elCross.textContent = Math.abs(cross).toFixed(2);
    if (badgeOrient) {
      if (Math.abs(cross) < 1e-4) {
        badgeOrient.textContent = 'Zero Area (Collinear)';
        badgeOrient.className = 'telemetry-badge badge-det-zero';
      } else if (cross > 0) {
        badgeOrient.textContent = 'Counter-Clockwise (+)';
        badgeOrient.className = 'telemetry-badge badge-det-pos';
      } else {
        badgeOrient.textContent = 'Clockwise (-)';
        badgeOrient.className = 'telemetry-badge badge-det-neg';
      }
    }
    if (elSpanDim) {
      elSpanDim.textContent = Math.abs(cross) < 1e-4 ? 'dim = 1' : 'dim = 2';
    }
    if (badgeIndep) {
      if (Math.abs(cross) < 1e-4) {
        badgeIndep.textContent = 'Linearly Dependent';
        badgeIndep.className = 'telemetry-badge badge-det-zero';
      } else {
        badgeIndep.textContent = 'Linearly Independent';
        badgeIndep.className = 'telemetry-badge badge-det-pos';
      }
    }
  }

  // ── 3D VectorSpace Renderer (Mode 5) ──────────────────────────────────────

  function drawPlanarSubspaceGrid(normal, pitchRad, yawRad, fov, originX, originY) {
    var arb = Math.abs(normal.z) < 0.9 ? new Vector3D(0, 0, 1) : new Vector3D(1, 0, 0);
    var u = normal.cross(arb).normalize();
    var v = normal.cross(u).normalize();

    ctx.save();
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.16)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    var extent = 3.2;
    var step = 0.8;
    for (var i = -extent; i <= extent; i += step) {
      var p1 = u.scale(i).add(v.scale(-extent));
      var p2 = u.scale(i).add(v.scale(extent));
      var proj1 = project3DTo2D(p1, pitchRad, yawRad, fov, state.camOrthographic);
      var proj2 = project3DTo2D(p2, pitchRad, yawRad, fov, state.camOrthographic);
      ctx.moveTo(originX + proj1.x, originY + proj1.y);
      ctx.lineTo(originX + proj2.x, originY + proj2.y);

      var q1 = v.scale(i).add(u.scale(-extent));
      var q2 = v.scale(i).add(u.scale(extent));
      var projQ1 = project3DTo2D(q1, pitchRad, yawRad, fov, state.camOrthographic);
      var projQ2 = project3DTo2D(q2, pitchRad, yawRad, fov, state.camOrthographic);
      ctx.moveTo(originX + projQ1.x, originY + projQ1.y);
      ctx.lineTo(originX + projQ2.x, originY + projQ2.y);
    }
    ctx.stroke();

    // Normal vector arrow from origin
    var normProj = project3DTo2D(normal.scale(2.2), pitchRad, yawRad, fov, state.camOrthographic);
    var o2D = project3DTo2D(new Vector3D(0, 0, 0), pitchRad, yawRad, fov, state.camOrthographic);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(originX + o2D.x, originY + o2D.y);
    ctx.lineTo(originX + normProj.x, originY + normProj.y);
    ctx.stroke();

    ctx.fillStyle = '#38bdf8';
    ctx.font = '700 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
    ctx.fillText('n (Normal)', originX + normProj.x + 6, originY + normProj.y - 4);
    ctx.restore();
  }

  function render3DSpace() {
    var yawRad = (state.camYaw * Math.PI) / 180;
    var pitchRad = (state.camPitch * Math.PI) / 180;
    var fov = 750;
    var originX = viewWidth / 2;
    var originY = viewHeight / 2;

    // Build 3D Transformation Matrix A = Rz * Ry * Rx * Shear * Scale
    var rotX = Matrix3x3.rotationX((state.rotX3D * Math.PI) / 180);
    var rotY = Matrix3x3.rotationY((state.rotY3D * Math.PI) / 180);
    var rotZ = Matrix3x3.rotationZ((state.rotZ3D * Math.PI) / 180);
    var scaleM = new Matrix3x3([
      state.scaleX3D, 0, 0,
      0, state.scaleY3D, 0,
      0, 0, state.scaleZ3D
    ]);
    var shearM = state.shearXY3D ? new Matrix3x3([ 1, state.shearXY3D, 0, 0, 1, 0, 0, 0, 1 ]) : Matrix3x3.identity();

    var transform3D = rotZ.multiply(rotY).multiply(rotX).multiply(shearM).multiply(scaleM);

    // Compute 3D Telemetry (Determinant, Trace, Rank, Columns)
    var det3D = transform3D.determinant();
    var trace3D = transform3D.trace();
    var rank3D = transform3D.rank();
    var cols3D = transform3D.getColumns();

    valDet3D.textContent = det3D.toFixed(2);
    var elTrace3D = $('val-trace-3d');
    var elDetSum3D = $('val-det-summary-3d');
    var elRank3D = $('badge-rank-3d');
    if (elTrace3D) elTrace3D.textContent = trace3D.toFixed(2);
    if (elDetSum3D) elDetSum3D.textContent = det3D.toFixed(2);
    if (elRank3D) {
      elRank3D.textContent = 'Rank: ' + rank3D;
      elRank3D.className = rank3D === 3 ? 'telemetry-badge badge-det-pos' : 'telemetry-badge badge-det-zero';
    }

    if (Math.abs(det3D) < 0.01) {
      badgeDet3D.textContent = 'Planar Collapse (det=0)';
      badgeDet3D.className = 'telemetry-badge badge-det-zero';
    } else {
      badgeDet3D.textContent = 'Volume: ' + Math.abs(det3D).toFixed(2);
      badgeDet3D.className = 'telemetry-badge badge-det-pos';
    }

    // 3x3 Matrix readout display
    var elMat3D = $('mat-3d-display');
    if (elMat3D) {
      var m = transform3D.m;
      elMat3D.innerHTML =
        '[ ' + m[0].toFixed(2).padStart(5, ' ') + ', ' + m[1].toFixed(2).padStart(5, ' ') + ', ' + m[2].toFixed(2).padStart(5, ' ') + ' ]<br>' +
        '[ ' + m[3].toFixed(2).padStart(5, ' ') + ', ' + m[4].toFixed(2).padStart(5, ' ') + ', ' + m[5].toFixed(2).padStart(5, ' ') + ' ]<br>' +
        '[ ' + m[6].toFixed(2).padStart(5, ' ') + ', ' + m[7].toFixed(2).padStart(5, ' ') + ', ' + m[8].toFixed(2).padStart(5, ' ') + ' ]';
    }

    // Planar equation if singular / rank-deficient
    var isPlanar = Math.abs(det3D) < 0.05;
    var elPlaneEq = $('val-plane-equation');
    var elPlaneNorm = $('val-plane-normal');
    if (isPlanar) {
      var n = cols3D[0].cross(cols3D[1]);
      if (n.magnitude() < 1e-4) n = cols3D[0].cross(cols3D[2]);
      if (n.magnitude() < 1e-4) n = cols3D[1].cross(cols3D[2]);
      if (n.magnitude() < 1e-4) n = new Vector3D(0, 0, 1);
      n = n.normalize();
      if (elPlaneEq) elPlaneEq.textContent = n.x.toFixed(2) + 'x + ' + n.y.toFixed(2) + 'y + ' + n.z.toFixed(2) + 'z = 0';
      if (elPlaneNorm) elPlaneNorm.textContent = '[ ' + n.x.toFixed(2) + ', ' + n.y.toFixed(2) + ', ' + n.z.toFixed(2) + ' ]';
      drawPlanarSubspaceGrid(n, pitchRad, yawRad, fov, originX, originY);
    } else {
      if (elPlaneEq) elPlaneEq.textContent = 'Full Rank ℝ³ (det ≠ 0)';
      if (elPlaneNorm) elPlaneNorm.textContent = '[ Span = ℝ³ ]';
    }

    // 1. Draw Static Reference 3D Axes
    var axes3D = [
      { v: new Vector3D(2.5, 0, 0), col: 'rgba(244, 63, 94, 0.45)', name: 'X' },
      { v: new Vector3D(0, 2.5, 0), col: 'rgba(16, 185, 129, 0.45)', name: 'Y' },
      { v: new Vector3D(0, 0, 2.5), col: 'rgba(6, 182, 212, 0.45)', name: 'Z' }
    ];

    var o2D = project3DTo2D(new Vector3D(0, 0, 0), pitchRad, yawRad, fov, state.camOrthographic);
    axes3D.forEach(function (ax) {
      var p = project3DTo2D(ax.v, pitchRad, yawRad, fov, state.camOrthographic);
      ctx.beginPath();
      ctx.strokeStyle = ax.col;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 2]);
      ctx.moveTo(originX + o2D.x, originY + o2D.y);
      ctx.lineTo(originX + p.x, originY + p.y);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // 2. Transformed 3D Basis Triad (Columns of Matrix A)
    var triad = [
      { v: cols3D[0], col: '#f43f5e', name: 'a₁' },
      { v: cols3D[1], col: '#10b981', name: 'a₂' },
      { v: cols3D[2], col: '#06b6d4', name: 'a₃' }
    ];
    triad.forEach(function (tr) {
      var p = project3DTo2D(tr.v, pitchRad, yawRad, fov, state.camOrthographic);
      var sx = originX + p.x, sy = originY + p.y;
      ctx.beginPath();
      ctx.strokeStyle = tr.col;
      ctx.lineWidth = 2.5;
      ctx.moveTo(originX + o2D.x, originY + o2D.y);
      ctx.lineTo(sx, sy);
      ctx.stroke();

      ctx.fillStyle = tr.col;
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = '700 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillText(tr.name + ' [' + tr.v.x.toFixed(1) + ',' + tr.v.y.toFixed(1) + ',' + tr.v.z.toFixed(1) + ']', sx + 6, sy - 4);
    });

    // 3. Render 3D Transformed Mesh (Cube) with Directional Lighting
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

    var transformedPts = rawCubeVertices.map(function (v) {
      return transform3D.apply(v);
    });

    var projectedPts = transformedPts.map(function (v) {
      var proj = project3DTo2D(v, pitchRad, yawRad, fov, state.camOrthographic);
      return { x: originX + proj.x, y: originY + proj.y, depth: proj.depth };
    });

    var faceList = cubeFaces.map(function (faceIndices) {
      var avgZ = (projectedPts[faceIndices[0]].depth + projectedPts[faceIndices[1]].depth + projectedPts[faceIndices[2]].depth + projectedPts[faceIndices[3]].depth) / 4;
      return { indices: faceIndices, depth: avgZ };
    });
    faceList.sort(function (a, b) { return a.depth - b.depth; });

    var lightDir = new Vector3D(0.5, 0.8, 0.4).normalize();
    faceList.forEach(function (f) {
      var v0 = transformedPts[f.indices[0]];
      var v1 = transformedPts[f.indices[1]];
      var v2 = transformedPts[f.indices[2]];
      var e1 = new Vector3D(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
      var e2 = new Vector3D(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
      var norm = e1.cross(e2).normalize();
      var diff = Math.max(0, norm.dot(lightDir));
      var intensity = 0.22 + 0.68 * diff;

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

      var r = det3D >= 0 ? Math.round(99 * intensity) : Math.round(245 * intensity);
      var g = det3D >= 0 ? Math.round(102 * intensity) : Math.round(158 * intensity);
      var b = det3D >= 0 ? Math.round(241 * intensity) : Math.round(11 * intensity);

      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ', 0.32)';
      ctx.strokeStyle = det3D >= 0 ? 'rgba(129, 140, 248, 0.75)' : 'rgba(251, 191, 36, 0.75)';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });

    drawTopRightHUDTag('3D: Yaw ' + state.camYaw + '°, Pitch ' + state.camPitch + '° (' + (state.camOrthographic ? 'Ortho' : 'Persp') + ')');
  }

  // ── LossLab Optimization Sandbox (Mode 6) ─────────────────────────────────

  function initLossParticles(customX, customY) {
    var startX = -2.0, startY = 2.0;
    if (typeof customX === 'number' && typeof customY === 'number') {
      startX = customX; startY = customY;
    } else if (state.lossKey === 'rosenbrock') {
      startX = -1.2; startY = 1.5;
    } else if (state.lossKey === 'beale') {
      startX = 2.4; startY = 1.6;
    } else if (state.lossKey === 'rastrigin') {
      startX = 1.7; startY = 1.6;
    }

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

    drawBackgroundGrid();

    // 1. Mathematically Aligned Energy Field Heatmap
    var gridStep = 20;
    ctx.save();
    for (var px = 0; px < viewWidth; px += gridStep) {
      for (var py = 0; py < viewHeight; py += gridStep) {
        var w = screenToWorld(px + gridStep / 2, py + gridStep / 2);
        var z = lossFn.evaluate(w.x, w.y);
        var intensity = Math.min(1, Math.max(0, Math.log(Math.abs(z) + 1) / 3.8));
        var rCol = Math.round(intensity * 130 + 15);
        var bCol = Math.round((1 - intensity) * 150 + 40);
        ctx.fillStyle = 'rgba(' + rCol + ', 35, ' + bCol + ', 0.18)';
        ctx.fillRect(px, py, gridStep, gridStep);
      }
    }
    ctx.restore();

    drawAxes();

    // 2. If running, step optimizers
    if (state.lossRunning) {
      state.particles.forEach(function (p) {
        p.step(lossFn, state.learningRate);
      });
    }

    // 3. Draw Particles, Negative Gradient Vectors (-∇L) & Fading Trajectories
    state.particles.forEach(function (p) {
      // Trajectory line
      if (p.history.length > 1) {
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = p.color;
        ctx.lineWidth = 2;
        p.history.forEach(function (pt, idx) {
          var sp = worldToScreen(pt.x, pt.y);
          if (idx === 0) ctx.moveTo(sp.x, sp.y);
          else ctx.lineTo(sp.x, sp.y);
        });
        ctx.stroke();
        ctx.restore();
      }

      // Negative gradient vector arrow (-∇L) showing immediate descent direction
      var grad = lossFn.gradient(p.x, p.y);
      var gMag = Math.hypot(grad.dx, grad.dy);
      if (gMag > 0.05) {
        var dirX = -grad.dx / gMag;
        var dirY = -grad.dy / gMag;
        var arrowLen = Math.min(1.2, Math.max(0.4, gMag * 0.15));
        var oSp = worldToScreen(p.x, p.y);
        var tipSp = worldToScreen(p.x + dirX * arrowLen, p.y + dirY * arrowLen);
        drawArrow(oSp.x, oSp.y, tipSp.x, tipSp.y, p.color, 1.8);
      }

      // Particle Ball
      var curSp = worldToScreen(p.x, p.y);
      ctx.beginPath();
      ctx.fillStyle = p.color;
      ctx.arc(curSp.x, curSp.y, 6.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.6;
      ctx.stroke();
    });

    // 4. Update Telemetry Values & Step Count
    if (state.particles.length >= 4) {
      lossValSGD.textContent = lossFn.evaluate(state.particles[0].x, state.particles[0].y).toFixed(3);
      lossValMom.textContent = lossFn.evaluate(state.particles[1].x, state.particles[1].y).toFixed(3);
      lossValRMS.textContent = lossFn.evaluate(state.particles[2].x, state.particles[2].y).toFixed(3);
      lossValAdam.textContent = lossFn.evaluate(state.particles[3].x, state.particles[3].y).toFixed(3);

      var elSteps = $('val-loss-steps');
      if (elSteps) elSteps.textContent = 'Step: ' + state.particles[0].stepCount;
    }

    // 5. Compute Hessian Matrix (∇²L) at Leading Optimizer Position (Adam or Origin)
    var probeP = state.particles[3] || state.particles[0];
    var probeX = probeP ? probeP.x : 0;
    var probeY = probeP ? probeP.y : 0;
    var H = lossFn.hessian ? lossFn.hessian(probeX, probeY) : new Matrix2x2(1, 0, 0, 1);

    var trH = H.trace();
    var detH = H.determinant();
    var discH = trH * trH - 4 * detH;
    var sqrtDisc = Math.sqrt(Math.max(0, discH));
    var lam1 = (trH + sqrtDisc) / 2;
    var lam2 = (trH - sqrtDisc) / 2;

    var abs1 = Math.abs(lam1), abs2 = Math.abs(lam2);
    var maxLam = Math.max(abs1, abs2);
    var minLam = Math.min(abs1, abs2);
    var condNum = minLam > 1e-4 ? maxLam / minLam : 999.0;

    var elHMat = $('val-hessian-matrix');
    var elHEigens = $('val-hessian-eigens');
    var elHCond = $('val-hessian-cond');
    var badgeHClass = $('badge-hessian-class');

    if (elHMat) {
      elHMat.innerHTML = '[ ' + H.a.toFixed(2) + ', ' + H.b.toFixed(2) + ' ]<br>[ ' + H.c.toFixed(2) + ', ' + H.d.toFixed(2) + ' ]';
    }
    if (elHEigens) {
      elHEigens.textContent = 'λ₁=' + lam1.toFixed(2) + ', λ₂=' + lam2.toFixed(2);
    }
    if (elHCond) {
      elHCond.textContent = condNum > 100 ? 'κ > 100 (High Anisotropy)' : 'κ = ' + condNum.toFixed(2) + (condNum > 10 ? ' (Ill-Conditioned Ravine)' : ' (Well-Conditioned)');
      elHCond.style.color = condNum > 10 ? '#ef4444' : '#10b981';
    }
    if (badgeHClass) {
      if (lam1 > 0 && lam2 > 0) {
        badgeHClass.textContent = 'Positive Definite (Minima)';
        badgeHClass.className = 'telemetry-badge badge-det-pos';
      } else if (lam1 < 0 && lam2 < 0) {
        badgeHClass.textContent = 'Negative Definite (Maxima)';
        badgeHClass.className = 'telemetry-badge badge-det-neg';
      } else if (lam1 * lam2 < 0) {
        badgeHClass.textContent = 'Indefinite (Saddle Point)';
        badgeHClass.className = 'telemetry-badge badge-det-zero';
      } else {
        badgeHClass.textContent = 'Degenerate Curvature';
        badgeHClass.className = 'telemetry-badge badge-det-zero';
      }
    }

    drawTopRightHUDTag('LossLab: ' + lossFn.name + ' (α=' + state.learningRate + ')');

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

  // ── Notes & Quiz Interactive Visual Blackboard ───────────────────────────

  function renderNotesCanvas() {
    var topicSelect = $('notes-topic-select');
    var topic = topicSelect ? topicSelect.value : 'matrix-transform';
    var o = worldToScreen(0, 0);

    drawTopRightHUDTag('Theory Vault: Interactive Concept Demonstration');
    drawAxes();

    if (topic === 'matrix-transform') {
      // Draw grid, transformed basis, and landing column indicators
      var m = state.matrix;
      drawTransformedGrid(m);
      drawTransformedShape(m);
      drawBasisVectors(m);
      var pI = worldToScreen(m.a, m.c);
      var pJ = worldToScreen(m.b, m.d);
      drawVectorLabel('Col 1: î lands at [' + m.a.toFixed(1) + ', ' + m.c.toFixed(1) + ']', pI.x, pI.y, '#10b981');
      drawVectorLabel('Col 2: ĵ lands at [' + m.b.toFixed(1) + ', ' + m.d.toFixed(1) + ']', pJ.x, pJ.y, '#8b5cf6');
    } else if (topic === 'determinant') {
      // Draw unit square to transformed parallelogram with signed area badge
      var m = state.matrix;
      drawTransformedShape(m);
      drawBasisVectors(m);
      var det = m.determinant();
      var center = worldToScreen((m.a + m.b) / 2, (m.c + m.d) / 2);
      drawVectorLabel('Area Factor |det(A)| = ' + Math.abs(det).toFixed(2), center.x, center.y, '#38bdf8');
      if (det < 0) {
        drawVectorLabel('⚠️ Orientation Flipped (det < 0)', center.x, center.y + 22, '#f43f5e');
      } else {
        drawVectorLabel('✅ Orientation Preserved (det > 0)', center.x, center.y + 22, '#10b981');
      }
    } else if (topic === 'eigenvalues') {
      // Draw invariant span lines and pulsating vector along invariant direction
      drawEigenSpanLines(state.matrix);
      drawBasisVectors(state.matrix);
      var eigens = Engine.solveEigensystem(state.matrix);
      if (eigens.isReal && eigens.eigenvectors.length > 0) {
        var v1 = eigens.eigenvectors[0].vector;
        var tScale = 1.2 + 0.6 * Math.sin(performance.now() / 350);
        var scaled = new Vector2D(v1.x * tScale, v1.y * tScale);
        var p = worldToScreen(scaled.x, scaled.y);
        drawArrow(o.x, o.y, p.x, p.y, '#eab308', 3);
        drawVectorLabel('Invariant Vector: Av = λv (Direction Never Rotates)', p.x, p.y, '#eab308');
      }
    } else if (topic === 'diagonalization') {
      // Draw eigenbasis factorization diagram
      drawEigenSpanLines(state.matrix);
      var eigens = Engine.solveEigensystem(state.matrix);
      if (eigens.isReal && eigens.eigenvectors.length >= 2) {
        var ev1 = eigens.eigenvectors[0].vector;
        var ev2 = eigens.eigenvectors[1].vector;
        var p1 = worldToScreen(ev1.x, ev1.y);
        var p2 = worldToScreen(ev2.x, ev2.y);
        drawArrow(o.x, o.y, p1.x, p1.y, '#eab308', 2.5);
        drawArrow(o.x, o.y, p2.x, p2.y, '#a855f7', 2.5);
        drawVectorLabel('Eigenbasis Axis 1 (λ₁ = ' + eigens.eigenvalues[0].value.toFixed(2) + ')', p1.x, p1.y, '#eab308');
        drawVectorLabel('Eigenbasis Axis 2 (λ₂ = ' + eigens.eigenvalues[1].value.toFixed(2) + ')', p2.x, p2.y, '#a855f7');
      }
    } else if (topic === 'svd') {
      // Draw SVD Unit Circle -> Transformed Ellipse
      state.shape = 'circle';
      drawTransformedShape(state.matrix);
      drawVectorLabel('SVD: Unit Circle → Transformed Hyper-Ellipse', o.x + 20, o.y - 120, '#38bdf8');
      drawVectorLabel('Semi-axes lengths = Singular Values σ₁, σ₂', o.x + 20, o.y - 95, '#10b981');
    } else if (topic === 'rank-nullity') {
      drawRankNullityLines(state.matrix);
      drawVectorLabel('Kernel ker(A): subspace compressed to 0', o.x - 140, o.y + 70, '#f43f5e');
      drawVectorLabel('Image im(A): column space 1D output range', o.x + 40, o.y - 70, '#10b981');
    } else if (topic === 'dot-product') {
      drawVectorSandbox();
    } else if (topic === 'gradient-descent' || topic === 'backpropagation') {
      renderLossLab();
    }
  }

  function renderQuizCanvas() {
    var o = worldToScreen(0, 0);
    drawTopRightHUDTag('Viva Question ' + (currentQuizIdx + 1) + ' of ' + QUIZ_QUESTIONS.length);
    drawAxes();

    // Render interactive pedagogical diagram tailored to the active question
    if (currentQuizIdx === 0) {
      // Question 1: Negative determinant (flipped space)
      var m = new Matrix2x2(-1.2, 0.4, 0.4, 1.2);
      drawTransformedShape(m);
      drawBasisVectors(m);
      drawVectorLabel('Mirrored Chirality: det(A) < 0 (Orientation Inverted)', o.x - 60, o.y - 90, '#f43f5e');
    } else if (currentQuizIdx === 1 || currentQuizIdx === 3) {
      // Question 2 or 4: Singular matrix / rank collapse
      var m = new Matrix2x2(1.5, 0.75, 1.0, 0.5);
      drawRankNullityLines(m);
      drawBasisVectors(m);
      drawVectorLabel('Dimension Squashed to 1D Line (det = 0, rank < 2)', o.x - 50, o.y - 80, '#ef4444');
    } else if (currentQuizIdx === 2) {
      // Question 3: Non-commutativity AB != BA
      var p1 = worldToScreen(state.matrix.a, state.matrix.c);
      var p2 = worldToScreen(state.matrixB.a, state.matrixB.c);
      drawArrow(o.x, o.y, p1.x, p1.y, '#38bdf8', 2.5);
      drawArrow(o.x, o.y, p2.x, p2.y, '#f59e0b', 2.5);
      drawVectorLabel('A·î', p1.x, p1.y, '#38bdf8');
      drawVectorLabel('B·î', p2.x, p2.y, '#f59e0b');
      drawVectorLabel('Order of transformations matters: AB ≠ BA', o.x - 80, o.y + 110, '#eab308');
    } else if (currentQuizIdx === 4) {
      // Question 5: Orthogonal vectors u . v = 0
      var uPos = worldToScreen(2.5, 0);
      var vPos = worldToScreen(0, 2.5);
      drawArrow(o.x, o.y, uPos.x, uPos.y, '#f43f5e', 2.5);
      drawArrow(o.x, o.y, vPos.x, vPos.y, '#06b6d4', 2.5);
      drawVectorLabel('u [2.5, 0]', uPos.x, uPos.y, '#f43f5e');
      drawVectorLabel('v [0, 2.5]', vPos.x, vPos.y, '#06b6d4');
      drawVectorLabel('Perpendicular: u · v = 0 (Angle θ = 90°)', o.x + 30, o.y - 60, '#10b981');
    } else {
      // Other questions: General transformation geometry
      drawTransformedShape(state.matrix);
      drawBasisVectors(state.matrix);
      drawEigenSpanLines(state.matrix);
      drawVectorLabel('Active Viva Concept Whiteboard', o.x - 60, o.y - 100, '#38bdf8');
    }
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
    ctx.font = '600 11px JetBrains Mono, monospace';

    var rad = typeof angle === 'number' ? angle : 0;
    var offX = Math.cos(rad) * 16;
    var offY = -Math.sin(rad) * 16;

    var posX = x + (offX >= 0 ? 10 : -10);
    var posY = y + (offY <= 0 ? -8 : 8);

    var metrics = ctx.measureText(text);
    var textWidth = metrics.width;
    var textHeight = 12;
    var padX = 5;
    var padY = 3;

    var boxX = offX >= 0 ? posX - padX : posX - textWidth - padX;
    var boxY = offY <= 0 ? posY - textHeight - padY : posY - padY;

    // Draw dark glass pill behind text to guarantee zero label/grid collisions
    ctx.fillStyle = 'rgba(10, 16, 30, 0.85)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(boxX, boxY, textWidth + padX * 2, textHeight + padY * 2, 4);
    } else {
      ctx.rect(boxX, boxY, textWidth + padX * 2, textHeight + padY * 2);
    }
    ctx.fill();
    ctx.stroke();

    // Draw text
    ctx.fillStyle = color;
    ctx.textAlign = offX >= 0 ? 'left' : 'right';
    ctx.textBaseline = offY <= 0 ? 'bottom' : 'top';
    ctx.fillText(text, posX, posY);
    ctx.restore();
  }

  // ── Number Display Safety Helpers ─────────────────────────────────────────

  // Format a number safely: cap extremely large values, show finite decimals
  function formatSafe(val, decimals) {
    if (typeof val !== 'number' || isNaN(val)) return '??';
    if (!isFinite(val)) return val > 0 ? '+∞' : '-∞';
    var d = typeof decimals === 'number' ? decimals : 2;
    if (Math.abs(val) > 1e6) return val.toExponential(2);
    return val.toFixed(d);
  }

  // ── Math Telemetry Updates ────────────────────────────────────────────────

  function updateTelemetry() {
    var m = state.matrix;
    var det = m.determinant();
    var tr = m.trace();
    var rk = m.rank();

    // Guard: only write determinant / eigenvalue display if in a 2D-mode
    var in2DMode = (state.mode === 'transform' || state.mode === 'eigen' ||
                    state.mode === 'mult'    || state.mode === 'vectors');

    if (readoutDet) readoutDet.textContent = formatSafe(det);
    if (badgeDet) {
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
    }

    if (readoutTrace) readoutTrace.textContent = formatSafe(tr);
    if (readoutRank) readoutRank.textContent = rk;

    var eigens = Engine.solveEigensystem(m);
    if (eigenFormulaSub) eigenFormulaSub.textContent = eigens.equationString;
    if (eigenQuadExpanded) eigenQuadExpanded.textContent = eigens.discriminantString;
    if (eigenDiscVal) eigenDiscVal.textContent = 'Δ = ' + formatSafe(eigens.discriminant);

    if (eigens.isReal) {
      if (eigenRow1) eigenRow1.textContent = 'λ₁ = ' + formatSafe(eigens.eigenvalues[0].value);
      if (eigenRow2) eigenRow2.textContent = 'λ₂ = ' + formatSafe(eigens.eigenvalues[1].value);
      if (badgeDisc) {
        badgeDisc.textContent = '2 Distinct Real Eigenvalues';
        badgeDisc.className = 'telemetry-badge badge-det-pos';
      }
    } else {
      var re = formatSafe(eigens.eigenvalues[0].real);
      var im = formatSafe(eigens.eigenvalues[0].imag);
      if (eigenRow1) eigenRow1.textContent = 'λ₁ = ' + re + ' + ' + im + 'i';
      if (eigenRow2) eigenRow2.textContent = 'λ₂ = ' + re + ' - ' + im + 'i';
      if (badgeDisc) {
        badgeDisc.textContent = 'Complex Roots (Pure Rotation/Spiral)';
        badgeDisc.className = 'telemetry-badge badge-det-neg';
      }
    }

    var diagText = $('diag-status-text');
    if (diagText) {
      if (eigens.isReal && eigens.eigenvectors.length >= 2 && Math.abs(eigens.eigenvalues[0].value - eigens.eigenvalues[1].value) > Engine.EPSILON) {
        var ev1 = eigens.eigenvectors[0].vector;
        var ev2 = eigens.eigenvectors[1].vector;
        diagText.innerHTML = 'Diagonalizable over \u211d:<br><strong>P</strong> = [ ' + formatSafe(ev1.x) + ', ' + formatSafe(ev2.x) + ' ; ' + formatSafe(ev1.y) + ', ' + formatSafe(ev2.y) + ' ]<br><strong>D</strong> = diag(' + formatSafe(eigens.eigenvalues[0].value) + ', ' + formatSafe(eigens.eigenvalues[1].value) + ')<br>Powers: A\u1d4f = P\u00b7D\u1d4f\u00b7P\u207b\u00b9';
      } else if (!eigens.isReal) {
        diagText.textContent = 'Cannot be diagonalized over ℝ (no real eigenbasis). A represents a rotation/spiral.';
      } else {
        diagText.textContent = 'Defective or uniform scalar matrix.';
      }
    }

    updateMultComparison();

    // Show/hide Mathematical Invariants section based on mode
    var invariantsSection = document.querySelector('.sidebar-section.telemetry-always-visible');
    if (invariantsSection) {
      var show2D = state.mode === 'transform' || state.mode === 'eigen' ||
                   state.mode === 'mult'      || state.mode === 'vectors';
      invariantsSection.style.display = show2D ? '' : 'none';
    }

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

    // Complex Phase Portrait telemetry
    var phaseCard = $('complex-phase-card');
    if (phaseCard) {
      phaseCard.classList.toggle('hidden', eigens.isReal);
      if (!eigens.isReal && eigens.modulus) {
        var rVal = eigens.modulus;
        $('complex-modulus-text').innerHTML = 'Modulus r = √(α²+β²) = <strong>' + rVal.toFixed(2) + '</strong> (radial growth per step)';
        $('complex-phase-text').innerHTML = 'Rotation Phase θ = <strong>' + eigens.phaseDeg.toFixed(1) + '°</strong> per step';
        var stabEl = $('complex-stability-text');
        if (stabEl) {
          if (Math.abs(rVal - 1.0) < 0.05) {
            stabEl.innerHTML = 'Dynamics: <strong style="color:#6ee7b7;">Elliptic Center Orbit (Stable Periodicity)</strong>';
          } else if (rVal < 1.0) {
            stabEl.innerHTML = 'Dynamics: <strong style="color:#38bdf8;">Spiral Sink / Attractor (Contracting to Origin)</strong>';
          } else {
            stabEl.innerHTML = 'Dynamics: <strong style="color:#f87171;">Spiral Source / Repeller (Expanding Outwards)</strong>';
          }
        }
      }
    }

    // Invariant Identities (Trace & Determinant)
    var sumEl = $('eigen-sum-val');
    var prodEl = $('eigen-prod-val');
    if (sumEl && prodEl) {
      if (eigens.isReal) {
        var s = eigens.eigenvalues[0].value + eigens.eigenvalues[1].value;
        var p = eigens.eigenvalues[0].value * eigens.eigenvalues[1].value;
        sumEl.innerHTML = s.toFixed(2) + ' == ' + tr.toFixed(2) + ' ✓';
        prodEl.innerHTML = p.toFixed(2) + ' == ' + det.toFixed(2) + ' ✓';
      } else {
        var aReal = eigens.eigenvalues[0].real;
        var aImag = eigens.eigenvalues[0].imag;
        var sComplex = aReal * 2;
        var pComplex = aReal * aReal + aImag * aImag;
        sumEl.innerHTML = sComplex.toFixed(2) + ' == ' + tr.toFixed(2) + ' ✓';
        prodEl.innerHTML = pComplex.toFixed(2) + ' == ' + det.toFixed(2) + ' ✓';
      }
    }

    // Rayleigh Quotient Telemetry
    var rayleighEl = $('rayleigh-val');
    var rayleighHint = $('rayleigh-error-hint');
    if (rayleighEl) {
      var rVal = m.rayleighQuotient(state.eigenProbe);
      rayleighEl.textContent = rVal.toFixed(3);
      if (rayleighHint) {
        if (eigens.isReal) {
          var domLambda = Math.max(eigens.eigenvalues[0].value, eigens.eigenvalues[1].value);
          var err = Math.abs(rVal - domLambda);
          rayleighHint.innerHTML = 'Dominant eigenvalue λ₁ = ' + domLambda.toFixed(2) + ' (Error: ' + err.toFixed(3) + ')';
        } else {
          rayleighHint.textContent = 'Quadratic form R(x) oscillating on complex spectrum';
        }
      }
    }

    // Gershgorin Discs Telemetry
    var g = m.gershgorinDiscs();
    var disc1El = $('gershgorin-disc1-text');
    var disc2El = $('gershgorin-disc2-text');
    if (disc1El && disc2El) {
      disc1El.innerHTML = '<strong>Disc 1:</strong> Center a = ' + m.a.toFixed(2) + ', Radius |b| = ' + Math.abs(m.b).toFixed(2);
      disc2El.innerHTML = '<strong>Disc 2:</strong> Center d = ' + m.d.toFixed(2) + ', Radius |c| = ' + Math.abs(m.c).toFixed(2);
    }

    // Update Custom Vector Output T(v) = Av
    var transVecEl = $('readout-trans-vec');
    if (transVecEl) {
      var vTrans = m.apply(state.customVec);
      transVecEl.textContent = '[ ' + formatSafe(vTrans.x) + ', ' + formatSafe(vTrans.y) + ' ]';
    }

    // Basis Angle & Orthogonality
    var basisAngleVal = $('basis-angle-val');
    var badgeOrtho = $('badge-basis-ortho');
    if (basisAngleVal && badgeOrtho) {
      var angleRad = m.basisAngle();
      var angleDeg = (angleRad * 180 / Math.PI).toFixed(1);
      basisAngleVal.textContent = angleDeg + '°';
      var isOrtho = m.isOrthogonalBasis();
      badgeOrtho.textContent = isOrtho ? 'Orthogonal (90°)' : 'Skew';
      badgeOrtho.className = isOrtho ? 'telemetry-badge badge-det-pos' : 'telemetry-badge';
    }

    // Linear System Solver (Ax = b => x = A^-1 b)
    var readoutPreimage = $('readout-preimage');
    var solverHint = $('solver-status-hint');
    if (readoutPreimage && solverHint) {
      if (Math.abs(det) < Engine.EPSILON) {
        readoutPreimage.textContent = 'Singular (No Unique Sol)';
        solverHint.innerHTML = '<span style="color:#f87171;">⚠️ Singular: det(A) = 0. Kernel dimension > 0; no unique preimage exists.</span>';
      } else {
        var xPre = m.solve(state.solveB);
        if (xPre) {
          readoutPreimage.textContent = '[ ' + formatSafe(xPre.x) + ', ' + formatSafe(xPre.y) + ' ]';
          solverHint.innerHTML = 'Unique solution verified: <span style="color:#38bdf8;">T(x) = Ax ≡ target b</span>.';
        }
      }
    }

    // Update Vector Sandbox Arithmetic
    var vecSum = state.vecU.add(state.vecV);
    var vecDiff = state.vecU.sub(state.vecV);
    var elSum = $('val-vec-sum');
    var elDiff = $('val-vec-diff');
    var elSumMag = $('val-vec-sum-mag');
    var elDiffMag = $('val-vec-diff-mag');
    if (elSum) elSum.textContent = '[ ' + formatSafe(vecSum.x) + ', ' + formatSafe(vecSum.y) + ' ]';
    if (elDiff) elDiff.textContent = '[ ' + formatSafe(vecDiff.x) + ', ' + formatSafe(vecDiff.y) + ' ]';
    if (elSumMag) elSumMag.textContent = '||u+v|| = ' + formatSafe(vecSum.magnitude());
    if (elDiffMag) elDiffMag.textContent = '||u-v|| = ' + formatSafe(vecDiff.magnitude());

    updateUrlHash();
  }

  function syncCustomVectorInputs() {
    var vxInput = $('vec-custom-x');
    var vyInput = $('vec-custom-y');
    if (vxInput && document.activeElement !== vxInput) vxInput.value = state.customVec.x.toFixed(2);
    if (vyInput && document.activeElement !== vyInput) vyInput.value = state.customVec.y.toFixed(2);
    var transVecEl = $('readout-trans-vec');
    if (transVecEl) {
      var vt = state.matrix.apply(state.customVec);
      transVecEl.textContent = '[ ' + formatSafe(vt.x) + ', ' + formatSafe(vt.y) + ' ]';
    }
  }

  function drawGhostShape(ghostMatrix, label) {
    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);

    var o = worldToScreen(0, 0);
    var iPos = worldToScreen(ghostMatrix.a, ghostMatrix.c);
    var jPos = worldToScreen(ghostMatrix.b, ghostMatrix.d);
    var sum = worldToScreen(ghostMatrix.a + ghostMatrix.b, ghostMatrix.c + ghostMatrix.d);

    ctx.beginPath();
    ctx.moveTo(o.x, o.y);
    ctx.lineTo(iPos.x, iPos.y);
    ctx.lineTo(sum.x, sum.y);
    ctx.lineTo(jPos.x, jPos.y);
    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = '#7dd3fc';
    ctx.font = '600 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
    ctx.textAlign = 'center';
    ctx.fillText(label, (o.x + sum.x) / 2, (o.y + sum.y) / 2 - 8);
    ctx.restore();
  }

  function updateMultComparison() {
    var AB = state.matrix.multiply(state.matrixB);
    var BA = state.matrixB.multiply(state.matrix);

    if (valProdAB) valProdAB.innerHTML = '[ ' + AB.a.toFixed(2) + ', ' + AB.b.toFixed(2) + ' ]<br>[ ' + AB.c.toFixed(2) + ', ' + AB.d.toFixed(2) + ' ]';
    if (valProdBA) valProdBA.innerHTML = '[ ' + BA.a.toFixed(2) + ', ' + BA.b.toFixed(2) + ' ]<br>[ ' + BA.c.toFixed(2) + ', ' + BA.d.toFixed(2) + ' ]';

    // Commutator [A, B] = AB - BA
    var comm = Matrix2x2.commutator(state.matrix, state.matrixB);
    var commNorm = Matrix2x2.commutatorNorm(state.matrix, state.matrixB);
    var badgeComm = $('badge-commutator');
    var valComm = $('val-commutator-matrix');

    if (badgeComm) {
      if (commNorm < 1e-4) {
        badgeComm.textContent = 'Commutative: [A, B] = 0';
        badgeComm.className = 'telemetry-badge badge-det-pos';
      } else {
        badgeComm.textContent = 'AB ≠ BA (Norm: ' + commNorm.toFixed(2) + ')';
        badgeComm.className = 'telemetry-badge badge-det-neg';
      }
    }
    if (valComm) {
      valComm.textContent = '[ ' + comm.a.toFixed(2) + ', ' + comm.b.toFixed(2) + ' ; ' + comm.c.toFixed(2) + ', ' + comm.d.toFixed(2) + ' ]';
    }

    // Determinant Multiplicative Law: det(AB) = det(A) * det(B)
    var detA = state.matrix.determinant();
    var detB = state.matrixB.determinant();
    var detAB = AB.determinant();
    var factorsEl = $('det-mult-factors');
    var productEl = $('det-mult-product');
    if (factorsEl) factorsEl.textContent = formatSafe(detA) + ' × ' + formatSafe(detB);
    if (productEl) productEl.innerHTML = formatSafe(detAB) + ' <span style="color:#6ee7b7;">(=' + (detA * detB).toFixed(2) + ') ✓</span>';

    // Bottom drawer labels
    var badgeA = $('mult-matrix-a-badge');
    var badgeB = $('mult-matrix-b-badge');
    if (badgeA) badgeA.textContent = 'A [' + state.matrix.a.toFixed(1) + ', ' + state.matrix.c.toFixed(1) + ']';
    if (badgeB) badgeB.textContent = 'B [' + state.matrixB.a.toFixed(1) + ', ' + state.matrixB.c.toFixed(1) + ']';

    // Sync dual matrix inputs in panel-mult
    var maA = $('mat-mult-a-a'), maB = $('mat-mult-a-b'), maC = $('mat-mult-a-c'), maD = $('mat-mult-a-d');
    if (maA) maA.value = state.matrix.a.toFixed(2);
    if (maB) maB.value = state.matrix.b.toFixed(2);
    if (maC) maC.value = state.matrix.c.toFixed(2);
    if (maD) maD.value = state.matrix.d.toFixed(2);
  }

  function syncMatrixInputs() {
    matAInput.value = state.matrix.a.toFixed(2);
    matBInput.value = state.matrix.b.toFixed(2);
    matCInput.value = state.matrix.c.toFixed(2);
    matDInput.value = state.matrix.d.toFixed(2);

    var maA = $('mat-mult-a-a'), maB = $('mat-mult-a-b'), maC = $('mat-mult-a-c'), maD = $('mat-mult-a-d');
    if (maA) maA.value = state.matrix.a.toFixed(2);
    if (maB) maB.value = state.matrix.b.toFixed(2);
    if (maC) maC.value = state.matrix.c.toFixed(2);
    if (maD) maD.value = state.matrix.d.toFixed(2);
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
    updateVectorSandboxTelemetry();
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

    if (state.mode === 'transform' && state.showSolver) {
      var bPos = worldToScreen(state.solveB.x, state.solveB.y);
      if (Math.hypot(sx - bPos.x, sy - bPos.y) < threshold) return 'solveB';
    }

    return null;
  }

  function syncSolverInputs() {
    var bx = $('solve-b-x');
    var by = $('solve-b-y');
    if (bx) bx.value = state.solveB.x.toFixed(2);
    if (by) by.value = state.solveB.y.toFixed(2);
  }

  function onPointerDown(e) {
    var pos = getMousePos(e);
    var hit = checkHitTarget(pos.x, pos.y);

    if (state.mode === '3d') {
      state.draggingTarget = '3d_orbit';
      state.dragStartMouse = pos;
      return;
    }

    if (state.mode === 'loss') {
      var w = screenToWorld(pos.x, pos.y);
      initLossParticles(w.x, w.y);
      render();
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
      state.customVec.x = sx; state.customVec.y = sy;
      syncCustomVectorInputs();
      updateTelemetry();
      render();
    } else if (state.draggingTarget === 'solveB') {
      state.solveB.x = sx; state.solveB.y = sy;
      syncSolverInputs();
      updateTelemetry();
      render();
    } else if (state.draggingTarget === 'probe') {
      var ang = Math.atan2(world.y, world.x);
      if (state.magneticSnap) {
        var eigens = Engine.solveEigensystem(state.matrix);
        if (eigens.isReal && eigens.eigenvectors.length > 0) {
          for (var evIdx = 0; evIdx < eigens.eigenvectors.length; evIdx++) {
            var ev = eigens.eigenvectors[evIdx].vector;
            var tAng = Math.atan2(ev.y, ev.x);
            var diffs = [
              Math.abs(ang - tAng),
              Math.abs(ang - (tAng + Math.PI)),
              Math.abs(ang - (tAng - Math.PI)),
              Math.abs(ang - (tAng + 2 * Math.PI)),
              Math.abs(ang - (tAng - 2 * Math.PI))
            ];
            var minDiff = Math.min.apply(null, diffs);
            if (minDiff < (4.5 * Math.PI) / 180) {
              ang = tAng; // Snap magnetically!
              break;
            }
          }
        }
      }
      state.eigenProbe = new Vector2D(Math.cos(ang), Math.sin(ang));
      var deg = Math.round(((ang * 180) / Math.PI + 360) % 360);
      var sAngle = $('slider-probe-angle');
      var vAngle = $('val-probe-angle');
      if (sAngle) sAngle.value = deg;
      if (vAngle) vAngle.textContent = deg + '°';
      updateTelemetry();
      render();
    } else if (state.draggingTarget === 'u') {
      if (state.snapToGrid) {
        var rx = Math.round(sx * 2) / 2;
        var ry = Math.round(sy * 2) / 2;
        if (Math.hypot(sx - rx, sy - ry) < 0.2) { sx = rx; sy = ry; }
      }
      state.vecU.x = sx; state.vecU.y = sy;
      vecUXInput.value = sx.toFixed(1); vecUYInput.value = sy.toFixed(1);
      updateVectorSandboxTelemetry();
      render();
    } else if (state.draggingTarget === 'v_sandbox') {
      if (state.snapToGrid) {
        var rx = Math.round(sx * 2) / 2;
        var ry = Math.round(sy * 2) / 2;
        if (Math.hypot(sx - rx, sy - ry) < 0.2) { sx = rx; sy = ry; }
      }
      state.vecV.x = sx; state.vecV.y = sy;
      vecVXInput.value = sx.toFixed(1); vecVYInput.value = sy.toFixed(1);
      updateVectorSandboxTelemetry();
      render();
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
      if (inp) inp.addEventListener('input', readMatrixBInputs);
    });

    // Dual Matrix A Inputs inside Panel Mult
    var maA = $('mat-mult-a-a'), maB = $('mat-mult-a-b'), maC = $('mat-mult-a-c'), maD = $('mat-mult-a-d');
    [maA, maB, maC, maD].forEach(function (inp) {
      if (inp) {
        inp.addEventListener('input', function () {
          state.matrix = new Matrix2x2(
            parseFloat(maA.value) || 0,
            parseFloat(maB.value) || 0,
            parseFloat(maC.value) || 0,
            parseFloat(maD.value) || 0
          );
          matAInput.value = state.matrix.a.toFixed(2);
          matBInput.value = state.matrix.b.toFixed(2);
          matCInput.value = state.matrix.c.toFixed(2);
          matDInput.value = state.matrix.d.toFixed(2);
          updateTelemetry();
          render();
        });
      }
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

    // Quick Matrix Tools (Transpose, Invert, Negate, Identity)
    var btnMatTranspose = $('btn-mat-transpose');
    if (btnMatTranspose) {
      btnMatTranspose.addEventListener('click', function () {
        state.matrix = state.matrix.transpose();
        syncMatrixInputs(); updateTelemetry(); render();
      });
    }

    var btnMatInverse = $('btn-mat-inverse');
    if (btnMatInverse) {
      btnMatInverse.addEventListener('click', function () {
        var inv = state.matrix.inverse();
        if (inv) {
          state.matrix = inv;
          syncMatrixInputs(); updateTelemetry(); render();
        } else {
          alert('Cannot invert singular matrix: det(A) = 0!');
        }
      });
    }

    var btnMatNegate = $('btn-mat-negate');
    if (btnMatNegate) {
      btnMatNegate.addEventListener('click', function () {
        state.matrix = new Matrix2x2(-state.matrix.a, -state.matrix.b, -state.matrix.c, -state.matrix.d);
        syncMatrixInputs(); updateTelemetry(); render();
      });
    }

    var btnMatIdentity = $('btn-mat-identity');
    if (btnMatIdentity) {
      btnMatIdentity.addEventListener('click', function () {
        applyPreset('identity');
      });
    }

    // Morph Timeline Controls
    var sliderMorph = $('slider-morph');
    var valMorph = $('val-morph');
    var btnMorphPlay = $('btn-morph-play');
    var morphTimer = null;

    if (sliderMorph) {
      sliderMorph.addEventListener('input', function () {
        state.morphT = parseFloat(this.value);
        if (valMorph) valMorph.textContent = Math.round(state.morphT * 100) + '%';
        render();
      });
    }

    if (btnMorphPlay) {
      btnMorphPlay.addEventListener('click', function () {
        state.morphPlaying = !state.morphPlaying;
        btnMorphPlay.textContent = state.morphPlaying ? '⏸ Pause' : '▶ Play';
        if (state.morphPlaying) {
          if (state.morphT >= 1.0) state.morphT = 0;
          var runMorph = function () {
            if (!state.morphPlaying) return;
            state.morphT += 0.012;
            if (state.morphT > 1.0) {
              state.morphT = 0.0;
            }
            if (sliderMorph) sliderMorph.value = state.morphT;
            if (valMorph) valMorph.textContent = Math.round(state.morphT * 100) + '%';
            render();
            morphTimer = requestAnimationFrame(runMorph);
          };
          morphTimer = requestAnimationFrame(runMorph);
        } else {
          if (morphTimer) cancelAnimationFrame(morphTimer);
        }
      });
    }

    // Linear System Solver Controls
    var solveBX = $('solve-b-x');
    var solveBY = $('solve-b-y');
    if (solveBX && solveBY) {
      var onSolveInput = function () {
        state.solveB.x = parseFloat(solveBX.value) || 0;
        state.solveB.y = parseFloat(solveBY.value) || 0;
        updateTelemetry();
        render();
      };
      solveBX.addEventListener('input', onSolveInput);
      solveBY.addEventListener('input', onSolveInput);
    }

    var btnToggleSolver = $('btn-toggle-solver');
    if (btnToggleSolver) {
      btnToggleSolver.addEventListener('click', function () {
        state.showSolver = !state.showSolver;
        this.textContent = state.showSolver ? 'Active' : 'Hidden';
        this.className = state.showSolver ? 'telemetry-badge badge-det-pos' : 'telemetry-badge badge-det-neg';
        render();
      });
    }

    // Matrix powers stepper — previews A^k without permanently mutating state.matrix
    document.querySelectorAll('.btn-power[data-power]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var p = parseInt(this.getAttribute('data-power'), 10);
        document.querySelectorAll('.btn-power').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        if (p === 1) {
          // Return to base matrix immediately
          powerAnimController.stop();
          state_basePowerMatrix = null;
          render();
          return;
        }
        // Animate from base matrix to A^p using the isolated power controller
        var base = state.matrix.clone();
        state_basePowerMatrix = base;
        var powered = base.power(p);
        powerAnimController.start(base, powered, 700);
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

    // 3D Camera Controls & Auto-Orbit
    var btnToggleOrbit3D = $('btn-toggle-orbit-3d');
    var orbit3DAnimId = null;
    if (btnToggleOrbit3D) {
      btnToggleOrbit3D.addEventListener('click', function () {
        state.autoOrbit3D = !state.autoOrbit3D;
        this.textContent = state.autoOrbit3D ? '⏹ Stop Orbit' : '▶ Auto Orbit';
        this.style.color = state.autoOrbit3D ? '#38bdf8' : '';
        var valCamStatus = $('val-cam-status');
        if (valCamStatus) valCamStatus.textContent = state.autoOrbit3D ? 'Auto Orbiting' : 'Manual Drag';

        if (state.autoOrbit3D) {
          function orbitTick() {
            if (!state.autoOrbit3D || state.mode !== '3d') {
              orbit3DAnimId = null;
              return;
            }
            state.camYaw = (state.camYaw + 0.45);
            if (state.camYaw > 180) state.camYaw -= 360;
            sliderYaw3D.value = Math.round(state.camYaw);
            valYaw3D.textContent = Math.round(state.camYaw) + '°';
            render();
            orbit3DAnimId = requestAnimationFrame(orbitTick);
          }
          orbit3DAnimId = requestAnimationFrame(orbitTick);
        } else if (orbit3DAnimId) {
          cancelAnimationFrame(orbit3DAnimId);
          orbit3DAnimId = null;
        }
      });
    }

    var btnToggleCamProj = $('btn-toggle-cam-proj');
    if (btnToggleCamProj) {
      btnToggleCamProj.addEventListener('click', function () {
        state.camOrthographic = !state.camOrthographic;
        this.textContent = state.camOrthographic ? 'Ortho' : 'Persp';
        this.style.color = state.camOrthographic ? '#10b981' : '';
        render();
      });
    }

    var btnResetCam3D = $('btn-reset-cam-3d');
    if (btnResetCam3D) {
      btnResetCam3D.addEventListener('click', function () {
        state.camYaw = 25;
        state.camPitch = 20;
        sliderYaw3D.value = 25;
        valYaw3D.textContent = '25°';
        sliderPitch3D.value = 20;
        valPitch3D.textContent = '20°';
        render();
      });
    }

    // 3D Presets
    document.querySelectorAll('.btn-preset[data-preset-3d], .btn-preset-3d').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var p = this.getAttribute('data-preset-3d');
        state.shearXY3D = 0;
        if (p === 'cube') {
          state.scaleX3D = 1.0; state.scaleY3D = 1.0; state.scaleZ3D = 1.0;
          state.rotX3D = 0; state.rotY3D = 0; state.rotZ3D = 0;
        } else if (p === 'sphere') {
          state.scaleX3D = 1.2; state.scaleY3D = 1.2; state.scaleZ3D = 1.2;
          state.rotX3D = 15; state.rotY3D = 30; state.rotZ3D = 0;
        } else if (p === 'shear3d') {
          state.scaleX3D = 1.0; state.scaleY3D = 1.0; state.scaleZ3D = 1.0;
          state.shearXY3D = 0.75;
        } else if (p === 'collapse3d') {
          state.scaleX3D = 1.2; state.scaleY3D = 1.2; state.scaleZ3D = 0.0;
        } else if (p === 'reflect3d') {
          state.scaleX3D = 1.0; state.scaleY3D = 1.0; state.scaleZ3D = -1.0;
        } else if (p === 'rot45') {
          state.rotX3D = 0; state.rotY3D = 45; state.rotZ3D = 0;
        }
        sliderScaleX.value = state.scaleX3D; $('val-scale-x').textContent = state.scaleX3D.toFixed(1);
        sliderScaleY.value = state.scaleY3D; $('val-scale-y').textContent = state.scaleY3D.toFixed(1);
        sliderScaleZ.value = Math.max(0, state.scaleZ3D); $('val-scale-z').textContent = state.scaleZ3D.toFixed(1);
        sliderRoll3D.value = state.rotZ3D; valRoll3D.textContent = state.rotZ3D + '°';
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

    var btnStepDescent = $('btn-step-descent');
    if (btnStepDescent) {
      btnStepDescent.addEventListener('click', function () {
        var lossFn = LossFunctions[state.lossKey] || LossFunctions.bowl;
        state.particles.forEach(function (p) {
          p.step(lossFn, state.learningRate);
        });
        render();
      });
    }

    var btnClearTrails = $('btn-clear-trails');
    if (btnClearTrails) {
      btnClearTrails.addEventListener('click', function () {
        state.particles.forEach(function (p) {
          p.history = [{ x: p.x, y: p.y }];
        });
        render();
      });
    }

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

    // Custom Vector (v) Inputs
    var vecCustomX = $('vec-custom-x');
    var vecCustomY = $('vec-custom-y');
    if (vecCustomX && vecCustomY) {
      var onCustomVecInput = function () {
        state.customVec.x = parseFloat(vecCustomX.value) || 0;
        state.customVec.y = parseFloat(vecCustomY.value) || 0;
        updateTelemetry();
        render();
      };
      vecCustomX.addEventListener('input', onCustomVecInput);
      vecCustomY.addEventListener('input', onCustomVecInput);
    }

    var btnToggleCustomVec = $('btn-toggle-custom-vec');
    if (btnToggleCustomVec) {
      btnToggleCustomVec.addEventListener('click', function () {
        state.showCustomVec = !state.showCustomVec;
        this.textContent = state.showCustomVec ? 'Visible' : 'Hidden';
        this.className = state.showCustomVec ? 'telemetry-badge badge-det-pos' : 'telemetry-badge badge-det-neg';
        render();
      });
    }

    document.querySelectorAll('.btn-vec-preset').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var vx = parseFloat(this.getAttribute('data-vx')) || 0;
        var vy = parseFloat(this.getAttribute('data-vy')) || 0;
        state.customVec.x = vx;
        state.customVec.y = vy;
        if (vecCustomX) vecCustomX.value = vx.toFixed(2);
        if (vecCustomY) vecCustomY.value = vy.toFixed(2);
        updateTelemetry();
        render();
      });
    });

    // Eigen Probe Slider & Locking Controls
    var sliderProbeAngle = $('slider-probe-angle');
    var valProbeAngle = $('val-probe-angle');
    if (sliderProbeAngle) {
      sliderProbeAngle.addEventListener('input', function () {
        var deg = parseFloat(this.value);
        if (valProbeAngle) valProbeAngle.textContent = deg + '°';
        var rad = (deg * Math.PI) / 180;
        state.eigenProbe = new Vector2D(Math.cos(rad), Math.sin(rad));
        render();
      });
    }

    var btnLockEigen1 = $('btn-lock-eigen1');
    if (btnLockEigen1) {
      btnLockEigen1.addEventListener('click', function () {
        var eigens = Engine.solveEigensystem(state.matrix);
        if (eigens.isReal && eigens.eigenvectors.length > 0) {
          var v1 = eigens.eigenvectors[0].vector;
          state.eigenProbe = new Vector2D(v1.x, v1.y).normalize();
          var deg = Math.round(((Math.atan2(v1.y, v1.x) * 180) / Math.PI + 360) % 360);
          if (sliderProbeAngle) sliderProbeAngle.value = deg;
          if (valProbeAngle) valProbeAngle.textContent = deg + '°';
          render();
        }
      });
    }

    var btnLockEigen2 = $('btn-lock-eigen2');
    if (btnLockEigen2) {
      btnLockEigen2.addEventListener('click', function () {
        var eigens = Engine.solveEigensystem(state.matrix);
        if (eigens.isReal && eigens.eigenvectors.length > 1) {
          var v2 = eigens.eigenvectors[1].vector;
          state.eigenProbe = new Vector2D(v2.x, v2.y).normalize();
          var deg = Math.round(((Math.atan2(v2.y, v2.x) * 180) / Math.PI + 360) % 360);
          if (sliderProbeAngle) sliderProbeAngle.value = deg;
          if (valProbeAngle) valProbeAngle.textContent = deg + '°';
          render();
        }
      });
    }

    var btnAutoScan = $('btn-auto-scan');
    if (btnAutoScan) {
      btnAutoScan.addEventListener('click', function () {
        var deg = 0;
        var timer = setInterval(function () {
          deg += 5;
          if (deg > 360) {
            clearInterval(timer);
            return;
          }
          if (sliderProbeAngle) sliderProbeAngle.value = deg % 360;
          if (valProbeAngle) valProbeAngle.textContent = (deg % 360) + '°';
          var rad = ((deg % 360) * Math.PI) / 180;
          state.eigenProbe = new Vector2D(Math.cos(rad), Math.sin(rad));
          render();
        }, 25);
      });
    }

    // Loop 2: Complex Spiral, Rayleigh Quotient, Magnetic Snap & Gershgorin Listeners
    var btnMagneticSnap = $('btn-magnetic-snap');
    if (btnMagneticSnap) {
      btnMagneticSnap.addEventListener('click', function () {
        state.magneticSnap = !state.magneticSnap;
        this.textContent = state.magneticSnap ? 'Magnetic Snap' : 'Free Sweep';
        this.className = state.magneticSnap ? 'telemetry-badge badge-det-pos' : 'telemetry-badge badge-det-neg';
      });
    }

    var btnStepPower = $('btn-step-power');
    if (btnStepPower) {
      btnStepPower.addEventListener('click', function () {
        var nextV = state.matrix.apply(state.eigenProbe);
        if (nextV.magnitude() > Engine.EPSILON) {
          state.eigenProbe = nextV.normalize();
          var ang = Math.atan2(state.eigenProbe.y, state.eigenProbe.x);
          var deg = Math.round(((ang * 180) / Math.PI + 360) % 360);
          if (sliderProbeAngle) sliderProbeAngle.value = deg;
          if (valProbeAngle) valProbeAngle.textContent = deg + '°';
          updateTelemetry();
          render();
        }
      });
    }

    var btnResetPower = $('btn-reset-power');
    if (btnResetPower) {
      btnResetPower.addEventListener('click', function () {
        state.eigenProbe = new Vector2D(1.0, 0.0);
        if (sliderProbeAngle) sliderProbeAngle.value = 0;
        if (valProbeAngle) valProbeAngle.textContent = '0°';
        updateTelemetry();
        render();
      });
    }

    var btnToggleGershgorin = $('btn-toggle-gershgorin');
    if (btnToggleGershgorin) {
      btnToggleGershgorin.addEventListener('click', function () {
        state.showGershgorin = !state.showGershgorin;
        this.textContent = state.showGershgorin ? 'Visible' : 'Hidden';
        this.className = state.showGershgorin ? 'telemetry-badge badge-det-pos' : 'telemetry-badge badge-det-neg';
        render();
      });
    }

    var btnToggleSpiral = $('btn-toggle-spiral');
    if (btnToggleSpiral) {
      btnToggleSpiral.addEventListener('click', function () {
        state.showSpiral = !state.showSpiral;
        this.textContent = state.showSpiral ? 'Show Orbit' : 'Hide Orbit';
        this.className = state.showSpiral ? 'telemetry-badge badge-det-pos' : 'telemetry-badge badge-det-neg';
        render();
      });
    }

    // Vector Sandbox Configuration Presets
    document.querySelectorAll('.btn-vec-config').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var cfg = this.getAttribute('data-vcfg');
        if (cfg === 'perpendicular') {
          state.vecV.x = -state.vecU.y;
          state.vecV.y = state.vecU.x;
        } else if (cfg === 'parallel') {
          state.vecV.x = state.vecU.x * 1.5;
          state.vecV.y = state.vecU.y * 1.5;
        } else if (cfg === 'opposite') {
          state.vecV.x = -state.vecU.x;
          state.vecV.y = -state.vecU.y;
        } else if (cfg === 'pythagoras') {
          state.vecU.x = 3.0; state.vecU.y = 0.0;
          state.vecV.x = 0.0; state.vecV.y = 4.0;
        } else if (cfg === 'unit') {
          state.vecU = state.vecU.normalize();
          state.vecV = state.vecV.normalize();
        } else if (cfg === 'swap') {
          var tmp = state.vecU.clone();
          state.vecU = state.vecV.clone();
          state.vecV = tmp;
        }
        vecUXInput.value = state.vecU.x.toFixed(1);
        vecUYInput.value = state.vecU.y.toFixed(1);
        vecVXInput.value = state.vecV.x.toFixed(1);
        vecVYInput.value = state.vecV.y.toFixed(1);
        updateVectorSandboxTelemetry();
        render();
      });
    });

    // Vector Sandbox quick helpers & controls
    var btnNormU = $('btn-normalize-u');
    if (btnNormU) {
      btnNormU.addEventListener('click', function () {
        state.vecU = state.vecU.normalize();
        vecUXInput.value = state.vecU.x.toFixed(1); vecUYInput.value = state.vecU.y.toFixed(1);
        updateVectorSandboxTelemetry();
        render();
      });
    }

    var btnNormV = $('btn-normalize-v');
    if (btnNormV) {
      btnNormV.addEventListener('click', function () {
        state.vecV = state.vecV.normalize();
        vecVXInput.value = state.vecV.x.toFixed(1); vecVYInput.value = state.vecV.y.toFixed(1);
        updateVectorSandboxTelemetry();
        render();
      });
    }

    var btnOrthoV = $('btn-orthogonalize-v');
    if (btnOrthoV) {
      btnOrthoV.addEventListener('click', function () {
        state.vecV = state.vecV.rejectFrom(state.vecU);
        vecVXInput.value = state.vecV.x.toFixed(1); vecVYInput.value = state.vecV.y.toFixed(1);
        updateVectorSandboxTelemetry();
        render();
      });
    }

    var btnApplyGS = $('btn-apply-gram-schmidt');
    if (btnApplyGS) {
      btnApplyGS.addEventListener('click', function () {
        state.vecU = state.vecU.normalize();
        var vPerp = state.vecV.rejectFrom(state.vecU);
        state.vecV = vPerp.normalize();
        vecUXInput.value = state.vecU.x.toFixed(1); vecUYInput.value = state.vecU.y.toFixed(1);
        vecVXInput.value = state.vecV.x.toFixed(1); vecVYInput.value = state.vecV.y.toFixed(1);
        updateVectorSandboxTelemetry();
        render();
      });
    }

    var btnToggleDecomp = $('btn-toggle-decomp');
    if (btnToggleDecomp) {
      btnToggleDecomp.addEventListener('click', function () {
        state.showDecomp = !state.showDecomp;
        this.classList.toggle('active', state.showDecomp);
        render();
      });
    }

    var btnToggleParallelogram = $('btn-toggle-parallelogram');
    if (btnToggleParallelogram) {
      btnToggleParallelogram.addEventListener('click', function () {
        state.showParallelogram = !state.showParallelogram;
        this.classList.toggle('active', state.showParallelogram);
        render();
      });
    }

    var btnToggleVSnap = $('btn-toggle-vsnap');
    if (btnToggleVSnap) {
      btnToggleVSnap.addEventListener('click', function () {
        state.snapToGrid = !state.snapToGrid;
        this.textContent = state.snapToGrid ? 'Snap: Grid' : 'Snap: Free';
        this.style.color = state.snapToGrid ? '#38bdf8' : 'var(--text-muted)';
      });
    }

    // Matrix multiplication drawer & Tab 3 features
    multSlider.addEventListener('input', function () {
      state.multT = parseFloat(this.value);
      multTDisplay.textContent = state.multT.toFixed(2);
      document.querySelectorAll('.btn-mult-step').forEach(function (b) {
        var s = parseFloat(b.getAttribute('data-step'));
        b.classList.toggle('active', Math.abs(s - state.multT) < 0.15);
      });
      render();
    });

    btnSwapMult.addEventListener('click', function () {
      state.multOrder = state.multOrder === 'AB' ? 'BA' : 'AB';
      btnSwapMult.textContent = 'Order: ' + state.multOrder;
      var step1Btn = document.querySelector('.btn-mult-step[data-step="0.5"]');
      var step2Btn = document.querySelector('.btn-mult-step[data-step="1.0"]');
      if (state.multOrder === 'AB') {
        if (step1Btn) step1Btn.textContent = '1: First Map (B)';
        if (step2Btn) step2Btn.textContent = '2: Composite (AB)';
      } else {
        if (step1Btn) step1Btn.textContent = '1: First Map (A)';
        if (step2Btn) step2Btn.textContent = '2: Composite (BA)';
      }
      updateTelemetry();
      render();
    });

    var btnMatBInvert = $('btn-mat-b-invert');
    if (btnMatBInvert) {
      btnMatBInvert.addEventListener('click', function () {
        var inv = state.matrixB.inverse();
        if (inv) {
          state.matrixB = inv;
          matBAInput.value = inv.a.toFixed(2);
          matBBInput.value = inv.b.toFixed(2);
          matBCInput.value = inv.c.toFixed(2);
          matBDInput.value = inv.d.toFixed(2);
          updateTelemetry();
          render();
        } else {
          alert('Cannot invert singular Matrix B: det(B) = 0!');
        }
      });
    }

    // Step jump buttons
    document.querySelectorAll('.btn-mult-step').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var step = parseFloat(this.getAttribute('data-step'));
        state.multT = step;
        multSlider.value = step;
        multTDisplay.textContent = step.toFixed(2);
        document.querySelectorAll('.btn-mult-step').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        render();
      });
    });

    // Preset pairs
    document.querySelectorAll('.btn-mult-pair').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var pair = this.getAttribute('data-pair');
        if (pair === 'rot-shear') {
          state.matrix = Matrix2x2.shear(0.6, 0);
          state.matrixB = Matrix2x2.rotation(Math.PI / 4);
        } else if (pair === 'commuting-rot') {
          state.matrix = Matrix2x2.rotation(Math.PI / 6);
          state.matrixB = Matrix2x2.rotation(Math.PI / 3);
        } else if (pair === 'inverse-pair') {
          state.matrix = new Matrix2x2(1.2, 0.4, 0.2, 1.0);
          state.matrixB = state.matrix.inverse() || Matrix2x2.identity();
        } else if (pair === 'scale-reflect') {
          state.matrix = Matrix2x2.scaling(1.5, 0.7);
          state.matrixB = Matrix2x2.reflection(Math.PI / 4);
        }
        syncMatrixInputs();
        matBAInput.value = state.matrixB.a.toFixed(2);
        matBBInput.value = state.matrixB.b.toFixed(2);
        matBCInput.value = state.matrixB.c.toFixed(2);
        matBDInput.value = state.matrixB.d.toFixed(2);
        updateTelemetry();
        render();
      });
    });

    // Run sequence animation
    var multAnimId = null;
    var btnRunMultSeq = $('btn-run-mult-sequence');
    if (btnRunMultSeq) {
      btnRunMultSeq.addEventListener('click', function () {
        if (multAnimId) {
          cancelAnimationFrame(multAnimId);
          multAnimId = null;
          btnRunMultSeq.textContent = '▶ Play Sequence';
          return;
        }
        state.multT = 0;
        multSlider.value = 0;
        multTDisplay.textContent = '0.00';
        btnRunMultSeq.textContent = '⏹ Stop Sequence';
        var startTime = performance.now();
        var duration = 2400; // 2.4 seconds
        function stepSeq(now) {
          var elapsed = now - startTime;
          var p = Math.min(1, elapsed / duration);
          state.multT = p;
          multSlider.value = p;
          multTDisplay.textContent = p.toFixed(2);
          document.querySelectorAll('.btn-mult-step').forEach(function (b) {
            var s = parseFloat(b.getAttribute('data-step'));
            b.classList.toggle('active', Math.abs(s - p) < 0.2);
          });
          render();
          if (p < 1) {
            multAnimId = requestAnimationFrame(stepSeq);
          } else {
            multAnimId = null;
            btnRunMultSeq.textContent = '▶ Play Sequence';
          }
        }
        multAnimId = requestAnimationFrame(stepSeq);
      });
    }

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

  function updateHUDTip(mode) {
    var tipEl = $('hud-tip');
    if (!tipEl) return;
    var tips = {
      transform: 'Tip: <strong>Drag arrows</strong>, hold <strong>Shift</strong> to snap, or tweak vector v in sidebar',
      eigen: 'Tip: <strong>Sweep Probe Angle</strong> or click <strong>Auto-Scan</strong> to discover invariant lines',
      mult: 'Tip: Scrub the <strong>Interpolation Slider (t)</strong> to inspect transformation ordering (AB ≠ BA)',
      vectors: 'Tip: Drag <strong>vectors u & v</strong> or click presets for orthogonal decomposition',
      '3d': 'Tip: <strong>Click & drag on canvas</strong> to orbit camera 360°, or tweak Euler sliders',
      loss: 'Tip: Select a loss surface and click <strong>▶ Run Race</strong> to compare gradient descent algorithms',
      autograd: 'Tip: Click <strong>▶ Forward Pass</strong> then <strong>◀ Backprop</strong> to trace reverse-mode gradients',
      notes: 'Tip: Select any <strong>FY Engineering Topic</strong> on the left to inspect illustrated interactive theory',
      quiz: 'Tip: Pick an option on the left and click <strong>Next Question</strong> to test active recall'
    };
    tipEl.innerHTML = tips[mode] || tips.transform;
  }

  function setMode(newMode) {
    state.mode = newMode;

    // Toggle panels in sidebar
    var panels = ['transform', 'eigen', 'mult', 'vectors', '3d', 'loss', 'autograd', 'notes', 'quiz'];
    panels.forEach(function (p) {
      var el = $('panel-' + p);
      if (el) {
        el.classList.toggle('hidden', newMode !== p);
        el.classList.toggle('active', newMode === p);
      }
    });

    // Update active mode buttons
    document.querySelectorAll('.mode-btn').forEach(function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-mode') === newMode);
    });

    multDrawer.classList.toggle('active', newMode === 'mult');

    updateHUDTip(newMode);
    updateTelemetry();

    if (newMode === 'loss') {
      initLossParticles();
    } else if (newMode === 'notes') {
      var sel = $('notes-topic-select');
      if (sel) updateNotesTopic(sel.value);
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
      // Safety guard: don't encode astronomically large or infinite values
      var vals = [m.a, m.b, m.c, m.d];
      var allSafe = vals.every(function (v) {
        return isFinite(v) && Math.abs(v) <= 20;
      });
      if (!allSafe) return;

      var hash = 'a=' + m.a.toFixed(2) + '&b=' + m.b.toFixed(2) + '&c=' + m.c.toFixed(2) + '&d=' + m.d.toFixed(2) + '&m=' + state.mode;
      window.history.replaceState(null, '', '#' + hash);
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
      var a = parseFloat(params.a);
      var b = parseFloat(params.b);
      var c = parseFloat(params.c);
      var d = parseFloat(params.d);

      // Sanity check: clamp to ±20 to prevent astronomical values from being restored
      var MAX_SAFE = 20;
      var isValid = [a, b, c, d].every(function (v) {
        return !isNaN(v) && isFinite(v) && Math.abs(v) <= MAX_SAFE;
      });

      if (isValid) {
        state.matrix = new Matrix2x2(a, b, c, d);
      } else {
        // Clear the bad hash so it stops poisoning reloads
        window.history.replaceState(null, '', window.location.pathname);
        console.warn('TensorForge: Corrupt URL hash discarded (values out of safe range).');
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
