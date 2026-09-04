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
    autogradTapeStep: -1,
    hoverAutogradNode: null,
    ag_x1: 1.5,
    ag_w1: 0.8,
    ag_x2: -1.0,
    ag_w2: 1.2,
    ag_b: 0.3,
    ag_yPred: 0.85,
    ag_yTrue: 0.50,
    ag_aff_x1: 1.0,
    ag_aff_x2: -0.5,
    ag_aff_w11: 1.2,
    ag_aff_w12: -0.6,
    ag_aff_w21: 0.4,
    ag_aff_w22: 0.9,
    ag_aff_b1: 0.2,
    ag_aff_b2: -0.3,

    // Theory Vault Notes State
    currentNotesTopic: 'matrix-transform',
    exploredNotesTopics: ['matrix-transform'],

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
  var btnStepTape = $('btn-step-tape');
  var btnResetAutograd = $('btn-reset-autograd');
  var autogradStatusText = $('autograd-status-text');
  var autogradFormulaTitle = $('autograd-formula-title');
  var badgeAgStep = $('badge-ag-step');
  var valAgJacobianMatrix = $('val-ag-jacobian-matrix');
  var valAgOutputVec = $('val-ag-output-vec');
  var currentAutogradNodes = [];

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
    beginLabelPass();

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
        if (state.showCustomVec && state.mode === 'transform') {
          drawCustomVector(activeMatrix);
        }
        if (state.showSolver && state.mode === 'transform') {
          drawLinearSolver(activeMatrix);
        }
      }
    }

    flushVectorLabels();
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
        drawVectorLabel('Area: ' + Math.abs(det).toFixed(2), cx, cy, det > 0 ? '#67e8f9' : '#fcd34d', 0, 10);
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
      ctx.restore();

      var labelDist = 3.6;
      var labelPos = worldToScreen(v.x * labelDist, v.y * labelDist);
      var slope = Math.abs(v.x) > 1e-4 ? (v.y / v.x).toFixed(2) : '∞';
      var text = 'Span(v' + (index + 1) + ') λ=' + ev.lambda.toFixed(2) + ' [y=' + slope + 'x]';
      var angle = Math.atan2(v.y, v.x);
      drawVectorLabel(text, labelPos.x, labelPos.y, colors[index % colors.length], angle, 12);
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

      var labelPos = worldToScreen(nullVec.x * 3.4, nullVec.y * 3.4);
      var angNull = Math.atan2(nullVec.y, nullVec.x);
      drawVectorLabel('Kernel / Nullspace (Ax = 0)', labelPos.x, labelPos.y, '#ef4444', angNull, 12);
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

      var colLabelPos = worldToScreen(colVec.x * 3.8, colVec.y * 3.8);
      var angCol = Math.atan2(colVec.y, colVec.x);
      drawVectorLabel('Column Space / Range im(A)', colLabelPos.x, colLabelPos.y, '#38bdf8', angCol, 12);
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
    // 0. Ground Reference Grid Plane (Perspective Depth Grounding)
    drawGroundGrid3D(pitchRad, yawRad, fov, originX, originY);

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

      drawVectorLabel(tr.name + ' [' + tr.v.x.toFixed(1) + ',' + tr.v.y.toFixed(1) + ',' + tr.v.z.toFixed(1) + ']', sx, sy, tr.col, 0, 6);
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

    // 4. Cube Corner Vertex Nodes
    projectedPts.forEach(function (pt) {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = det3D >= 0 ? '#c7d2fe' : '#fde68a';
      ctx.fill();
    });

    drawOrientationGizmo3D(pitchRad, yawRad);
  }

  function drawGroundGrid3D(pitchRad, yawRad, fov, originX, originY) {
    var groundY = -2.2;
    var range = 3.6;
    var step = 0.9;
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;

    for (var gx = -range; gx <= range + 1e-4; gx += step) {
      var pStart = project3DTo2D(new Vector3D(gx, groundY, -range), pitchRad, yawRad, fov, state.camOrthographic);
      var pEnd = project3DTo2D(new Vector3D(gx, groundY, range), pitchRad, yawRad, fov, state.camOrthographic);
      ctx.beginPath();
      ctx.moveTo(originX + pStart.x, originY + pStart.y);
      ctx.lineTo(originX + pEnd.x, originY + pEnd.y);
      ctx.stroke();
    }
    for (var gz = -range; gz <= range + 1e-4; gz += step) {
      var pStart = project3DTo2D(new Vector3D(-range, groundY, gz), pitchRad, yawRad, fov, state.camOrthographic);
      var pEnd = project3DTo2D(new Vector3D(range, groundY, gz), pitchRad, yawRad, fov, state.camOrthographic);
      ctx.beginPath();
      ctx.moveTo(originX + pStart.x, originY + pStart.y);
      ctx.lineTo(originX + pEnd.x, originY + pEnd.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawOrientationGizmo3D(pitchRad, yawRad) {
    var gx = 52;
    var gy = viewHeight - 52;
    var r = 26;

    ctx.save();
    ctx.beginPath();
    ctx.arc(gx, gy, r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(11, 17, 32, 0.85)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();

    var axes = [
      { v: new Vector3D(1, 0, 0), col: '#f43f5e', name: 'X' },
      { v: new Vector3D(0, 1, 0), col: '#10b981', name: 'Y' },
      { v: new Vector3D(0, 0, 1), col: '#06b6d4', name: 'Z' }
    ];

    axes.forEach(function (ax) {
      var p = project3DTo2D(ax.v, pitchRad, yawRad, 1.0, true);
      var tx = gx + p.x * 18;
      var ty = gy + p.y * 18;
      ctx.beginPath();
      ctx.moveTo(gx, gy);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = ax.col;
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = ax.col;
      ctx.font = '700 9px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(ax.name, tx + p.x * 6, ty + p.y * 6);
    });
    ctx.restore();
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

    drawLossFieldAndContours(lossFn);
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

      // Particle Ball with Luminous Glow
      var curSp = worldToScreen(p.x, p.y);
      ctx.save();
      ctx.beginPath();
      ctx.arc(curSp.x, curSp.y, 6.5, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.restore();
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
      elHCond.textContent = condNum > 100 ? '> 100 (Anisotropic)' : condNum.toFixed(2) + (condNum > 10 ? ' (Ill-Cond)' : ' (Well-Cond)');
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

    if (state.lossRunning) {
      requestAnimationFrame(render);
    }
  }

  function drawLossFieldAndContours(lossFn) {
    // 1. Smooth energy field
    var gridStep = 16;
    ctx.save();
    for (var px = 0; px < viewWidth; px += gridStep) {
      for (var py = 0; py < viewHeight; py += gridStep) {
        var w = screenToWorld(px + gridStep / 2, py + gridStep / 2);
        var z = lossFn.evaluate(w.x, w.y);
        var intensity = Math.min(1, Math.max(0, Math.log(Math.abs(z) + 1) / 3.6));
        var rCol = Math.round(intensity * 140 + 12);
        var gCol = Math.round(intensity * 25 + 8);
        var bCol = Math.round((1 - intensity) * 160 + 35);
        ctx.fillStyle = 'rgba(' + rCol + ', ' + gCol + ', ' + bCol + ', 0.22)';
        ctx.fillRect(px, py, gridStep, gridStep);
      }
    }
    ctx.restore();

    // 2. Mathematically exact equipotential contour rings
    ctx.save();
    if (state.lossKey === 'bowl') {
      var levels = [0.15, 0.4, 0.8, 1.5, 2.8, 4.5, 7.0, 10.5, 15.0];
      levels.forEach(function (c, idx) {
        var rx = Math.sqrt(2 * c) * state.scale;
        var ry = Math.sqrt(c) * state.scale;
        var o = worldToScreen(0, 0);
        ctx.beginPath();
        ctx.ellipse(o.x, o.y, rx, ry, 0, 0, Math.PI * 2);
        var alpha = 0.15 + 0.04 * (idx % 3);
        ctx.strokeStyle = 'rgba(56, 189, 248, ' + alpha + ')';
        ctx.lineWidth = idx === 2 ? 1.5 : 1;
        ctx.stroke();
      });
    } else if (state.lossKey === 'rosenbrock') {
      ctx.beginPath();
      for (var vx = -2.5; vx <= 2.5; vx += 0.05) {
        var vy = vx * vx;
        var sp = worldToScreen(vx, vy);
        if (vx === -2.5) ctx.moveTo(sp.x, sp.y);
        else ctx.lineTo(sp.x, sp.y);
      }
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();

    // 3. Draw Global Optimum Bullseye Marker
    var opt = { x: 0, y: 0, label: '(0, 0)' };
    if (state.lossKey === 'rosenbrock') opt = { x: 1.0, y: 1.0, label: '(1, 1)' };
    else if (state.lossKey === 'beale') opt = { x: 3.0, y: 0.5, label: '(3, 0.5)' };
    else if (state.lossKey === 'saddle') opt = { x: 0, y: 0, label: 'Saddle (0, 0)' };

    var osp = worldToScreen(opt.x, opt.y);
    ctx.save();
    ctx.beginPath();
    ctx.arc(osp.x, osp.y, 12, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(250, 204, 21, 0.55)';
    ctx.lineWidth = 1.4;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.arc(osp.x, osp.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#facc15';
    ctx.shadowColor = 'rgba(250, 204, 21, 0.8)';
    ctx.shadowBlur = 8;
    ctx.fill();

    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(osp.x - 7, osp.y); ctx.lineTo(osp.x + 7, osp.y);
    ctx.moveTo(osp.x, osp.y - 7); ctx.lineTo(osp.x, osp.y + 7);
    ctx.stroke();

    drawVectorLabel('Min ' + opt.label, osp.x, osp.y, '#facc15', Math.PI / 4, 8);
    ctx.restore();
  }

  // ── MicroGraph Autograd DAG Renderer (Mode 7) ─────────────────────────────

  function renderAutogradGraph() {
    var W = viewWidth || canvas.width;
    var H = viewHeight || canvas.height;
    var nodes = [];
    var edges = [];
    var reverseTape = [];
    var bw = 96, bh = 54;

    var isForward = state.autogradStep === 'forward' || state.autogradStep === 'backward';
    var isBackward = state.autogradStep === 'backward';

    if (state.autogradPreset === 'neuron') {
      var x1 = state.ag_x1, w1 = state.ag_w1;
      var x2 = state.ag_x2, w2 = state.ag_w2;
      var b = state.ag_b;

      var p1 = x1 * w1;
      var p2 = x2 * w2;
      var sum = p1 + p2 + b;
      var reluOut = sum > 0 ? sum : 0;

      // Backward gradients via chain rule
      var dRelu = 1.0;
      var dSum = sum > 0 ? 1.0 : 0.0;
      var dP1 = dSum * 1.0;
      var dP2 = dSum * 1.0;
      var dB = dSum * 1.0;
      var dW1 = dP1 * x1;
      var dX1 = dP1 * w1;
      var dW2 = dP2 * x2;
      var dX2 = dP2 * w2;

      // Node layout coordinates
      var col1 = W * 0.15;
      var col2 = W * 0.40;
      var col3 = W * 0.65;
      var col4 = W * 0.88;

      nodes = [
        { label: 'x₁', val: x1, grad: dX1, x: col1, y: H * 0.18, type: 'in', desc: 'Input Feature 1' },
        { label: 'w₁', val: w1, grad: dW1, x: col1, y: H * 0.36, type: 'param', desc: 'Weight 1' },
        { label: 'x₂', val: x2, grad: dX2, x: col1, y: H * 0.64, type: 'in', desc: 'Input Feature 2' },
        { label: 'w₂', val: w2, grad: dW2, x: col1, y: H * 0.82, type: 'param', desc: 'Weight 2' },
        { label: 'w₁·x₁', val: p1, grad: dP1, x: col2, y: H * 0.27, type: 'op', desc: 'Product 1' },
        { label: 'w₂·x₂', val: p2, grad: dP2, x: col2, y: H * 0.73, type: 'op', desc: 'Product 2' },
        { label: 'b', val: b, grad: dB, x: col2, y: H * 0.90, type: 'param', desc: 'Neuron Bias' },
        { label: 'Σ (+)', val: sum, grad: dSum, x: col3, y: H * 0.50, type: 'op', desc: 'Affine Sum (z)' },
        { label: 'ReLU', val: reluOut, grad: dRelu, x: col4, y: H * 0.50, type: 'out', desc: 'Activation Output (a)' }
      ];

      edges = [
        { from: 0, to: 4, local: '×' + w1.toFixed(1), dOut: dX1 },
        { from: 1, to: 4, local: '×' + x1.toFixed(1), dOut: dW1 },
        { from: 2, to: 5, local: '×' + w2.toFixed(1), dOut: dX2 },
        { from: 3, to: 5, local: '×' + x2.toFixed(1), dOut: dW2 },
        { from: 4, to: 7, local: '+1.0', dOut: dP1 },
        { from: 5, to: 7, local: '+1.0', dOut: dP2 },
        { from: 6, to: 7, local: '+1.0', dOut: dB },
        { from: 7, to: 8, local: sum > 0 ? '1 (act)' : '0 (kill)', dOut: dSum }
      ];

      reverseTape = [8, 7, 6, 5, 4, 3, 2, 1, 0];

    } else if (state.autogradPreset === 'loss') {
      var yPred = state.ag_yPred;
      var yTrue = state.ag_yTrue;
      var diff = yPred - yTrue;
      var mse = diff * diff;

      // Backward
      var dMse = 1.0;
      var dDiff = 2 * diff;
      var dYPred = dDiff * 1.0;
      var dYTrue = dDiff * -1.0;

      var col1 = W * 0.20;
      var col2 = W * 0.52;
      var col3 = W * 0.84;

      nodes = [
        { label: 'y_pred', val: yPred, grad: dYPred, x: col1, y: H * 0.35, type: 'param', desc: 'Model Prediction' },
        { label: 'y_true', val: yTrue, grad: dYTrue, x: col1, y: H * 0.65, type: 'in', desc: 'Ground Truth Label' },
        { label: 'diff (-)', val: diff, grad: dDiff, x: col2, y: H * 0.50, type: 'op', desc: 'Residual Error (y_pred - y_true)' },
        { label: 'MSE (²)', val: mse, grad: dMse, x: col3, y: H * 0.50, type: 'out', desc: 'Squared Error Loss L' }
      ];

      edges = [
        { from: 0, to: 2, local: '+1.0', dOut: dYPred },
        { from: 1, to: 2, local: '-1.0', dOut: dYTrue },
        { from: 2, to: 3, local: '2·diff (' + (2 * diff).toFixed(2) + ')', dOut: dDiff }
      ];

      reverseTape = [3, 2, 0, 1];

    } else if (state.autogradPreset === 'affine') {
      var ax1 = state.ag_aff_x1, ax2 = state.ag_aff_x2;
      var W11 = state.ag_aff_w11, W12 = state.ag_aff_w12;
      var W21 = state.ag_aff_w21, W22 = state.ag_aff_w22;
      var ab1 = state.ag_aff_b1, ab2 = state.ag_aff_b2;

      var y1 = W11 * ax1 + W12 * ax2 + ab1;
      var y2 = W21 * ax1 + W22 * ax2 + ab2;
      var loss = 0.5 * (y1 * y1 + y2 * y2);

      // Backward
      var dLoss = 1.0;
      var dY1 = y1, dY2 = y2;
      var dW11 = dY1 * ax1, dW12 = dY1 * ax2;
      var dW21 = dY2 * ax1, dW22 = dY2 * ax2;
      var dAX1 = W11 * dY1 + W21 * dY2;
      var dAX2 = W12 * dY1 + W22 * dY2;
      var dAB1 = dY1, dAB2 = dY2;

      var col1 = W * 0.16;
      var col2 = W * 0.44;
      var col3 = W * 0.68;
      var col4 = W * 0.88;

      nodes = [
        { label: 'x [x₁,x₂]', val: Math.hypot(ax1, ax2), grad: Math.hypot(dAX1, dAX2), x: col1, y: H * 0.25, type: 'in', desc: 'Input Vector x' },
        { label: 'W (2×2)', val: Math.hypot(W11, W22), grad: Math.hypot(dW11, dW22), x: col1, y: H * 0.50, type: 'param', desc: 'Weight Matrix W' },
        { label: 'b [b₁,b₂]', val: Math.hypot(ab1, ab2), grad: Math.hypot(dAB1, dAB2), x: col1, y: H * 0.75, type: 'param', desc: 'Bias Vector b' },
        { label: 'W·x (Map)', val: Math.hypot(y1 - ab1, y2 - ab2), grad: Math.hypot(dY1, dY2), x: col2, y: H * 0.38, type: 'op', desc: 'Linear Map W·x' },
        { label: 'y = Wx+b', val: Math.hypot(y1, y2), grad: Math.hypot(dY1, dY2), x: col3, y: H * 0.50, type: 'op', desc: 'Affine Vector y' },
        { label: 'Loss ½||y||²', val: loss, grad: dLoss, x: col4, y: H * 0.50, type: 'out', desc: 'Scalar L2 Loss' }
      ];

      edges = [
        { from: 0, to: 3, local: 'J = W', dOut: Math.hypot(dAX1, dAX2) },
        { from: 1, to: 3, local: 'x ⊗', dOut: Math.hypot(dW11, dW22) },
        { from: 3, to: 4, local: '+I', dOut: Math.hypot(dY1, dY2) },
        { from: 2, to: 4, local: '+I', dOut: Math.hypot(dAB1, dAB2) },
        { from: 4, to: 5, local: 'yᵀ', dOut: dLoss }
      ];

      reverseTape = [5, 4, 3, 2, 1, 0];

      // Update Jacobian card telemetry
      if (valAgJacobianMatrix) {
        valAgJacobianMatrix.innerHTML = '[ ' + W11.toFixed(2) + ', ' + W12.toFixed(2) + ' ]<br>[ ' + W21.toFixed(2) + ', ' + W22.toFixed(2) + ' ]';
      }
      if (valAgOutputVec) {
        valAgOutputVec.textContent = '[ ' + y1.toFixed(2) + ', ' + y2.toFixed(2) + ' ]';
      }
    }

    // Save bounding boxes for mouse hover hit testing
    currentAutogradNodes = nodes.map(function (n) {
      return { x: n.x, y: n.y, bw: bw, bh: bh, label: n.label, type: n.type, val: n.val, grad: n.grad, desc: n.desc };
    });

    var activeTapeNodeIdx = null;
    if (state.autogradTapeStep >= 0 && state.autogradTapeStep < reverseTape.length) {
      activeTapeNodeIdx = reverseTape[state.autogradTapeStep];
    }

    // ── 1. Draw Explicit Directed Edges ─────────────────────────────────────
    ctx.save();
    edges.forEach(function (e) {
      var nFrom = nodes[e.from];
      var nTo = nodes[e.to];
      if (!nFrom || !nTo) return;

      var p1x = nFrom.x + bw / 2;
      var p1y = nFrom.y;
      var p2x = nTo.x - bw / 2;
      var p2y = nTo.y;
      var mx = (p1x + p2x) / 2;
      var my = (p1y + p2y) / 2;

      var isHighlighted = (state.hoverAutogradNode === e.from || state.hoverAutogradNode === e.to);
      var isTapeActive = (activeTapeNodeIdx === e.from || activeTapeNodeIdx === e.to);

      ctx.beginPath();
      ctx.moveTo(p1x, p1y);
      var cx = (p1x + p2x) / 2;
      ctx.bezierCurveTo(cx, p1y, cx, p2y, p2x, p2y);

      if (isTapeActive) {
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.8;
      } else if (isHighlighted) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.5;
      } else if (isBackward) {
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
        ctx.lineWidth = 1.8;
      } else {
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
        ctx.lineWidth = 1.4;
      }
      ctx.stroke();

      // Draw Arrowhead pointing to (p2x, p2y)
      var arrowSize = 6;
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.moveTo(p2x, p2y);
      ctx.lineTo(p2x - arrowSize * 1.4, p2y - arrowSize);
      ctx.lineTo(p2x - arrowSize * 1.4, p2y + arrowSize);
      ctx.closePath();
      ctx.fill();

      // Draw Local Derivative Pill at midpoint
      if (isBackward || isHighlighted || isTapeActive) {
        ctx.save();
        ctx.font = '600 9px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
        var txt = e.local;
        var tw = ctx.measureText(txt).width;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
        ctx.strokeStyle = isTapeActive ? '#f59e0b' : 'rgba(245, 158, 11, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(mx - tw / 2 - 4, my - 8, tw + 8, 16, 4);
        else ctx.rect(mx - tw / 2 - 4, my - 8, tw + 8, 16);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = isTapeActive ? '#fde68a' : '#fbbf24';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(txt, mx, my);
        ctx.restore();
      }
    });
    ctx.restore();

    // ── 2. Draw Nodes ───────────────────────────────────────────────────────
    nodes.forEach(function (n, idx) {
      ctx.save();
      var isTapeTarget = (activeTapeNodeIdx === idx);
      var isHovered = (state.hoverAutogradNode === idx);

      // Node card backdrop
      ctx.fillStyle = n.type === 'out' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(15, 23, 42, 0.88)';

      if (isTapeTarget) {
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#f59e0b';
        ctx.shadowBlur = 12;
      } else if (isHovered) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = '#38bdf8';
        ctx.shadowBlur = 8;
      } else if (isBackward && n.type === 'param') {
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
      } else if (n.type === 'out') {
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
      } else if (n.type === 'op') {
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
        ctx.lineWidth = 1.5;
      } else {
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
        ctx.lineWidth = 1.5;
      }

      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(n.x - bw / 2, n.y - bh / 2, bw, bh, 8);
      else ctx.rect(n.x - bw / 2, n.y - bh / 2, bw, bh);
      ctx.fill();
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Header Label
      ctx.font = '700 12px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.textAlign = 'center';
      if (n.type === 'param') ctx.fillStyle = '#e879f9';
      else if (n.type === 'out') ctx.fillStyle = '#34d399';
      else if (n.type === 'op') ctx.fillStyle = '#38bdf8';
      else ctx.fillStyle = '#f8fafc';
      ctx.fillText(n.label, n.x, n.y - 12);

      // Forward Value Pill
      ctx.font = '600 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillStyle = isForward ? '#38bdf8' : '#94a3b8';
      ctx.fillText('v: ' + (n.val >= 0 ? '+' : '') + n.val.toFixed(2), n.x, n.y + 3);

      // Backward Gradient Pill
      if (isBackward || isTapeTarget) {
        var gradCol = n.grad > 0.001 ? '#10b981' : (n.grad < -0.001 ? '#f43f5e' : '#94a3b8');
        ctx.fillStyle = gradCol;
        ctx.font = '700 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
        ctx.fillText('∇: ' + (n.grad >= 0 ? '+' : '') + n.grad.toFixed(2), n.x, n.y + 17);
      } else {
        ctx.fillStyle = '#64748b';
        ctx.font = '500 9px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
        ctx.fillText('∇: pending', n.x, n.y + 17);
      }

      ctx.restore();
    });
  }

  // ── Notes & Quiz Interactive Visual Blackboard ───────────────────────────

  function renderNotesCanvas() {
    var topic = state.currentNotesTopic || 'matrix-transform';
    var o = worldToScreen(0, 0);

    drawBackgroundGrid();
    drawAxes();

    if (topic === 'matrix-transform') {
      var m = state.matrix;
      drawTransformedGrid(m);
      drawTransformedShape(m);
      var o = worldToScreen(0, 0);
      var pI = worldToScreen(m.a, m.c);
      var pJ = worldToScreen(m.b, m.d);
      var angI = Math.atan2(m.c, m.a);
      var angJ = Math.atan2(m.d, m.b);

      drawArrow(o.x, o.y, pI.x, pI.y, '#f43f5e', 2.8);
      drawVectorHandle(pI.x, pI.y, '#f43f5e', 'i', false);
      drawVectorLabel('Col 1: î → [' + m.a.toFixed(1) + ', ' + m.c.toFixed(1) + ']', pI.x, pI.y, '#f43f5e', angI, 5);

      drawArrow(o.x, o.y, pJ.x, pJ.y, '#06b6d4', 2.8);
      drawVectorHandle(pJ.x, pJ.y, '#06b6d4', 'j', false);
      drawVectorLabel('Col 2: ĵ → [' + m.b.toFixed(1) + ', ' + m.d.toFixed(1) + ']', pJ.x, pJ.y, '#06b6d4', angJ, 5);

    } else if (topic === 'determinant') {
      var mDet = state.matrix;
      // Draw unit square reference (original space)
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      var u0 = worldToScreen(0, 0), u1 = worldToScreen(1, 0), u2 = worldToScreen(1, 1), u3 = worldToScreen(0, 1);
      ctx.beginPath();
      ctx.moveTo(u0.x, u0.y); ctx.lineTo(u1.x, u1.y); ctx.lineTo(u2.x, u2.y); ctx.lineTo(u3.x, u3.y); ctx.closePath();
      ctx.stroke();
      ctx.restore();

      drawTransformedShape(mDet);
      drawBasisVectors(mDet);
      var det = mDet.determinant();
      var center = worldToScreen((mDet.a + mDet.b) / 2, (mDet.c + mDet.d) / 2);
      drawVectorLabel('Area Factor |det(A)| = ' + Math.abs(det).toFixed(2), center.x, center.y, '#38bdf8');
      if (det < -0.01) {
        drawVectorLabel('⚠️ Orientation Inverted (det < 0)', center.x, center.y + 22, '#f43f5e');
      } else if (Math.abs(det) <= 0.01) {
        drawVectorLabel('⚠️ Dimension Collapsed (det = 0, Area = 0)', center.x, center.y + 22, '#ef4444');
      } else {
        drawVectorLabel('✅ Orientation Preserved (det > 0)', center.x, center.y + 22, '#10b981');
      }

    } else if (topic === 'eigenvalues' || topic === 'eigen') {
      drawEigenSpanLines(state.matrix);
      drawBasisVectors(state.matrix);
      var eigens = Engine.solveEigensystem(state.matrix);
      if (eigens.isReal && eigens.eigenvectors.length > 0) {
        var v1 = eigens.eigenvectors[0].vector;
        var tScale = 1.3 + 0.5 * Math.sin(performance.now() / 320);
        var scaled = new Vector2D(v1.x * tScale, v1.y * tScale);
        var p = worldToScreen(scaled.x, scaled.y);
        drawArrow(o.x, o.y, p.x, p.y, '#eab308', 3);
        drawVectorLabel('Invariant Vector: A·v = λ·v (Direction Never Rotates)', p.x, p.y, '#eab308');
      } else {
        drawVectorLabel('Complex Eigensystem: Space Rotates (No Real Invariant Lines)', o.x - 140, o.y - 80, '#c084fc');
      }

    } else if (topic === 'diagonalization') {
      drawEigenSpanLines(state.matrix);
      var eigensDiag = Engine.solveEigensystem(state.matrix);
      if (eigensDiag.isReal && eigensDiag.eigenvectors.length >= 2) {
        var ev1 = eigensDiag.eigenvectors[0].vector;
        var ev2 = eigensDiag.eigenvectors[1].vector;
        var p1 = worldToScreen(ev1.x, ev1.y);
        var p2 = worldToScreen(ev2.x, ev2.y);
        drawArrow(o.x, o.y, p1.x, p1.y, '#eab308', 2.5);
        drawArrow(o.x, o.y, p2.x, p2.y, '#a855f7', 2.5);
        drawVectorLabel('Eigenbasis Axis 1 (λ₁ = ' + eigensDiag.eigenvalues[0].value.toFixed(2) + ')', p1.x, p1.y, '#eab308');
        drawVectorLabel('Eigenbasis Axis 2 (λ₂ = ' + eigensDiag.eigenvalues[1].value.toFixed(2) + ')', p2.x, p2.y, '#a855f7');

        // Draw A^k power trajectory projection
        var probeK = ev1.scale(1.2).add(ev2.scale(0.8));
        var pk0 = worldToScreen(probeK.x, probeK.y);
        drawVectorLabel('Aᵏ = P·Dᵏ·P⁻¹ (Power trajectory decoupled along eigenaxes)', pk0.x, pk0.y - 30, '#38bdf8');
      } else {
        drawVectorLabel('A = P·D·P⁻¹ requires 2 linearly independent eigenvectors', o.x - 120, o.y - 60, '#f59e0b');
      }

    } else if (topic === 'svd') {
      // Draw unit circle (original input space)
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(o.x, o.y, state.scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // Transformed Hyper-Ellipse
      var svd = Engine.computeSVD2x2(state.matrix);
      var steps = 64;
      ctx.save();
      ctx.beginPath();
      for (var s = 0; s <= steps; s++) {
        var theta = (s / steps) * Math.PI * 2;
        var tw = state.matrix.apply(new Vector2D(Math.cos(theta), Math.sin(theta)));
        var sp = worldToScreen(tw.x, tw.y);
        if (s === 0) ctx.moveTo(sp.x, sp.y); else ctx.lineTo(sp.x, sp.y);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(6, 182, 212, 0.12)';
      ctx.fill();
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();

      // Semi-axes vectors (Singular Values σ₁ and σ₂)
      var ax1 = worldToScreen(svd.u1.x * svd.sigma1, svd.u1.y * svd.sigma1);
      var ax2 = worldToScreen(svd.u2.x * svd.sigma2, svd.u2.y * svd.sigma2);
      drawArrow(o.x, o.y, ax1.x, ax1.y, '#38bdf8', 2.5);
      drawArrow(o.x, o.y, ax2.x, ax2.y, '#10b981', 2.5);
      drawVectorLabel('σ₁·u₁ = ' + svd.sigma1.toFixed(2), ax1.x, ax1.y, '#38bdf8');
      drawVectorLabel('σ₂·u₂ = ' + svd.sigma2.toFixed(2), ax2.x, ax2.y, '#10b981');
      drawVectorLabel('SVD: A = U·Σ·Vᵀ maps Unit Circle to Hyper-Ellipse', o.x - 120, o.y + 120, '#eab308');

    } else if (topic === 'rank-nullity') {
      var singM = new Matrix2x2(1.2, 0.6, 1.2, 0.6); // rank 1
      drawRankNullityLines(singM);
      var pSample = worldToScreen(1.5, 2.0);
      var tSample = singM.apply(new Vector2D(1.5, 2.0));
      var pResult = worldToScreen(tSample.x, tSample.y);
      drawArrow(o.x, o.y, pSample.x, pSample.y, 'rgba(255, 255, 255, 0.4)', 1.5);
      drawArrow(o.x, o.y, pResult.x, pResult.y, '#10b981', 2.5);
      drawVectorLabel('Any Input Vector v', pSample.x, pSample.y, '#94a3b8');
      drawVectorLabel('Mapped to 1D Column Space im(A)', pResult.x, pResult.y, '#10b981');
      drawVectorLabel('dim(ker A) [1] + dim(im A) [1] = n [2]', o.x - 100, o.y + 110, '#f59e0b');

    } else if (topic === 'optimizers') {
      // Draw 2D Loss Landscape Contours (Chalkboard view)
      ctx.save();
      for (var r = 0.5; r <= 3.5; r += 0.6) {
        ctx.strokeStyle = 'rgba(99, 102, 241, ' + (0.15 + r * 0.05) + ')';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        // Elliptical contour L = w1^2 + 2*w2^2
        var ellSteps = 48;
        for (var si = 0; si <= ellSteps; si++) {
          var ang = (si / ellSteps) * Math.PI * 2;
          var cw = worldToScreen(r * Math.cos(ang), (r / 1.414) * Math.sin(ang));
          if (si === 0) ctx.moveTo(cw.x, cw.y); else ctx.lineTo(cw.x, cw.y);
        }
        ctx.closePath();
        ctx.stroke();
      }
      ctx.restore();

      // Current optimization particle position
      var px = 2.0, py = 1.4;
      var pPos = worldToScreen(px, py);
      ctx.beginPath();
      ctx.arc(pPos.x, pPos.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#f8fafc';
      ctx.fill();

      // Negative Gradient -∇L = [-2w1, -4w2]
      var gradStep = worldToScreen(px - 0.8 * 2.0 * 0.35, py - 0.8 * 2.8 * 0.35);
      drawArrow(pPos.x, pPos.y, gradStep.x, gradStep.y, '#ef4444', 2.5);
      drawVectorLabel('Steepest Descent -∇L', gradStep.x, gradStep.y, '#ef4444');

      // Momentum Inertia Arrow
      var momStep = worldToScreen(px - 1.1, py - 0.4);
      drawArrow(pPos.x, pPos.y, momStep.x, momStep.y, '#f59e0b', 2);
      drawVectorLabel('Momentum Step (with velocity v)', momStep.x, momStep.y, '#f59e0b');

      drawVectorLabel('Minimizing Loss L(w): w_{t+1} = w_t - α·∇L', o.x - 90, o.y + 110, '#10b981');

    } else if (topic === 'backprop') {
      // Draw Mini DAG on blackboard
      var n1 = worldToScreen(-2.2, 0.8);
      var n2 = worldToScreen(-2.2, -0.8);
      var n3 = worldToScreen(0.0, 0.0);
      var n4 = worldToScreen(2.2, 0.0);

      // Connectors
      drawArrow(n1.x, n1.y, n3.x, n3.y, 'rgba(56, 189, 248, 0.5)', 2);
      drawArrow(n2.x, n2.y, n3.x, n3.y, 'rgba(56, 189, 248, 0.5)', 2);
      drawArrow(n3.x, n3.y, n4.x, n4.y, 'rgba(56, 189, 248, 0.5)', 2);

      // Reverse gradient dashed arrows
      ctx.save();
      ctx.setLineDash([4, 4]);
      drawArrow(n4.x, n4.y - 18, n3.x, n3.y - 18, '#f59e0b', 2);
      drawArrow(n3.x, n3.y - 18, n1.x, n1.y - 18, '#f59e0b', 2);
      ctx.restore();

      drawVectorLabel('Forward Pass: Activations (Cyan →)', o.x - 80, o.y - 95, '#38bdf8');
      drawVectorLabel('Backward Pass: Chain Rule Gradients (Amber ←)', o.x - 110, o.y - 70, '#f59e0b');
      drawVectorLabel('∂L/∂w = (∂L/∂y) · (∂y/∂w)', o.x - 60, o.y + 90, '#10b981');
    }
  }

  function renderQuizCanvas() {
    var o = worldToScreen(0, 0);
    drawAxes();

    // Pedagogical chalkboard visualization tailored for all 15 Viva questions
    if (currentQuizIdx === 0) {
      // Q1: Det < 0 (Orientation Inversion / Flipped Chirality)
      var m = new Matrix2x2(-1.2, 0.4, 0.4, 1.2);
      drawTransformedShape(m);
      drawBasisVectors(m);
      drawVectorLabel('det(A) = -1.60 < 0 (Chirality Mirrored)', o.x - 70, o.y - 100, '#f43f5e');
    } else if (currentQuizIdx === 1) {
      // Q2: λ = 0 (Singular Matrix, det = 0, dimension collapse)
      var m = new Matrix2x2(1.5, 0.75, 1.0, 0.5);
      drawRankNullityLines(m);
      drawBasisVectors(m);
      drawVectorLabel('λ = 0 ⟹ det(A) = 0 (Squashed to 1D Line)', o.x - 70, o.y - 90, '#ef4444');
    } else if (currentQuizIdx === 2) {
      // Q3: Non-commutativity AB ≠ BA
      var p1 = worldToScreen(state.matrix.a, state.matrix.c);
      var p2 = worldToScreen(state.matrixB.a, state.matrixB.c);
      drawArrow(o.x, o.y, p1.x, p1.y, '#38bdf8', 2.5);
      drawArrow(o.x, o.y, p2.x, p2.y, '#f59e0b', 2.5);
      drawVectorLabel('AB · î', p1.x, p1.y, '#38bdf8');
      drawVectorLabel('BA · î', p2.x, p2.y, '#f59e0b');
      drawVectorLabel('Non-Commutative: AB ≠ BA (Order Matters)', o.x - 90, o.y + 110, '#eab308');
    } else if (currentQuizIdx === 3) {
      // Q4: SVD Unit Circle mapped to Ellipse
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      var rPix = state.scale;
      ctx.beginPath();
      ctx.arc(o.x, o.y, rPix, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      var m = new Matrix2x2(1.8, 0.5, 0.3, 1.1);
      drawTransformedShape(m);
      var sv1 = worldToScreen(1.87, 0.52);
      drawArrow(o.x, o.y, sv1.x, sv1.y, '#10b981', 2.5);
      drawVectorLabel('σ₁ u₁ (Major Axis)', sv1.x, sv1.y, '#10b981');
      drawVectorLabel('SVD: Circle ↦ Ellipse (Semi-axes σ₁, σ₂)', o.x - 80, o.y - 110, '#38bdf8');
    } else if (currentQuizIdx === 4) {
      // Q5: Diagonalization A = P D P⁻¹
      var m = new Matrix2x2(1.6, 0.8, 0.8, 1.2);
      drawTransformedShape(m);
      drawEigenSpanLines(m);
      drawBasisVectors(m);
      drawVectorLabel('P = [ v₁ | v₂ ] (Columns are Eigenvectors)', o.x - 90, o.y - 100, '#a855f7');
    } else if (currentQuizIdx === 5) {
      // Q6: Rank-Nullity Theorem (Rank + Nullity = n)
      var m = new Matrix2x2(1.4, 0.7, 0.8, 0.4);
      drawRankNullityLines(m);
      drawBasisVectors(m);
      drawVectorLabel('Rank(1) + Nullity(1) = Dimension(2)', o.x - 80, o.y - 90, '#f59e0b');
    } else if (currentQuizIdx === 6) {
      // Q7: Orthogonal vectors u · v = 0 (θ = 90°)
      var uPos = worldToScreen(2.5, 0.8);
      var vPos = worldToScreen(-0.8, 2.5);
      drawArrow(o.x, o.y, uPos.x, uPos.y, '#f43f5e', 2.5);
      drawArrow(o.x, o.y, vPos.x, vPos.y, '#06b6d4', 2.5);
      drawVectorLabel('u [2.5, 0.8]', uPos.x, uPos.y, '#f43f5e');
      drawVectorLabel('v [-0.8, 2.5]', vPos.x, vPos.y, '#06b6d4');
      drawVectorLabel('u · v = 0 ⟹ Perpendicular (θ = 90°)', o.x - 70, o.y - 95, '#10b981');
    } else if (currentQuizIdx === 7) {
      // Q8: Optimization Ravines: Adam vs SGD
      ctx.save();
      for (var r = 1; r <= 3; r++) {
        ctx.strokeStyle = 'rgba(255, 255, 255, ' + (0.08 * (4 - r)) + ')';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.ellipse(o.x, o.y, r * 55, r * 110, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      var sgdPts = [[-2.2, 1.8], [-1.4, -1.5], [-0.8, 1.2], [-0.3, -0.9], [0, 0]];
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      sgdPts.forEach(function (p, i) {
        var sp = worldToScreen(p[0], p[1]);
        if (i === 0) ctx.moveTo(sp.x, sp.y); else ctx.lineTo(sp.x, sp.y);
      });
      ctx.stroke();

      var adamPts = [[-2.2, 1.8], [-1.2, 0.9], [-0.5, 0.3], [0, 0]];
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      adamPts.forEach(function (p, i) {
        var sp = worldToScreen(p[0], p[1]);
        if (i === 0) ctx.moveTo(sp.x, sp.y); else ctx.lineTo(sp.x, sp.y);
      });
      ctx.stroke();
      ctx.restore();

      drawVectorLabel('SGD (Red Oscillation) vs Adam (Emerald Direct Momentum)', o.x - 120, o.y - 120, '#f59e0b');
    } else if (currentQuizIdx === 8) {
      // Q9: Power Iteration Aᵏx₀ → Dominant Eigenvector v₁
      var x0 = worldToScreen(1.0, 2.2);
      var x1 = worldToScreen(1.8, 1.8);
      var x2 = worldToScreen(2.4, 1.2);
      var vDom = worldToScreen(2.8, 0.7);
      drawArrow(o.x, o.y, x0.x, x0.y, '#94a3b8', 1.5);
      drawArrow(o.x, o.y, x1.x, x1.y, '#38bdf8', 2);
      drawArrow(o.x, o.y, x2.x, x2.y, '#818cf8', 2.2);
      drawArrow(o.x, o.y, vDom.x, vDom.y, '#f59e0b', 3);
      drawVectorLabel('x₀', x0.x, x0.y, '#94a3b8');
      drawVectorLabel('Ax₀', x1.x, x1.y, '#38bdf8');
      drawVectorLabel('Dominant v₁', vDom.x, vDom.y, '#f59e0b');
      drawVectorLabel('Power Iteration: Aᵏx₀ converges to Dominant Eigenvector', o.x - 110, o.y - 100, '#fde68a');
    } else if (currentQuizIdx === 9) {
      // Q10: Autograd Reverse-Mode Chain Rule
      var n1 = worldToScreen(-2.0, 0.5);
      var n2 = worldToScreen(0, 0.5);
      var n3 = worldToScreen(2.0, 0.5);
      drawArrow(n1.x, n1.y, n2.x, n2.y, '#38bdf8', 2.5);
      drawArrow(n2.x, n2.y, n3.x, n3.y, '#38bdf8', 2.5);
      drawArrow(n3.x, n3.y - 30, n2.x, n2.y - 30, '#f59e0b', 2.5);
      drawArrow(n2.x, n2.y - 30, n1.x, n1.y - 30, '#f59e0b', 2.5);
      drawVectorLabel('Forward: Activations (Cyan →)', o.x - 70, o.y + 40, '#38bdf8');
      drawVectorLabel('Backward: ∂L/∂w = (∂L/∂y)·(∂y/∂w) (Amber ←)', o.x - 100, o.y + 70, '#f59e0b');
    } else if (currentQuizIdx === 10) {
      // Q11: Invertible System Ax = b
      var m = new Matrix2x2(1.5, 0.4, 0.3, 1.2);
      var xSol = { x: 1.2, y: 1.0 };
      var bTarget = { x: m.a * xSol.x + m.b * xSol.y, y: m.c * xSol.x + m.d * xSol.y };
      var spX = worldToScreen(xSol.x, xSol.y);
      var spB = worldToScreen(bTarget.x, bTarget.y);
      drawArrow(o.x, o.y, spX.x, spX.y, '#38bdf8', 2.5);
      drawArrow(o.x, o.y, spB.x, spB.y, '#f59e0b', 2.5);
      drawVectorLabel('Solution x = A⁻¹b', spX.x, spX.y, '#38bdf8');
      drawVectorLabel('Target b', spB.x, spB.y, '#f59e0b');
      drawVectorLabel('det(A) ≠ 0 ⟹ Unique Solution x exists for every b', o.x - 100, o.y - 100, '#10b981');
    } else if (currentQuizIdx === 11) {
      // Q12: Cross Product Parallelogram Area
      var u = { x: 2.2, y: 0.5 };
      var v = { x: 0.8, y: 2.0 };
      var uScreen = worldToScreen(u.x, u.y);
      var vScreen = worldToScreen(v.x, v.y);
      var uvScreen = worldToScreen(u.x + v.x, u.y + v.y);

      ctx.save();
      ctx.fillStyle = 'rgba(245, 158, 11, 0.22)';
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(o.x, o.y);
      ctx.lineTo(uScreen.x, uScreen.y);
      ctx.lineTo(uvScreen.x, uvScreen.y);
      ctx.lineTo(vScreen.x, vScreen.y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      drawArrow(o.x, o.y, uScreen.x, uScreen.y, '#f43f5e', 2.5);
      drawArrow(o.x, o.y, vScreen.x, vScreen.y, '#06b6d4', 2.5);
      drawVectorLabel('u', uScreen.x, uScreen.y, '#f43f5e');
      drawVectorLabel('v', vScreen.x, vScreen.y, '#06b6d4');
      drawVectorLabel('Parallelogram Area = |u × v| = 4.00', o.x - 70, o.y - 100, '#f59e0b');
    } else if (currentQuizIdx === 12) {
      // Q13: 3D Pure Rotation Matrix det(R) = +1
      var ax = worldToScreen(1.8, -0.6);
      var ay = worldToScreen(0.5, 1.9);
      var az = worldToScreen(-1.2, 1.2);
      drawArrow(o.x, o.y, ax.x, ax.y, '#f43f5e', 2.5);
      drawArrow(o.x, o.y, ay.x, ay.y, '#10b981', 2.5);
      drawArrow(o.x, o.y, az.x, az.y, '#38bdf8', 2.5);
      drawVectorLabel('R·î', ax.x, ax.y, '#f43f5e');
      drawVectorLabel('R·ĵ', ay.x, ay.y, '#10b981');
      drawVectorLabel('R·k̂', az.x, az.y, '#38bdf8');
      drawVectorLabel('Pure 3D Rotation R ∈ SO(3): det(R) = +1 (Rigid Body)', o.x - 110, o.y - 100, '#6ee7b7');
    } else if (currentQuizIdx === 13) {
      // Q14: Hessian Saddle Point: λ₁ > 0, λ₂ < 0
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 1.2;
      for (var k = 1; k <= 3; k++) {
        var d = k * 35;
        ctx.beginPath();
        ctx.moveTo(o.x - d, o.y - d * 0.8);
        ctx.quadraticCurveTo(o.x, o.y - d * 0.2, o.x + d, o.y - d * 0.8);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(o.x - d * 0.8, o.y - d);
        ctx.quadraticCurveTo(o.x - d * 0.2, o.y, o.x - d * 0.8, o.y + d);
        ctx.stroke();
      }
      ctx.restore();

      var up = worldToScreen(0, 1.8);
      var down = worldToScreen(1.8, 0);
      drawArrow(o.x, o.y, up.x, up.y, '#10b981', 2.5);
      drawArrow(o.x, o.y, down.x, down.y, '#f43f5e', 2.5);
      drawVectorLabel('λ₁ > 0 (Min Curve)', up.x, up.y, '#10b981');
      drawVectorLabel('λ₂ < 0 (Max Curve)', down.x, down.y, '#f43f5e');
      drawVectorLabel('Hessian ∇²L: Mixed Signs ⟹ Saddle Point', o.x - 90, o.y - 100, '#fde68a');
    } else if (currentQuizIdx === 14) {
      // Q15: Trace tr(A) = sum of eigenvalues
      var m = new Matrix2x2(1.8, 0.6, 0.4, 1.2);
      drawTransformedShape(m);
      drawEigenSpanLines(m);
      drawBasisVectors(m);
      drawVectorLabel('tr(A) = a₁₁ + a₂₂ = 1.8 + 1.2 = 3.00', o.x - 80, o.y - 110, '#38bdf8');
      drawVectorLabel('λ₁ + λ₂ = 2.10 + 0.90 = 3.00 (Invariant Sum)', o.x - 90, o.y + 110, '#10b981');
    } else {
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

  // ── Dynamic Collision-Free Label Engine ──────────────────────────────────
  var activeLabelQueue = [];
  var isQueueingLabels = false;

  function beginLabelPass() {
    activeLabelQueue = [];
    isQueueingLabels = true;
  }

  function drawVectorLabel(text, x, y, color, angle, priority) {
    if (!text) return;
    if (isQueueingLabels) {
      activeLabelQueue.push({
        text: text,
        anchorX: x,
        anchorY: y,
        color: color || '#f8fafc',
        angle: typeof angle === 'number' ? angle : 0,
        priority: typeof priority === 'number' ? priority : 0
      });
    } else {
      renderSingleLabelPill(text, x, y, color || '#f8fafc', typeof angle === 'number' ? angle : 0, x, y, false);
    }
  }

  function renderSingleLabelPill(text, boxX, boxY, color, anchorX, anchorY, leader) {
    ctx.save();
    ctx.font = '600 11px JetBrains Mono, monospace';
    var metrics = ctx.measureText(text);
    var textWidth = metrics.width;
    var textHeight = 12;
    var padX = 6;
    var padY = 3.5;
    var totalW = textWidth + padX * 2 + 10;
    var totalH = textHeight + padY * 2;

    // Leader line if displaced from anchor
    if (leader && (Math.abs(boxX + totalW / 2 - anchorX) > 18 || Math.abs(boxY + totalH / 2 - anchorY) > 18)) {
      ctx.beginPath();
      ctx.moveTo(anchorX, anchorY);
      ctx.lineTo(boxX + totalW / 2, boxY + totalH / 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Outer shadow
    ctx.shadowColor = 'rgba(0, 0, 0, 0.65)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetY = 2;

    // Dark glass pill
    ctx.fillStyle = 'rgba(9, 14, 26, 0.92)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.16)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(boxX, boxY, totalW, totalH, 5);
    } else {
      ctx.rect(boxX, boxY, totalW, totalH);
    }
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.stroke();

    // Colored accent dot
    var dotRadius = 2.5;
    var dotX = boxX + padX + dotRadius;
    var dotY = boxY + totalH / 2;
    ctx.beginPath();
    ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // Text with slight indent
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, boxX + padX + dotRadius * 2 + 5, boxY + totalH / 2 + 0.5);
    ctx.restore();
  }

  function flushVectorLabels() {
    isQueueingLabels = false;
    if (activeLabelQueue.length === 0) return;

    ctx.save();
    ctx.font = '600 11px JetBrains Mono, monospace';

    // 1. Calculate initial bounding boxes
    var computed = activeLabelQueue.map(function (item) {
      var metrics = ctx.measureText(item.text);
      var textWidth = metrics.width;
      var totalW = textWidth + 12 + 5 + 6;
      var totalH = 19;

      var rad = item.angle;
      var offX = Math.cos(rad) * 18;
      var offY = -Math.sin(rad) * 18;

      var initX = offX >= 0 ? item.anchorX + 8 : item.anchorX - totalW - 8;
      var initY = offY <= 0 ? item.anchorY - totalH - 4 : item.anchorY + 4;

      return {
        text: item.text,
        color: item.color,
        anchorX: item.anchorX,
        anchorY: item.anchorY,
        x: initX,
        y: initY,
        w: totalW,
        h: totalH,
        priority: item.priority
      };
    });
    ctx.restore();

    // 2. Iterative collision relaxation (up to 4 passes)
    for (var pass = 0; pass < 4; pass++) {
      var moved = false;
      for (var i = 0; i < computed.length; i++) {
        for (var j = i + 1; j < computed.length; j++) {
          var a = computed[i];
          var b = computed[j];

          var margin = 4;
          var overlapX = (a.x < b.x + b.w + margin) && (a.x + a.w + margin > b.x);
          var overlapY = (a.y < b.y + b.h + margin) && (a.y + a.h + margin > b.y);

          if (overlapX && overlapY) {
            var shift = (a.h + margin) / 2 + 1;
            if (a.y <= b.y) {
              a.y -= shift;
              b.y += shift;
            } else {
              a.y += shift;
              b.y -= shift;
            }
            if (Math.abs(a.x - b.x) < 20) {
              if (a.anchorX <= b.anchorX) {
                a.x -= 8;
                b.x += 8;
              } else {
                a.x += 8;
                b.x -= 8;
              }
            }
            moved = true;
          }
        }
      }
      if (!moved) break;
    }

    // 3. Render all relaxed labels
    computed.forEach(function (c) {
      renderSingleLabelPill(c.text, c.x, c.y, c.color, c.anchorX, c.anchorY, true);
    });
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

    if (state.mode === 'autograd') {
      var foundIdx = null;
      for (var ni = 0; ni < currentAutogradNodes.length; ni++) {
        var nd = currentAutogradNodes[ni];
        if (Math.abs(sx - nd.x) <= nd.bw / 2 && Math.abs(sy - nd.y) <= nd.bh / 2) {
          foundIdx = ni;
          break;
        }
      }
      state.hoverAutogradNode = foundIdx;
      if (foundIdx !== null) return 'ag_node_' + foundIdx;
      return null;
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

    updateAdaptiveHUD(pos, world);

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
      formula: 'T(v) = A · v = [a·x + b·y,  c·x + d·y]ᵀ',
      tabTarget: 'transform',
      presetMatrix: new Matrix2x2(1.5, 0.5, 0.5, 1.2),
      vivaQ: 'If a 2×2 matrix maps î into [2, 0]ᵀ and ĵ into [0, 3]ᵀ, what is the matrix?',
      vivaA: 'The matrix is [2, 0; 0, 3] (a non-uniform scaling matrix). The columns of ANY matrix represent the destination coordinates of the standard basis vectors î and ĵ.',
      html: `
        <h4>Core Intuition</h4>
        <p>A matrix is not just a table of numbers—it is a <strong>spatial coordinate transformer</strong> that maps any vector in ℝ² to a new location while preserving two fundamental linear rules:</p>
        <ul>
          <li>The origin (0, 0) remains permanently anchored at the center.</li>
          <li>All grid lines remain straight, parallel, and evenly spaced.</li>
        </ul>
        <div class="theory-highlight">The columns of matrix A tell you where the standard basis vectors land:<br>• Column 1 = î landing position [a, c]ᵀ<br>• Column 2 = ĵ landing position [b, d]ᵀ</div>
        <h4>Real-World AI & Graphics Application</h4>
        <p>In neural networks, fully connected linear layers compute <em>y = W · x + b</em>. The weight matrix <em>W</em> rotates, shears, and stretches feature embeddings to make raw data linearly separable.</p>
      `
    },
    'determinant': {
      title: '2. The Determinant & Volume Scaling',
      formula: 'det(A) = ad - bc = Area(A · Shape) / Area(Shape)',
      tabTarget: 'transform',
      presetMatrix: new Matrix2x2(1.2, 0.4, 0.4, 1.0),
      vivaQ: 'Why does det(A) = 0 imply non-invertibility?',
      vivaA: 'Because the transformation squashes 2D space into a 1D line or point. Multiple distinct input points are mapped to the same output, meaning information is irreversibly destroyed and no inverse function can reconstruct the original preimage.',
      html: `
        <h4>Geometric Meaning</h4>
        <p>The determinant <strong>det(A) = ad - bc</strong> measures the exact scaling factor by which areas (in 2D) or volumes (in 3D) expand or contract under the transformation.</p>
        <ul>
          <li><strong>det(A) > 0:</strong> Space scales by |det| and spatial orientation (chirality) is preserved.</li>
          <li><strong>det(A) < 0:</strong> Space is flipped or mirrored (like viewing the plane through a mirror).</li>
          <li><strong>det(A) = 0:</strong> Space is squashed into a lower dimension (2D plane collapses to a 1D line or point). Information is lost forever—the matrix is <strong>singular</strong>!</li>
        </ul>
        <div class="theory-highlight">Transformed Area = |det(A)| · Original Unit Square Area</div>
      `
    },
    'eigenvalues': {
      title: '3. Eigenvalues, Eigenvectors & Resonance',
      formula: 'A · v = λ · v   ⟺   det(A - λI) = 0',
      tabTarget: 'eigen',
      presetMatrix: new Matrix2x2(2.0, 1.0, 0.0, 1.5),
      vivaQ: 'How do you find eigenvalues mathematically?',
      vivaA: 'By setting det(A - λI) = 0 (the characteristic polynomial) and finding the roots λ. For a 2×2 matrix, this is λ² - tr(A)λ + det(A) = 0.',
      html: `
        <h4>What are Eigenvectors?</h4>
        <p>Most vectors get knocked off their span line when transformed by <em>A</em>. An <strong>eigenvector</strong> is a special invariant vector that does NOT rotate—it only scales along its original line!</p>
        <div class="theory-highlight">A · v = λ · v   (where λ is the scalar eigenvalue factor)</div>
        <ul>
          <li><strong>λ > 1:</strong> Vector stretches outwards along span line.</li>
          <li><strong>0 < λ < 1:</strong> Vector contracts inwards towards origin.</li>
          <li><strong>λ < 0:</strong> Vector flips to opposite direction along invariant axis.</li>
          <li><strong>Complex λ:</strong> Space undergoes pure rotation or spiral—no real vectors remain invariant!</li>
        </ul>
      `
    },
    'diagonalization': {
      title: '4. Diagonalization & Matrix Powers (A = PDP⁻¹)',
      formula: 'A = P · D · P⁻¹   ⟹   Aᵏ = P · diag(λ₁ᵏ, λ₂ᵏ) · P⁻¹',
      tabTarget: 'mult',
      presetMatrix: new Matrix2x2(1.5, 0.5, 0.0, 0.8),
      vivaQ: 'When is an n×n matrix diagonalizable?',
      vivaA: 'When it possesses n linearly independent eigenvectors. This is guaranteed if all n eigenvalues are distinct real numbers.',
      html: `
        <h4>Why Diagonalize?</h4>
        <p>If a matrix has <em>n</em> linearly independent eigenvectors, we can switch our coordinate system to the <strong>eigenbasis</strong>. In this frame of reference, matrix operations become completely uncoupled!</p>
        <div class="theory-highlight">A = P · D · P⁻¹   (where D is a diagonal matrix of eigenvalues)</div>
        <p>Computing <em>A¹⁰⁰</em> directly requires 100 expensive matrix multiplications. With diagonalization, powers are computed instantly:</p>
        <div class="theory-highlight">Aᵏ = P · diag(λ₁ᵏ, λ₂ᵏ) · P⁻¹</div>
      `
    },
    'svd': {
      title: '5. Singular Value Decomposition (SVD)',
      formula: 'A = U · Σ · Vᵀ   (σᵢ = √λᵢ(AᵀA))',
      tabTarget: 'transform',
      presetMatrix: new Matrix2x2(1.4, 0.6, 0.2, 0.9),
      vivaQ: 'What are the geometric roles of U, Σ, and Vᵀ in SVD?',
      vivaA: 'Vᵀ rotates the input space to align with principal directions, Σ stretches space along orthogonal axes by singular values σ₁ and σ₂, and U applies a final rotation in the output space.',
      html: `
        <h4>The Master Matrix Factorization</h4>
        <p>Eigenvalues only work for square matrices (n × n). <strong>SVD</strong> works for *any* matrix of any rectangular dimensions (m × n), making it the foundation of PCA and modern LLM compression!</p>
        <div class="theory-highlight">A = U · Σ · Vᵀ<br>Rotate (Vᵀ) ⟶ Stretch (Σ by σ₁, σ₂) ⟶ Rotate (U)</div>
        <p>Geometrically, SVD proves that <strong>any linear transformation maps a unit circle into a hyper-ellipse</strong>. The semi-axis lengths of this ellipse are the singular values σ₁ and σ₂.</p>
      `
    },
    'rank-nullity': {
      title: '6. Rank-Nullity Theorem & Dimensional Collapse',
      formula: 'dim(ker A) + dim(im A) = n   (Nullity + Rank = n)',
      tabTarget: 'transform',
      presetMatrix: new Matrix2x2(1.0, 1.0, 1.0, 1.0),
      vivaQ: 'State the Rank-Nullity Theorem and its physical interpretation.',
      vivaA: 'dim(ker A) + dim(im A) = n. Every dimension of the input space is either preserved in the output column space (image) or crushed to the origin in the nullspace (kernel).',
      html: `
        <h4>The Fundamental Conservation Law</h4>
        <p>When a matrix transforms space, every input vector either lands somewhere in the <strong>Column Space (Image)</strong> or gets squashed to the zero vector in the <strong>Nullspace (Kernel)</strong>.</p>
        <div class="theory-highlight">dim(ker A) + dim(im A) = n   (Rank-Nullity Theorem)</div>
        <p>For a 2×2 matrix with det = 0:</p>
        <ul>
          <li>Rank = 1 (Column space is a 1D line)</li>
          <li>Nullity = 1 (Kernel is an orthogonal 1D line compressed to 0)</li>
          <li>1 + 1 = 2 total dimensions accounted for!</li>
        </ul>
      `
    },
    'optimizers': {
      title: '7. Gradient Descent & Loss Landscapes',
      formula: 'w_{t+1} = w_t - α · ∇L(w_t)   (Adam: m̂ / (√v̂ + ε))',
      tabTarget: 'loss',
      presetMatrix: null,
      vivaQ: 'Why does standard SGD struggle in narrow ravines compared to Momentum and Adam?',
      vivaA: 'In narrow ravines with high condition numbers, the gradient oscillates wildly across steep ravine walls rather than moving down the gentle floor. Momentum adds inertia to cancel transversal oscillations, while Adam adapts individual parameter learning rates.',
      html: `
        <h4>How AI Learns</h4>
        <p>Training machine learning models means finding weights <em>(w₁, w₂)</em> that minimize a scalar loss function <em>L(w)</em>. The negative gradient <strong>-∇L</strong> points in the direction of steepest descent:</p>
        <div class="theory-highlight">w_{t+1} = w_t - α · ∇L(w_t)</div>
        <ul>
          <li><strong>SGD:</strong> Raw gradient steps; oscillates aggressively in ill-conditioned ravines.</li>
          <li><strong>Momentum:</strong> Adds velocity momentum <em>v</em> like a rolling bowling ball to bypass saddle points.</li>
          <li><strong>RMSprop:</strong> Normalizes gradient steps by running variance to scale unequal coordinate axes.</li>
          <li><strong>Adam:</strong> Combines Momentum + RMSprop with bias correction—the industry standard for training LLMs!</li>
        </ul>
      `
    },
    'backprop': {
      title: '8. Automatic Differentiation & The Chain Rule',
      formula: '∂L / ∂w = (∂L / ∂y) · (∂y / ∂w)',
      tabTarget: 'autograd',
      presetMatrix: null,
      vivaQ: 'Why is reverse-mode autodiff preferred over forward-mode for deep neural networks?',
      vivaA: 'Neural networks have millions of input weights but only a single scalar output (Loss L). Reverse-mode autodiff computes gradients for ALL weights in a single backward pass O(1), whereas forward-mode would require millions of passes O(N).',
      html: `
        <h4>Reverse-Mode Autodiff</h4>
        <p>To train deep neural networks with billions of parameters, calculating numerical finite differences would require billions of forward passes. Backpropagation calculates exact gradients in <strong>one single pass</strong> via the chain rule on a Directed Acyclic Graph (DAG):</p>
        <div class="theory-highlight">∂L / ∂w = (∂L / ∂y) · (∂y / ∂w)</div>
        <p>During the forward pass, node activations flow from inputs to output. During the backward pass, adjoint gradients propagate in reverse from output to inputs!</p>
      `
    }
  };
  LECTURE_NOTES['eigen'] = LECTURE_NOTES['eigenvalues'];

  function updateNotesTopic(key) {
    var item = LECTURE_NOTES[key] || LECTURE_NOTES['matrix-transform'];
    state.currentNotesTopic = key;

    // Track explored topics
    if (!state.exploredNotesTopics) state.exploredNotesTopics = [];
    if (state.exploredNotesTopics.indexOf(key) === -1) {
      state.exploredNotesTopics.push(key);
    }
    var badge = $('notes-progress-badge');
    if (badge) {
      badge.textContent = 'Explored: ' + state.exploredNotesTopics.length + '/8';
    }

    // Synchronize UI active button
    document.querySelectorAll('.btn-notes-topic').forEach(function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-topic') === key);
    });
    var sel = $('notes-topic-select');
    if (sel && sel.value !== key) sel.value = key;

    // Render Note Content Card
    var container = $('notes-content-container');
    if (container) {
      container.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;">
          <h3>${item.title}</h3>
        </div>
        <div class="theory-highlight" style="font-size: 0.82rem; margin: 0.2rem 0 0.5rem 0;">
          <strong>Identity:</strong> <code>${item.formula}</code>
        </div>
        ${item.html}
        <div class="viva-tip" style="margin-top: 0.6rem;">
          <button class="viva-accordion-toggle" id="btn-toggle-viva">
            <span>🎓 University Viva Question</span> <span style="font-size: 0.7rem; color: #fde68a;">[Click to Reveal]</span>
          </button>
          <div style="font-weight: 600; margin-top: 0.35rem; color: #fde68a;">${item.vivaQ}</div>
          <div class="viva-answer hidden" id="viva-answer-content">${item.vivaA}</div>
        </div>
      `;

      // Wire viva reveal toggle
      var btnToggleViva = $('btn-toggle-viva');
      var ansContent = $('viva-answer-content');
      if (btnToggleViva && ansContent) {
        btnToggleViva.addEventListener('click', function () {
          var isHidden = ansContent.classList.toggle('hidden');
          btnToggleViva.querySelector('span:last-child').textContent = isHidden ? '[Click to Reveal]' : '[Hide Answer]';
        });
      }
    }

    render();
    updateAdaptiveHUD();
  }

  // ── Viva Prep Quiz Engine ─────────────────────────────────────────────────

  var QUIZ_QUESTIONS = [
    {
      category: '2D Transformations',
      q: '1. Geometrically, what does it mean if the determinant of a 2x2 matrix is negative (det(A) < 0)?',
      opts: [
        'A) Space expands infinitely',
        'B) Space has collapsed into a point',
        'C) Space is inverted / mirrored (orientation reversed)',
        'D) The matrix is an identity matrix'
      ],
      correct: 2,
      hint: 'Think about looking in a mirror or flipping a sheet of paper upside-down.',
      exp: 'Step 1: det(A) measures the signed area scaling factor.\nStep 2: Positive determinant preserves counter-clockwise chirality.\nStep 3: Negative determinant inverts orientation (mirror reflection).'
    },
    {
      category: 'Eigensystems',
      q: '2. If an eigenvalue λ = 0 for a matrix A, what does this guarantee geometrically and algebraically?',
      opts: [
        'A) The matrix is symmetric',
        'B) The matrix is singular (det(A) = 0 and has no inverse)',
        'C) The matrix is an orthogonal rotation',
        'D) All eigenvalues must be zero'
      ],
      correct: 1,
      hint: 'Recall that det(A) equals the product of all its eigenvalues: det(A) = λ₁ · λ₂.',
      exp: 'Step 1: det(A) = Π λᵢ. If λ₁ = 0, then det(A) = 0 · λ₂ = 0.\nStep 2: A zero determinant means the transformation flattens space into a lower dimension.\nStep 3: Hence, matrix A is singular and non-invertible.'
    },
    {
      category: 'Matrix Composition',
      q: '3. Why is matrix multiplication generally non-commutative (AB ≠ BA)?',
      opts: [
        'A) Rounding errors in floating point arithmetic',
        'B) Because consecutive spatial transformations depend on order (e.g. rotate then shear ≠ shear then rotate)',
        'C) Because vectors do not have inverses',
        'D) It is actually commutative for all matrices'
      ],
      correct: 1,
      hint: 'Imagine rotating a card 90° then shifting it right, vs shifting it right first then rotating.',
      exp: 'Step 1: Matrix multiplication represents sequential composition of linear maps.\nStep 2: The order of successive geometric maps matters in Euclidean space.\nStep 3: Rotating then shearing leaves vectors in different positions than shearing then rotating.'
    },
    {
      category: 'Singular Value Decomposition',
      q: '4. What shape does a 2D unit circle always transform into under any linear transformation?',
      opts: [
        'A) A triangle',
        'B) An ellipse (or collapsed line segment)',
        'C) A square',
        'D) A parabola'
      ],
      correct: 1,
      hint: 'SVD states that A = U Σ V^T decomposes any linear map into rotation, axial stretch, and rotation.',
      exp: 'Step 1: By the SVD theorem, A = U Σ V^T.\nStep 2: The unit circle is rotated by V^T, stretched along orthogonal axes by singular values σ₁ and σ₂, and rotated by U.\nStep 3: The resulting geometric figure is always an ellipse.'
    },
    {
      category: 'Diagonalization',
      q: '5. In the diagonalization formula A = PDP⁻¹, what do the columns of matrix P represent?',
      opts: [
        'A) The gradient vectors',
        'B) The linearly independent eigenvectors of A',
        'C) The inverse determinant',
        'D) The standard basis vectors'
      ],
      correct: 1,
      hint: 'P is the change-of-basis matrix into the coordinate system where A acts as pure coordinate scaling.',
      exp: 'Step 1: When A acts on its eigenvectors vᵢ, Avᵢ = λᵢvᵢ.\nStep 2: Assembling columns P = [v₁ | v₂] diagonalizes the action into D = diag(λ₁, λ₂).\nStep 3: Thus, P consists of the linearly independent eigenvectors of A.'
    },
    {
      category: 'Rank-Nullity Theorem',
      q: '6. According to the Rank-Nullity Theorem, if a 2x2 matrix has rank 1, what is the dimension of its nullspace (kernel)?',
      opts: [
        'A) 0',
        'B) 1',
        'C) 2',
        'D) Infinity'
      ],
      correct: 1,
      hint: 'The fundamental theorem states: dim(Col A) + dim(Null A) = total number of columns (n).',
      exp: 'Step 1: Rank-Nullity Theorem states rank(A) + nullity(A) = n.\nStep 2: Here n = 2 columns and rank = 1 (1D line output).\nStep 3: Nullity = 2 - 1 = 1 (a 1D line squashed completely to the zero vector).'
    },
    {
      category: 'Vector Geometry',
      q: '7. What does the dot product u · v = 0 indicate about two non-zero vectors in ℝ²?',
      opts: [
        'A) They are parallel',
        'B) They are orthogonal (perpendicular, angle 90°)',
        'C) One vector is the zero vector',
        'D) They have equal magnitude'
      ],
      correct: 1,
      hint: 'The geometric definition of the dot product is u · v = ||u|| ||v|| cos(θ).',
      exp: 'Step 1: u · v = ||u|| ||v|| cos(θ).\nStep 2: Since ||u|| ≠ 0 and ||v|| ≠ 0, u · v = 0 requires cos(θ) = 0.\nStep 3: Therefore, the angle θ between the vectors is 90° (orthogonal).'
    },
    {
      category: 'LossLab Optimizers',
      q: '8. In Deep Learning optimization, why is the Adam optimizer preferred over standard SGD on complex ravine surfaces?',
      opts: [
        'A) It does not require calculating gradients',
        'B) It combines Momentum (velocity) with RMSprop (adaptive per-parameter learning rate)',
        'C) It only works on convex quadratic functions',
        'D) It is slower and requires more memory'
      ],
      correct: 1,
      hint: 'Adam tracks running 1st moments (momentum) and 2nd moments (uncentered variance).',
      exp: 'Step 1: SGD violently oscillates across steep ravine walls with high condition numbers.\nStep 2: Adam dampens perpendicular oscillation via running first moment momentum.\nStep 3: Adam rescales step sizes along flat directions via running second moment variance.'
    },
    {
      category: 'Numerical Linear Algebra',
      q: '9. What does the Power Iteration algorithm compute by repeatedly multiplying a vector by matrix A (Aᵏx)?',
      opts: [
        'A) The smallest eigenvalue',
        'B) The dominant eigenvector (corresponding to the eigenvalue with largest magnitude)',
        'C) The determinant of A',
        'D) The trace of A'
      ],
      correct: 1,
      hint: 'The component along the largest eigenvalue grows exponentially as λ₁ᵏ.',
      exp: 'Step 1: Any starting vector can be expressed in the eigenvector basis: x = c₁v₁ + c₂v₂.\nStep 2: Multiplying k times gives Aᵏx = c₁λ₁ᵏv₁ + c₂λ₂ᵏv₂ = λ₁ᵏ[c₁v₁ + c₂(λ₂/λ₁)ᵏv₂].\nStep 3: Since |λ₂/λ₁| < 1, as k → ∞ the vector aligns purely with dominant eigenvector v₁.'
    },
    {
      category: 'MicroGraph Autograd',
      q: '10. What is the fundamental calculus principle used by Reverse-Mode Autograd (Backpropagation)?',
      opts: [
        'A) Simpson’s Rule',
        'B) The Chain Rule of calculus applied in reverse topological order',
        'C) Monte Carlo integration',
        'D) Cauchy-Schwarz Inequality'
      ],
      correct: 1,
      hint: 'Gradients flow backwards from scalar loss output to leaf inputs.',
      exp: 'Step 1: Neural networks are compositions of elementary mathematical operations.\nStep 2: The Chain Rule enables computing ∂L/∂w = (∂L/∂y) · (∂y/∂w).\nStep 3: Reverse topological traversal caches forward activations and accumulates upstream gradients in O(1) passes.'
    },
    {
      category: 'Linear Systems & Invertibility',
      q: '11. For a 2x2 matrix A, when does the linear system Ax = b have a unique solution for every vector b?',
      opts: [
        'A) Only when all entries are positive',
        'B) When det(A) ≠ 0 (matrix is non-singular and invertible)',
        'C) When trace(A) = 0',
        'D) When A is a diagonal matrix'
      ],
      correct: 1,
      hint: 'Consider whether the transformation maps ℝ² onto the entire 2D plane without collapsing dimensions.',
      exp: 'Step 1: A system Ax = b has a unique solution x = A⁻¹b if and only if A has an inverse.\nStep 2: Invertibility requires det(A) ≠ 0 and full rank 2.\nStep 3: If det(A) = 0, columns are collinear and space collapses to 1D, making general solutions impossible.'
    },
    {
      category: 'Vector Spaces & Cross Product',
      q: '12. What does the magnitude of the 2D cross product |u × v| = |u_x v_y - u_y v_x| geometrically represent?',
      opts: [
        'A) The perimeter of the triangle formed by u and v',
        'B) The area of the parallelogram spanned by vectors u and v',
        'C) The sum of lengths ||u|| + ||v||',
        'D) The projection of u onto v'
      ],
      correct: 1,
      hint: 'Think of the determinant of the 2x2 matrix with u and v as its column vectors.',
      exp: 'Step 1: |u × v| = ||u|| ||v|| |sin(θ)|.\nStep 2: In the spanned parallelogram, base = ||u|| and altitude = ||v|| |sin(θ)|.\nStep 3: Area = Base × Altitude = |u_x v_y - u_y v_x| = |det([u | v])|.'
    },
    {
      category: '3D Transformations',
      q: '13. What is the determinant of any 3D pure rotation matrix R ∈ SO(3)?',
      opts: [
        'A) 0',
        'B) +1',
        'C) -1',
        'D) π'
      ],
      correct: 1,
      hint: 'Pure rotations preserve volume and maintain right-handed coordinate chirality.',
      exp: 'Step 1: Rotations are orthogonal matrices (Rᵀ R = I), so det(R)² = 1, meaning det(R) = ±1.\nStep 2: Special orthogonal group SO(3) preserves coordinate orientation (no reflections).\nStep 3: Therefore, det(R) = +1 strictly for all proper rigid-body rotations.'
    },
    {
      category: 'Curvature & Hessians',
      q: '14. In multivariate optimization, what does a Hessian matrix ∇²L with one positive and one negative eigenvalue indicate at a critical point?',
      opts: [
        'A) A global minimum',
        'B) A saddle point (minimax point)',
        'C) A global maximum',
        'D) A flat plateau'
      ],
      correct: 1,
      hint: 'Along one principal direction the surface curves upwards, while along the other it curves downwards.',
      exp: 'Step 1: Eigenvalues of the Hessian represent principal curvatures of the loss surface.\nStep 2: λ₁ > 0 means concave up (local min along axis 1); λ₂ < 0 means concave down (local max along axis 2).\nStep 3: This mixed signature defines an indefinite saddle point.'
    },
    {
      category: 'Matrix Invariants',
      q: '15. For any square matrix A, how is the trace tr(A) related to its eigenvalues?',
      opts: [
        'A) tr(A) = λ₁ · λ₂ · ... · λₙ',
        'B) tr(A) = λ₁ + λ₂ + ... + λₙ (sum of all eigenvalues)',
        'C) tr(A) = max(λᵢ) - min(λᵢ)',
        'D) tr(A) = 1 / det(A)'
      ],
      correct: 1,
      hint: 'Remember the characteristic polynomial: det(λI - A) = λⁿ - tr(A)λⁿ⁻¹ + ... + (-1)ⁿ det(A).',
      exp: 'Step 1: By Vieta’s formulas on the characteristic polynomial, the sum of roots equals tr(A).\nStep 2: tr(A) is invariant under similarity transformations: tr(PDP⁻¹) = tr(D).\nStep 3: In the diagonal basis, the diagonal elements are eigenvalues, hence tr(A) = Σ λᵢ.'
    }
  ];

  var currentQuizIdx = 0;
  var quizScore = 0;
  var quizStreak = 0;
  var maxQuizStreak = 0;
  var userQuizAnswers = {};
  var quizHintVisible = false;

  function renderQuizQuestion() {
    var q = QUIZ_QUESTIONS[currentQuizIdx];
    var container = $('quiz-question-card');
    if (!container) return;

    var isAnswered = !!userQuizAnswers[currentQuizIdx];
    var recorded = userQuizAnswers[currentQuizIdx];

    // Topic tag & Badges
    if ($('quiz-topic-tag')) $('quiz-topic-tag').textContent = 'Topic: ' + q.category;
    if ($('quiz-score-badge')) $('quiz-score-badge').textContent = 'Score: ' + quizScore + '/' + QUIZ_QUESTIONS.length;
    if ($('quiz-streak-badge')) $('quiz-streak-badge').textContent = '🔥 Streak: ' + quizStreak;
    if ($('quiz-hint-text')) $('quiz-hint-text').textContent = q.hint;

    // Reset hint box visibility unless already open
    var hintBox = $('quiz-hint-box');
    if (hintBox) {
      if (!quizHintVisible) hintBox.classList.add('hidden');
      else hintBox.classList.remove('hidden');
    }

    // Build question prompt and options
    var html = '<div class="quiz-prompt">' + q.q + '</div><div class="quiz-options-list">';
    q.opts.forEach(function (opt, idx) {
      var optClass = 'quiz-option';
      if (isAnswered) {
        if (idx === q.correct) optClass += ' correct';
        else if (idx === recorded.selected) optClass += ' incorrect';
      }
      html += '<div class="' + optClass + '" data-idx="' + idx + '">' + opt + '</div>';
    });
    html += '</div>';

    // Feedback explanation
    if (isAnswered) {
      var statusColor = recorded.isCorrect ? '#6ee7b7' : '#fca5a5';
      var statusPrefix = recorded.isCorrect ? '✓ Correct!' : '✗ Incorrect.';
      html += '<div id="quiz-feedback-box" class="quiz-explanation">';
      html += '<strong style="color:' + statusColor + '">' + statusPrefix + '</strong> ' + q.hint;
      html += '<div class="quiz-step-derivation">' + q.exp + '</div>';
      html += '</div>';
    } else {
      html += '<div id="quiz-feedback-box" class="quiz-explanation hidden"></div>';
    }

    container.innerHTML = html;

    // Attach option clicks if not answered yet
    if (!isAnswered) {
      container.querySelectorAll('.quiz-option').forEach(function (el) {
        el.addEventListener('click', function () {
          var selectedIdx = parseInt(this.getAttribute('data-idx'), 10);
          checkQuizAnswer(selectedIdx, q);
        });
      });
    }

    // Update Summary Scorecard if applicable
    updateQuizScorecard();
    updateAdaptiveHUD();
  }

  function checkQuizAnswer(selectedIdx, q) {
    var isCorrect = (selectedIdx === q.correct);
    if (isCorrect) {
      quizScore++;
      quizStreak++;
      if (quizStreak > maxQuizStreak) maxQuizStreak = quizStreak;
    } else {
      quizStreak = 0;
    }

    userQuizAnswers[currentQuizIdx] = {
      selected: selectedIdx,
      isCorrect: isCorrect
    };

    renderQuizQuestion();
    render();
  }

  function updateQuizScorecard() {
    var summaryCard = $('quiz-summary-card');
    if (!summaryCard) return;

    var answeredCount = Object.keys(userQuizAnswers).length;
    if (answeredCount < 3) {
      summaryCard.classList.add('hidden');
      return;
    }

    var total = QUIZ_QUESTIONS.length;
    var pct = Math.round((quizScore / total) * 100);
    var grade = 'A+ (Linear Algebra Master)';
    var gradeColor = '#10b981';

    if (pct < 60) {
      grade = 'C (Needs Revision)';
      gradeColor = '#f59e0b';
    } else if (pct < 75) {
      grade = 'B (Proficient Student)';
      gradeColor = '#38bdf8';
    } else if (pct < 90) {
      grade = 'A (Advanced Practitioner)';
      gradeColor = '#6ee7b7';
    }

    summaryCard.classList.remove('hidden');
    summaryCard.innerHTML =
      '<div class="quiz-scorecard-header">' +
        '<span style="font-size: 0.76rem; font-weight: 600; color: #cbd5e1;">Mastery Scorecard (' + answeredCount + '/' + total + ' Completed)</span>' +
        '<span class="quiz-scorecard-grade" style="color:' + gradeColor + '">' + grade.split(' ')[0] + '</span>' +
      '</div>' +
      '<div style="font-size: 0.72rem; color: ' + gradeColor + '; font-weight: 500; margin-bottom: 0.35rem;">' + grade + '</div>' +
      '<div class="quiz-stat-grid">' +
        '<div class="quiz-stat-pill"><span>Score</span><strong>' + quizScore + '/' + total + '</strong></div>' +
        '<div class="quiz-stat-pill"><span>Accuracy</span><strong>' + pct + '%</strong></div>' +
        '<div class="quiz-stat-pill"><span>Max Streak</span><strong>🔥 ' + maxQuizStreak + '</strong></div>' +
      '</div>' +
      '<button id="btn-quiz-restart" class="btn-micro" style="width: 100%; margin-top: 0.4rem; padding: 0.35rem; color: #a7f3d0; border-color: rgba(16, 185, 129, 0.3); background: rgba(16, 185, 129, 0.1);">↺ Reset &amp; Retake Quiz</button>';

    var btnRestart = $('btn-quiz-restart');
    if (btnRestart) {
      btnRestart.addEventListener('click', function () {
        currentQuizIdx = 0;
        quizScore = 0;
        quizStreak = 0;
        maxQuizStreak = 0;
        userQuizAnswers = {};
        quizHintVisible = false;
        renderQuizQuestion();
        render();
      });
    }
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
    function syncAutogradPresetUI() {
      var p = state.autogradPreset;
      var secNeuron = $('section-ag-neuron-params');
      var secLoss = $('section-ag-loss-params');
      var secAffine = $('section-ag-affine-params');
      var secJacobian = $('section-ag-jacobian');

      if (secNeuron) secNeuron.classList.toggle('hidden', p !== 'neuron');
      if (secLoss) secLoss.classList.toggle('hidden', p !== 'loss');
      if (secAffine) secAffine.classList.toggle('hidden', p !== 'affine');
      if (secJacobian) secJacobian.classList.toggle('hidden', p !== 'affine');

      if (badgeAgStep) {
        if (state.autogradStep === 'backward') {
          badgeAgStep.textContent = '∇ Backprop Active';
          badgeAgStep.className = 'telemetry-badge badge-det-pos';
        } else if (state.autogradStep === 'forward') {
          badgeAgStep.textContent = 'Forward Evaluated';
          badgeAgStep.className = 'telemetry-badge badge-det-pos';
        } else {
          badgeAgStep.textContent = 'Status: Idle';
          badgeAgStep.className = 'telemetry-badge';
        }
      }
    }

    document.querySelectorAll('.btn-autograd-preset[data-graph]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.btn-autograd-preset').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        state.autogradPreset = this.getAttribute('data-graph');
        state.autogradStep = 'idle';
        state.autogradTapeStep = -1;
        if (autogradStatusText) {
          autogradStatusText.textContent = 'Preset loaded. Click Forward Pass or scrub parameters on left!';
        }
        syncAutogradPresetUI();
        render();
      });
    });

    // Neuron Parameter Sliders
    var bindAgSlider = function (id, valId, stateKey, isFloat) {
      var sl = $(id);
      var vl = $(valId);
      if (sl) {
        sl.addEventListener('input', function () {
          var v = isFloat ? parseFloat(this.value) : parseInt(this.value, 10);
          state[stateKey] = v;
          if (vl) vl.textContent = v.toFixed(isFloat ? (id.indexOf('ypred') !== -1 || id.indexOf('ytrue') !== -1 ? 2 : 1) : 0);
          render();
        });
      }
    };

    bindAgSlider('slider-ag-x1', 'val-ag-x1', 'ag_x1', true);
    bindAgSlider('slider-ag-w1', 'val-ag-w1', 'ag_w1', true);
    bindAgSlider('slider-ag-x2', 'val-ag-x2', 'ag_x2', true);
    bindAgSlider('slider-ag-w2', 'val-ag-w2', 'ag_w2', true);
    bindAgSlider('slider-ag-b', 'val-ag-b', 'ag_b', true);

    // MSE Loss Sliders
    bindAgSlider('slider-ag-ypred', 'val-ag-ypred', 'ag_yPred', true);
    bindAgSlider('slider-ag-ytrue', 'val-ag-ytrue', 'ag_yTrue', true);

    // Affine Sliders
    bindAgSlider('slider-ag-aff-x1', 'val-ag-aff-x1', 'ag_aff_x1', true);
    bindAgSlider('slider-ag-aff-x2', 'val-ag-aff-x2', 'ag_aff_x2', true);
    bindAgSlider('slider-ag-aff-w11', 'val-ag-aff-w11', 'ag_aff_w11', true);
    bindAgSlider('slider-ag-aff-w22', 'val-ag-aff-w22', 'ag_aff_w22', true);

    if (btnForwardPass) {
      btnForwardPass.addEventListener('click', function () {
        state.autogradStep = 'forward';
        state.autogradTapeStep = -1;
        if (autogradStatusText) {
          autogradStatusText.textContent = 'Forward pass complete: intermediate activation values computed!';
        }
        syncAutogradPresetUI();
        render();
      });
    }

    if (btnBackwardPass) {
      btnBackwardPass.addEventListener('click', function () {
        state.autogradStep = 'backward';
        state.autogradTapeStep = 999;
        if (autogradStatusText) {
          autogradStatusText.textContent = 'Backprop complete: ∂L/∂w gradients accumulated along DAG via chain rule!';
        }
        syncAutogradPresetUI();
        render();
      });
    }

    if (btnStepTape) {
      btnStepTape.addEventListener('click', function () {
        state.autogradStep = 'backward';
        var maxSteps = state.autogradPreset === 'neuron' ? 9 : (state.autogradPreset === 'loss' ? 4 : 6);
        state.autogradTapeStep = (state.autogradTapeStep + 1) % maxSteps;

        if (autogradStatusText) {
          if (state.autogradPreset === 'neuron') {
            var msgs = [
              'Tape Step 1/9: Output ReLU gradient initialized to ∂a/∂a = 1.00',
              'Tape Step 2/9: Affine Sum ∂a/∂z = (z > 0 ? 1 : 0) propagates backward',
              'Tape Step 3/9: Bias gradient ∂a/∂b = ∂a/∂z · 1 = ' + (state.ag_b > 0 ? '+1.00' : '0.00'),
              'Tape Step 4/9: Product 2 ∂a/∂p₂ = ∂a/∂z · 1',
              'Tape Step 5/9: Product 1 ∂a/∂p₁ = ∂a/∂z · 1',
              'Tape Step 6/9: Weight 2 gradient ∂a/∂w₂ = (∂a/∂p₂) · x₂ = ' + (state.ag_x2).toFixed(2),
              'Tape Step 7/9: Input 2 gradient ∂a/∂x₂ = (∂a/∂p₂) · w₂ = ' + (state.ag_w2).toFixed(2),
              'Tape Step 8/9: Weight 1 gradient ∂a/∂w₁ = (∂a/∂p₁) · x₁ = ' + (state.ag_x1).toFixed(2),
              'Tape Step 9/9: Input 1 gradient ∂a/∂x₁ = (∂a/∂p₁) · w₁ = ' + (state.ag_w1).toFixed(2)
            ];
            autogradStatusText.textContent = msgs[state.autogradTapeStep] || 'Tape complete!';
          } else if (state.autogradPreset === 'loss') {
            var diffVal = (state.ag_yPred - state.ag_yTrue);
            var msgsL = [
              'Tape Step 1/4: Output Loss gradient initialized to ∂L/∂L = 1.00',
              'Tape Step 2/4: Residual Error gradient ∂L/∂diff = 2 · diff = ' + (2 * diffVal).toFixed(2),
              'Tape Step 3/4: Model Weight update direction ∂L/∂y_pred = +2·diff = ' + (2 * diffVal).toFixed(2),
              'Tape Step 4/4: Target sensitivity ∂L/∂y_true = -2·diff = ' + (-2 * diffVal).toFixed(2)
            ];
            autogradStatusText.textContent = msgsL[state.autogradTapeStep] || 'Tape complete!';
          } else {
            var msgsA = [
              'Tape Step 1/6: Scalar L2 Loss gradient initialized to ∂L/∂L = 1.00',
              'Tape Step 2/6: Output Affine Vector gradient ∂L/∂y = y',
              'Tape Step 3/6: Linear map intermediate gradient ∂L/∂(Wx) = ∂L/∂y · I',
              'Tape Step 4/6: Bias vector gradient ∂L/∂b = y',
              'Tape Step 5/6: Weight matrix gradient ∇_W L = y ⊗ x',
              'Tape Step 6/6: Input vector gradient ∇_x L = Wᵀ y (Layer Jacobian Adjoint)'
            ];
            autogradStatusText.textContent = msgsA[state.autogradTapeStep] || 'Tape complete!';
          }
        }
        syncAutogradPresetUI();
        render();
      });
    }

    if (btnResetAutograd) {
      btnResetAutograd.addEventListener('click', function () {
        state.ag_x1 = 1.5; state.ag_w1 = 0.8; state.ag_x2 = -1.0; state.ag_w2 = 1.2; state.ag_b = 0.3;
        state.ag_yPred = 0.85; state.ag_yTrue = 0.50;
        state.ag_aff_x1 = 1.0; state.ag_aff_x2 = -0.5; state.ag_aff_w11 = 1.2; state.ag_aff_w22 = 0.9;
        state.autogradStep = 'idle';
        state.autogradTapeStep = -1;

        var setVal = function (id, v, fixed) {
          var sl = $(id);
          if (sl) sl.value = v;
          var vl = $('val-' + id.replace('slider-', ''));
          if (vl) vl.textContent = v.toFixed(fixed || 1);
        };
        setVal('slider-ag-x1', 1.5);
        setVal('slider-ag-w1', 0.8);
        setVal('slider-ag-x2', -1.0);
        setVal('slider-ag-w2', 1.2);
        setVal('slider-ag-b', 0.3);
        setVal('slider-ag-ypred', 0.85, 2);
        setVal('slider-ag-ytrue', 0.50, 2);
        setVal('slider-ag-aff-x1', 1.0);
        setVal('slider-ag-aff-x2', -0.5);
        setVal('slider-ag-aff-w11', 1.2);
        setVal('slider-ag-aff-w22', 0.9);

        if (autogradStatusText) {
          autogradStatusText.textContent = 'Values reset to standard tutorial presets. Click Forward Pass!';
        }
        syncAutogradPresetUI();
        render();
      });
    }

    // Notes Topic Selector & Pill Buttons
    document.querySelectorAll('.btn-notes-topic[data-topic]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var topicKey = this.getAttribute('data-topic');
        updateNotesTopic(topicKey);
      });
    });

    var notesSelect = $('notes-topic-select');
    if (notesSelect) {
      notesSelect.addEventListener('change', function () {
        updateNotesTopic(this.value);
      });
    }

    // Instant Search Filter for Theory Topics
    var notesSearchInput = $('notes-search-input');
    if (notesSearchInput) {
      notesSearchInput.addEventListener('input', function () {
        var q = this.value.toLowerCase().trim();
        document.querySelectorAll('.btn-notes-topic').forEach(function (btn) {
          var tKey = btn.getAttribute('data-topic') || '';
          var text = btn.textContent.toLowerCase();
          var matched = tKey.indexOf(q) !== -1 || text.indexOf(q) !== -1;
          btn.style.display = matched ? 'block' : 'none';
        });
      });
    }

    // One-Click Formula Copy Button
    var btnCopyFormula = $('btn-copy-formula');
    if (btnCopyFormula) {
      btnCopyFormula.addEventListener('click', function () {
        var curr = LECTURE_NOTES[state.currentNotesTopic] || LECTURE_NOTES['matrix-transform'];
        var text = curr.formula;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text);
        }
        var orig = btnCopyFormula.textContent;
        btnCopyFormula.textContent = 'Copied! ✓';
        btnCopyFormula.style.color = '#10b981';
        btnCopyFormula.style.borderColor = '#10b981';
        setTimeout(function () {
          btnCopyFormula.textContent = orig;
          btnCopyFormula.style.color = '';
          btnCopyFormula.style.borderColor = '';
        }, 1800);
      });
    }

    // "Experiment in Tab" Action
    var btnNotesExperiment = $('btn-notes-experiment');
    if (btnNotesExperiment) {
      btnNotesExperiment.addEventListener('click', function () {
        var curr = LECTURE_NOTES[state.currentNotesTopic] || LECTURE_NOTES['matrix-transform'];
        if (curr.presetMatrix) {
          state.matrix = curr.presetMatrix.clone();
          syncMatrixInputs();
          updateTelemetry();
        }
        setMode(curr.tabTarget);
      });
    }

    // Viva Quiz Navigation & Action Buttons
    var btnQuizNext = $('btn-quiz-next');
    if (btnQuizNext) {
      btnQuizNext.addEventListener('click', function () {
        currentQuizIdx = (currentQuizIdx + 1) % QUIZ_QUESTIONS.length;
        quizHintVisible = false;
        renderQuizQuestion();
        render();
      });
    }

    var btnQuizPrev = $('btn-quiz-prev');
    if (btnQuizPrev) {
      btnQuizPrev.addEventListener('click', function () {
        currentQuizIdx = (currentQuizIdx - 1 + QUIZ_QUESTIONS.length) % QUIZ_QUESTIONS.length;
        quizHintVisible = false;
        renderQuizQuestion();
        render();
      });
    }

    var btnQuizHint = $('btn-quiz-hint');
    if (btnQuizHint) {
      btnQuizHint.addEventListener('click', function () {
        quizHintVisible = !quizHintVisible;
        var hintBox = $('quiz-hint-box');
        if (hintBox) {
          if (quizHintVisible) hintBox.classList.remove('hidden');
          else hintBox.classList.add('hidden');
        }
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
        if (step1Btn) step1Btn.textContent = '1: Map B';
        if (step2Btn) step2Btn.textContent = '2: Map AB';
      } else {
        if (step1Btn) step1Btn.textContent = '1: Map A';
        if (step2Btn) step2Btn.textContent = '2: Map BA';
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

  function updateAdaptiveHUD(pos, world) {
    var coordsEl = $('hud-coords');
    if (!coordsEl) return;
    var wX = world ? world.x : 0;
    var wY = world ? world.y : 0;

    if (state.mode === 'transform') {
      coordsEl.innerHTML = '2D Cursor: <strong>[' + wX.toFixed(2) + ', ' + wY.toFixed(2) + ']</strong>';
    } else if (state.mode === 'eigen') {
      var pDeg = (state.eigenProbe.angle() * 180 / Math.PI);
      if (pDeg < 0) pDeg += 360;
      coordsEl.innerHTML = 'Probe θ: <strong>' + pDeg.toFixed(1) + '°</strong> • Cursor: <strong>[' + wX.toFixed(2) + ', ' + wY.toFixed(2) + ']</strong>';
    } else if (state.mode === 'mult') {
      var pct = Math.round(state.multT * 100);
      coordsEl.innerHTML = 'Composition: <strong>' + state.multOrder + '</strong> • Tween (t): <strong>' + pct + '%</strong>';
    } else if (state.mode === 'vectors') {
      coordsEl.innerHTML = '2D Cursor: <strong>[' + wX.toFixed(2) + ', ' + wY.toFixed(2) + ']</strong>';
    } else if (state.mode === '3d') {
      coordsEl.innerHTML = '3D Camera Orbit: <strong>Yaw ' + Math.round(state.camYaw) + '°, Pitch ' + Math.round(state.camPitch) + '°</strong>';
    } else if (state.mode === 'loss') {
      var lFn = LossFunctions[state.lossKey] || LossFunctions.bowl;
      var curVal = lFn.evaluate(wX, wY);
      coordsEl.innerHTML = 'Landscape: <strong>' + lFn.name + '</strong> • L(cursor): <strong>' + formatSafe(curVal) + '</strong>';
    } else if (state.mode === 'autograd') {
      if (typeof state.hoverAutogradNode === 'number' && currentAutogradNodes && currentAutogradNodes[state.hoverAutogradNode]) {
        var nd = currentAutogradNodes[state.hoverAutogradNode];
        coordsEl.innerHTML = 'Node: <strong style="color:#38bdf8;">' + nd.label + '</strong> (val: ' + formatSafe(nd.val) + ', grad: ' + (nd.grad !== null && nd.grad !== undefined ? formatSafe(nd.grad) : 'pending') + ')';
      } else {
        coordsEl.innerHTML = 'MicroGraph DAG: <strong>' + state.autogradPreset.toUpperCase() + '</strong> (Status: <strong>' + state.autogradStep.toUpperCase() + '</strong>)';
      }
    } else if (state.mode === 'notes') {
      var topicObj = (typeof LECTURE_NOTES !== 'undefined') ? LECTURE_NOTES[state.currentNotesTopic] : null;
      coordsEl.innerHTML = 'Theory Vault: <strong style="color:#a5b4fc;">' + (topicObj ? topicObj.title : state.currentNotesTopic) + '</strong>';
    } else if (state.mode === 'quiz') {
      coordsEl.innerHTML = 'Viva Prep Quiz: <strong>Q' + (currentQuizIdx + 1) + ' of ' + (typeof QUIZ_QUESTIONS !== 'undefined' ? QUIZ_QUESTIONS.length : 15) + '</strong> • Score: <strong>' + quizScore + '/' + (typeof QUIZ_QUESTIONS !== 'undefined' ? QUIZ_QUESTIONS.length : 15) + '</strong>';
    }
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
    updateAdaptiveHUD();
    updateTelemetry();

    if (newMode === 'loss') {
      initLossParticles();
    } else if (newMode === 'autograd') {
      if (typeof syncAutogradPresetUI === 'function') syncAutogradPresetUI();
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
    if (typeof syncAutogradPresetUI === 'function') syncAutogradPresetUI();
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
