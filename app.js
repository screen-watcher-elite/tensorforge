/**
 * TensorForge — Interactive Canvas Renderer & Application Logic
 * 60 FPS Canvas coordinate engine with vector dragging, eigen span lines,
 * warped grid visualization, and step-by-step matrix composition.
 */

(function () {
  'use strict';

  var Engine = window.TensorForgeEngine;
  var Vector2D = Engine.Vector2D;
  var Matrix2x2 = Engine.Matrix2x2;

  // ── State ─────────────────────────────────────────────────────────────────

  var state = {
    mode: 'transform', // 'transform' | 'eigen' | 'mult' | 'vectors'
    matrix: new Matrix2x2(1.5, 0.5, 0.5, 1.2), // Default interesting transformation
    matrixB: new Matrix2x2(0.8, -0.6, 0.6, 0.8), // Rotation matrix for composition
    multT: 0, // Interpolation factor for A * B [0 -> 1]
    multOrder: 'AB', // 'AB' or 'BA'
    
    // Custom test vector
    customVec: new Vector2D(1.0, 1.0),
    showCustomVec: true,
    showTransformedGrid: true,
    showEigenSpans: true,

    // Vector Sandbox mode
    vecU: new Vector2D(2.0, 1.0),
    vecV: new Vector2D(1.0, 2.0),

    // Viewport transform
    scale: 65, // Pixels per math unit
    panX: 0,
    panY: 0,

    // Interaction
    draggingTarget: null, // 'i' | 'j' | 'v' | 'u' | 'pan'
    dragStartMouse: { x: 0, y: 0 },
    dragStartPan: { x: 0, y: 0 },
    hoverTarget: null
  };

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

  var matAInput = $('mat-a');
  var matBInput = $('mat-b');
  var matCInput = $('mat-c');
  var matDInput = $('mat-d');

  var readoutDet = $('telemetry-det');
  var badgeDet = $('badge-det');
  var readoutTrace = $('telemetry-trace');
  var readoutRank = $('telemetry-rank');
  var eigenRow1 = $('eigen-val-1');
  var eigenRow2 = $('eigen-val-2');

  var rotationSlider = $('slider-rotation');
  var rotationValue = $('val-rotation');
  var shearSlider = $('slider-shear');
  var shearValue = $('val-shear');

  var multDrawer = $('mult-drawer');
  var multSlider = $('slider-mult');
  var multTDisplay = $('val-mult-t');
  var btnSwapMult = $('btn-swap-mult');

  var hudCoords = $('hud-coords');

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

    ctx.scale(dpr, dpr);
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

    var origin = worldToScreen(0, 0);

    // 1. Static Cartesian background grid
    drawBackgroundGrid();

    if (state.mode === 'vectors') {
      // Sandbox mode: draw vector U and V + dot product projection
      drawVectorSandbox();
    } else {
      // Linear algebra transformation mode
      var activeMatrix = getActiveMatrixForRender();

      // 2. Transformed coordinate grid (3Blue1Brown style warped lines)
      if (state.showTransformedGrid) {
        drawTransformedGrid(activeMatrix);
      }

      // 3. Unit Square area fill showing determinant
      drawDeterminantArea(activeMatrix);

      // 4. Invariant Eigenvector Span lines
      if (state.showEigenSpans && state.mode !== 'mult') {
        drawEigenSpanLines(activeMatrix);
      }

      // 5. Axes
      drawAxes();

      // 6. Basis vectors i-hat and j-hat
      drawBasisVectors(activeMatrix);

      // 7. Custom vector V and transformed T(V)
      if (state.showCustomVec) {
        drawCustomVector(activeMatrix);
      }
    }
  }

  function getActiveMatrixForRender() {
    if (state.mode === 'mult') {
      // Interpolate between Identity -> Matrix B -> A * B
      var t = state.multT;
      var M1 = state.multOrder === 'AB' ? state.matrixB : state.matrix;
      var M2 = state.multOrder === 'AB' ? state.matrix : state.matrixB;
      var product = M2.multiply(M1);

      if (t <= 0.5) {
        // Phase 1: Identity to first matrix
        var localT = t * 2;
        return Matrix2x2.lerp(Matrix2x2.identity(), M1, localT);
      } else {
        // Phase 2: First matrix to full composed product
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
    // Lines parallel to transformed j-hat
    for (var i = -range; i <= range; i++) {
      var base = col1.scale(i);
      var pStart = worldToScreen(base.x - col2.x * range, base.y - col2.y * range);
      var pEnd = worldToScreen(base.x + col2.x * range, base.y + col2.y * range);
      ctx.moveTo(pStart.x, pStart.y);
      ctx.lineTo(pEnd.x, pEnd.y);
    }
    // Lines parallel to transformed i-hat
    for (var j = -range; j <= range; j++) {
      var base = col2.scale(j);
      var pStart = worldToScreen(base.x - col1.x * range, base.y - col1.y * range);
      var pEnd = worldToScreen(base.x + col1.x * range, base.y + col1.y * range);
      ctx.moveTo(pStart.x, pStart.y);
      ctx.lineTo(pEnd.x, pEnd.y);
    }
    ctx.stroke();
  }

  // ── Draw Unit Square / Determinant Parallelogram ──────────────────────────

  function drawDeterminantArea(matrix) {
    var o = worldToScreen(0, 0);
    var iHat = worldToScreen(matrix.a, matrix.c);
    var sum = worldToScreen(matrix.a + matrix.b, matrix.c + matrix.d);
    var jHat = worldToScreen(matrix.b, matrix.d);

    var det = matrix.determinant();

    ctx.beginPath();
    ctx.moveTo(o.x, o.y);
    ctx.lineTo(iHat.x, iHat.y);
    ctx.lineTo(sum.x, sum.y);
    ctx.lineTo(jHat.x, jHat.y);
    ctx.closePath();

    if (Math.abs(det) < Engine.EPSILON) {
      // Collapsed to 1D line
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (det > 0) {
      // Preserves orientation
      ctx.fillStyle = 'rgba(6, 182, 212, 0.14)';
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    } else {
      // Inverts orientation (flipped)
      ctx.fillStyle = 'rgba(245, 158, 11, 0.16)';
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    }

    // Center area label
    if (Math.abs(det) > 0.05) {
      var centerX = (o.x + iHat.x + sum.x + jHat.x) / 4;
      var centerY = (o.y + iHat.y + sum.y + jHat.y) / 4;
      ctx.font = '600 11px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillStyle = det > 0 ? '#67e8f9' : '#fcd34d';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Area: ' + Math.abs(det).toFixed(2), centerX, centerY);
    }
  }

  // ── Draw Invariant Eigenvector Span Lines ──────────────────────────────────

  function drawEigenSpanLines(matrix) {
    var eigens = Engine.solveEigensystem(matrix);
    if (!eigens.isReal || eigens.eigenvectors.length === 0) return;

    var colors = ['#f59e0b', '#a855f7'];
    var lineLength = 20;

    eigens.eigenvectors.forEach(function (ev, index) {
      var v = ev.vector;
      var p1 = worldToScreen(-v.x * lineLength, -v.y * lineLength);
      var p2 = worldToScreen(v.x * lineLength, v.y * lineLength);

      ctx.save();
      ctx.strokeStyle = colors[index % colors.length];
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 6]);
      ctx.globalAlpha = 0.6;

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      // Label on span line
      var labelPos = worldToScreen(v.x * 2.8, v.y * 2.8);
      ctx.fillStyle = colors[index % colors.length];
      ctx.font = '600 10px ' + getComputedStyle(document.body).getPropertyValue('--font-mono');
      ctx.fillText('Span(v' + (index + 1) + ') λ=' + ev.lambda.toFixed(2), labelPos.x + 6, labelPos.y - 6);

      ctx.restore();
    });
  }

  // ── Draw Standard Axes ────────────────────────────────────────────────────

  function drawAxes() {
    var o = worldToScreen(0, 0);

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';

    // X axis
    ctx.beginPath();
    ctx.moveTo(0, o.y);
    ctx.lineTo(viewWidth, o.y);
    ctx.stroke();

    // Y axis
    ctx.beginPath();
    ctx.moveTo(o.x, 0);
    ctx.lineTo(o.x, viewHeight);
    ctx.stroke();

    // Origin indicator
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.beginPath();
    ctx.arc(o.x, o.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // ── Draw Basis Vectors (i-hat & j-hat) ─────────────────────────────────────

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

    // Input vector (Faint/dashed)
    ctx.save();
    ctx.setLineDash([4, 4]);
    drawArrow(o.x, o.y, vInput.x, vInput.y, 'rgba(16, 185, 129, 0.5)', 1.5);
    ctx.restore();

    drawVectorHandle(vInput.x, vInput.y, '#10b981', 'v', state.hoverTarget === 'v');

    // Transformed vector (Solid green)
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

    // Projection vector
    drawArrow(o.x, o.y, projPos.x, projPos.y, '#f59e0b', 3);
    ctx.restore();
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

    // Stem
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Arrowhead
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
    ctx.arc(x, y, isHovered ? 7 : 5, 0, Math.PI * 2);
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
    if (eigens.isReal) {
      eigenRow1.textContent = 'λ₁ = ' + eigens.eigenvalues[0].value.toFixed(2);
      eigenRow2.textContent = 'λ₂ = ' + eigens.eigenvalues[1].value.toFixed(2);
    } else {
      var re = eigens.eigenvalues[0].real.toFixed(2);
      var im = eigens.eigenvalues[0].imag.toFixed(2);
      eigenRow1.textContent = 'λ₁ = ' + re + ' + ' + im + 'i';
      eigenRow2.textContent = 'λ₂ = ' + re + ' - ' + im + 'i';
    }
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

  // ── Mouse & Touch Event Mechanics ─────────────────────────────────────────

  function getMousePos(e) {
    var rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }

  function checkHitTarget(sx, sy) {
    var threshold = 18; // px radius for grab handle

    if (state.mode === 'vectors') {
      var uPos = worldToScreen(state.vecU.x, state.vecU.y);
      var vPos = worldToScreen(state.vecV.x, state.vecV.y);

      if (Math.hypot(sx - uPos.x, sy - uPos.y) < threshold) return 'u';
      if (Math.hypot(sx - vPos.x, sy - vPos.y) < threshold) return 'v_sandbox';
      return null;
    }

    var iPos = worldToScreen(state.matrix.a, state.matrix.c);
    var jPos = worldToScreen(state.matrix.b, state.matrix.d);
    var vPos = worldToScreen(state.customVec.x, state.customVec.y);

    if (Math.hypot(sx - iPos.x, sy - iPos.y) < threshold) return 'i';
    if (Math.hypot(sx - jPos.x, sy - jPos.y) < threshold) return 'j';
    if (state.showCustomVec && Math.hypot(sx - vPos.x, sy - vPos.y) < threshold) return 'v';

    return null;
  }

  canvas.addEventListener('mousedown', function (e) {
    var pos = getMousePos(e);
    var hit = checkHitTarget(pos.x, pos.y);

    if (hit) {
      state.draggingTarget = hit;
    } else {
      // Pan canvas
      state.draggingTarget = 'pan';
      state.dragStartMouse = pos;
      state.dragStartPan = { x: state.panX, y: state.panY };
    }
  });

  window.addEventListener('mousemove', function (e) {
    var pos = getMousePos(e);
    var world = screenToWorld(pos.x, pos.y);

    // Update HUD coordinates
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

    // Snap to 0.25 if Shift is pressed
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
    } else if (state.draggingTarget === 'u') {
      state.vecU.x = snappedX;
      state.vecU.y = snappedY;
      render();
    } else if (state.draggingTarget === 'v_sandbox') {
      state.vecV.x = snappedX;
      state.vecV.y = snappedY;
      render();
    } else if (state.draggingTarget === 'pan') {
      var dx = pos.x - state.dragStartMouse.x;
      var dy = pos.y - state.dragStartMouse.y;
      state.panX = state.dragStartPan.x + dx;
      state.panY = state.dragStartPan.y + dy;
      render();
    }
  });

  window.addEventListener('mouseup', function () {
    state.draggingTarget = null;
    canvas.style.cursor = state.hoverTarget ? 'grab' : 'crosshair';
  });

  // Mouse wheel to zoom
  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    var zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
    state.scale = Math.min(180, Math.max(25, state.scale * zoomFactor));
    render();
  }, { passive: false });

  // ── Presets & Transformation Animations ───────────────────────────────────

  var PRESETS = {
    identity: Matrix2x2.identity(),
    rotate45: Matrix2x2.rotation(Math.PI / 4),
    rotate90: Matrix2x2.rotation(Math.PI / 2),
    shearX: Matrix2x2.shearX(1.2),
    shearY: Matrix2x2.shearY(1.0),
    scaleUp: Matrix2x2.scale(1.8, 1.4),
    reflectX: new Matrix2x2(1, 0, 0, -1),
    reflectY: new Matrix2x2(-1, 0, 0, 1),
    singular: new Matrix2x2(1, 1, 1, 1) // Det = 0 collapse
  };

  function applyPreset(name) {
    var target = PRESETS[name];
    if (!target) return;

    // Highlight active preset button
    document.querySelectorAll('.btn-preset').forEach(function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-preset') === name);
    });

    animController.start(state.matrix, target, 500);
  }

  // ── Setup UI Event Listeners ──────────────────────────────────────────────

  function initEvents() {
    // Resize
    window.addEventListener('resize', resizeCanvas);

    // Matrix inputs
    [matAInput, matBInput, matCInput, matDInput].forEach(function (inp) {
      inp.addEventListener('input', readMatrixInputs);
    });

    // Preset buttons
    document.querySelectorAll('.btn-preset[data-preset]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        applyPreset(this.getAttribute('data-preset'));
      });
    });

    // Rotation slider
    rotationSlider.addEventListener('input', function () {
      var deg = parseFloat(this.value);
      rotationValue.textContent = deg + '°';
      var rad = (deg * Math.PI) / 180;
      state.matrix = Matrix2x2.rotation(rad);
      syncMatrixInputs();
      updateTelemetry();
      render();
    });

    // Shear slider
    shearSlider.addEventListener('input', function () {
      var k = parseFloat(this.value);
      shearValue.textContent = k.toFixed(2);
      state.matrix = Matrix2x2.shearX(k);
      syncMatrixInputs();
      updateTelemetry();
      render();
    });

    // Mode buttons
    document.querySelectorAll('.mode-btn[data-mode]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.mode-btn').forEach(function (b) { b.classList.remove('active'); });
        this.classList.add('active');
        setMode(this.getAttribute('data-mode'));
      });
    });

    // HUD Zoom controls
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

    // Multiplication drawer slider
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
  }

  function setMode(newMode) {
    state.mode = newMode;
    multDrawer.classList.toggle('active', newMode === 'mult');
    render();
  }

  // ── Initialization ────────────────────────────────────────────────────────

  function init() {
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
    reset: function () {
      applyPreset('identity');
    }
  };

  init();

})();
