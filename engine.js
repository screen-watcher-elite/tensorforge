/**
 * TensorForge — Core Linear Algebra & Math Engine
 * Zero dependencies. Pure JavaScript vector & matrix mathematics.
 * Designed for FY Engineering Linear Algebra & embeddable into Lectern.
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.TensorForgeEngine = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var EPSILON = 1e-9;

  // ── 2D Vector Class ───────────────────────────────────────────────────────

  function Vector2D(x, y) {
    this.x = typeof x === 'number' && !isNaN(x) ? x : 0;
    this.y = typeof y === 'number' && !isNaN(y) ? y : 0;
  }

  Vector2D.prototype.clone = function () {
    return new Vector2D(this.x, this.y);
  };

  Vector2D.prototype.add = function (v) {
    return new Vector2D(this.x + v.x, this.y + v.y);
  };

  Vector2D.prototype.sub = function (v) {
    return new Vector2D(this.x - v.x, this.y - v.y);
  };

  Vector2D.prototype.scale = function (s) {
    return new Vector2D(this.x * s, this.y * s);
  };

  Vector2D.prototype.dot = function (v) {
    return this.x * v.x + this.y * v.y;
  };

  Vector2D.prototype.cross = function (v) {
    return this.x * v.y - this.y * v.x;
  };

  Vector2D.prototype.magnitude = function () {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  };

  Vector2D.prototype.magnitudeSq = function () {
    return this.x * this.x + this.y * this.y;
  };

  Vector2D.prototype.normalize = function () {
    var mag = this.magnitude();
    if (mag < EPSILON) return new Vector2D(0, 0);
    return new Vector2D(this.x / mag, this.y / mag);
  };

  Vector2D.prototype.angle = function () {
    return Math.atan2(this.y, this.x);
  };

  Vector2D.prototype.projectOnto = function (v) {
    var denom = v.magnitudeSq();
    if (denom < EPSILON) return new Vector2D(0, 0);
    var scalar = this.dot(v) / denom;
    return v.scale(scalar);
  };

  Vector2D.prototype.distTo = function (v) {
    var dx = this.x - v.x;
    var dy = this.y - v.y;
    return Math.sqrt(dx * dx + dy * dy);
  };

  Vector2D.lerp = function (a, b, t) {
    return new Vector2D(
      a.x + (b.x - a.x) * t,
      a.y + (b.y - a.y) * t
    );
  };

  // ── 2x2 Matrix Class ──────────────────────────────────────────────────────
  // Representation:
  // [ a  b ]   a: col 0, row 0   b: col 1, row 0
  // [ c  d ]   c: col 0, row 1   d: col 1, row 1
  //
  // Geometrically:
  // Column 1 [a, c]^T is the transformed basis vector i_hat
  // Column 2 [b, d]^T is the transformed basis vector j_hat

  function Matrix2x2(a, b, c, d) {
    this.a = typeof a === 'number' && !isNaN(a) ? a : 1;
    this.b = typeof b === 'number' && !isNaN(b) ? b : 0;
    this.c = typeof c === 'number' && !isNaN(c) ? c : 0;
    this.d = typeof d === 'number' && !isNaN(d) ? d : 1;
  }

  Matrix2x2.prototype.clone = function () {
    return new Matrix2x2(this.a, this.b, this.c, this.d);
  };

  Matrix2x2.fromColumns = function (col1, col2) {
    return new Matrix2x2(col1.x, col2.x, col1.y, col2.y);
  };

  Matrix2x2.identity = function () {
    return new Matrix2x2(1, 0, 0, 1);
  };

  Matrix2x2.rotation = function (rad) {
    var cos = Math.cos(rad);
    var sin = Math.sin(rad);
    return new Matrix2x2(cos, -sin, sin, cos);
  };

  Matrix2x2.shearX = function (k) {
    return new Matrix2x2(1, k, 0, 1);
  };

  Matrix2x2.shearY = function (k) {
    return new Matrix2x2(1, 0, k, 1);
  };

  Matrix2x2.scale = function (sx, sy) {
    return new Matrix2x2(sx, 0, 0, typeof sy === 'number' ? sy : sx);
  };

  Matrix2x2.reflection = function (axisAngleRad) {
    // Reflection across line at angle theta: [cos(2θ)  sin(2θ)]
    //                                       [sin(2θ) -cos(2θ)]
    var doubleAngle = 2 * axisAngleRad;
    return new Matrix2x2(Math.cos(doubleAngle), Math.sin(doubleAngle), Math.sin(doubleAngle), -Math.cos(doubleAngle));
  };

  Matrix2x2.prototype.getCol1 = function () {
    return new Vector2D(this.a, this.c);
  };

  Matrix2x2.prototype.getCol2 = function () {
    return new Vector2D(this.b, this.d);
  };

  Matrix2x2.prototype.apply = function (v) {
    return new Vector2D(
      this.a * v.x + this.b * v.y,
      this.c * v.x + this.d * v.y
    );
  };

  Matrix2x2.prototype.multiply = function (m) {
    // this * m
    return new Matrix2x2(
      this.a * m.a + this.b * m.c,
      this.a * m.b + this.b * m.d,
      this.c * m.a + this.d * m.c,
      this.c * m.b + this.d * m.d
    );
  };

  Matrix2x2.prototype.determinant = function () {
    return this.a * this.d - this.b * this.c;
  };

  Matrix2x2.prototype.trace = function () {
    return this.a + this.d;
  };

  Matrix2x2.prototype.rank = function () {
    var det = this.determinant();
    if (Math.abs(det) > EPSILON) return 2;
    // Det is 0: check if all entries are 0
    if (Math.abs(this.a) < EPSILON && Math.abs(this.b) < EPSILON &&
        Math.abs(this.c) < EPSILON && Math.abs(this.d) < EPSILON) {
      return 0;
    }
    return 1;
  };

  Matrix2x2.prototype.inverse = function () {
    var det = this.determinant();
    if (Math.abs(det) < EPSILON) return null; // Singular matrix
    var invDet = 1 / det;
    return new Matrix2x2(
      this.d * invDet,
      -this.b * invDet,
      -this.c * invDet,
      this.a * invDet
    );
  };

  Matrix2x2.prototype.transpose = function () {
    return new Matrix2x2(this.a, this.c, this.b, this.d);
  };

  Matrix2x2.prototype.power = function (k) {
    if (k <= 0) return Matrix2x2.identity();
    var res = this.clone();
    for (var i = 1; i < k; i++) {
      res = res.multiply(this);
    }
    return res;
  };

  // Fundamental Theorem: Kernel / Nullspace ker(A) = {x : Ax = 0}
  Matrix2x2.prototype.nullspace = function () {
    var det = this.determinant();
    if (Math.abs(det) > EPSILON) return null; // Trivial {0}

    var v = null;
    if (this.a * this.a + this.b * this.b > EPSILON) {
      v = new Vector2D(-this.b, this.a).normalize();
    } else if (this.c * this.c + this.d * this.d > EPSILON) {
      v = new Vector2D(-this.d, this.c).normalize();
    } else {
      // Entire 2D space is nullspace (Zero matrix)
      v = new Vector2D(1, 0);
    }
    return v;
  };

  // Column space (Range / Im(A))
  Matrix2x2.prototype.columnspace = function () {
    var col1 = this.getCol1();
    var col2 = this.getCol2();
    if (col1.magnitudeSq() > EPSILON) return col1.normalize();
    if (col2.magnitudeSq() > EPSILON) return col2.normalize();
    return new Vector2D(0, 0);
  };

  Matrix2x2.lerp = function (m1, m2, t) {
    return new Matrix2x2(
      m1.a + (m2.a - m1.a) * t,
      m1.b + (m2.b - m1.b) * t,
      m1.c + (m2.c - m1.c) * t,
      m1.d + (m2.d - m1.d) * t
    );
  };

  // ── Eigenvalues & Eigenvectors Solver ─────────────────────────────────────
  // Characteristic equation:
  // det(A - λI) = 0
  // (a - λ)(d - λ) - bc = 0
  // λ² - (a + d)λ + (ad - bc) = 0
  // λ² - tr(A)λ + det(A) = 0
  //
  // Discriminant Δ = tr(A)² - 4*det(A)

  function solveEigensystem(matrix) {
    var tr = matrix.trace();
    var det = matrix.determinant();
    var discriminant = tr * tr - 4 * det;

    var result = {
      trace: tr,
      determinant: det,
      discriminant: discriminant,
      isReal: discriminant >= -EPSILON,
      // LaTeX / educational step breakdown
      equationString: 'λ² - (' + tr.toFixed(2) + ')λ + (' + det.toFixed(2) + ') = 0',
      discriminantString: 'Δ = (' + tr.toFixed(2) + ')² - 4(' + det.toFixed(2) + ') = ' + discriminant.toFixed(2),
      eigenvalues: [],
      eigenvectors: []
    };

    if (result.isReal) {
      // Numerical stability cleanup for zero
      var disc = Math.max(0, discriminant);
      var sqrtDisc = Math.sqrt(disc);
      var lambda1 = (tr + sqrtDisc) / 2;
      var lambda2 = (tr - sqrtDisc) / 2;

      result.eigenvalues = [
        { value: lambda1, isReal: true },
        { value: lambda2, isReal: true }
      ];

      // Solve for eigenvectors: (A - λI)v = 0
      // [a - λ    b   ] [x] = [0]
      // [  c    d - λ ] [y]   [0]
      var v1 = findEigenvector(matrix, lambda1);
      var v2 = findEigenvector(matrix, lambda2);

      result.eigenvectors = [v1, v2];
    } else {
      // Complex conjugate pair: α ± βi
      var realPart = tr / 2;
      var imagPart = Math.sqrt(-discriminant) / 2;
      result.eigenvalues = [
        { real: realPart, imag: imagPart, isReal: false },
        { real: realPart, imag: -imagPart, isReal: false }
      ];
      result.eigenvectors = []; // No real invariant directions
    }

    return result;
  }

  function findEigenvector(matrix, lambda) {
    var a = matrix.a - lambda;
    var b = matrix.b;
    var c = matrix.c;
    var d = matrix.d - lambda;

    // We want a non-zero vector (x, y) such that:
    // a*x + b*y = 0  and  c*x + d*y = 0
    var x = 0, y = 0;

    if (Math.abs(b) > EPSILON) {
      // x = 1, y = -a / b
      x = 1;
      y = -a / b;
    } else if (Math.abs(c) > EPSILON) {
      // y = 1, x = -d / c
      y = 1;
      x = -d / c;
    } else if (Math.abs(a) > EPSILON) {
      // b == 0, so a*x = 0 => x = 0, y = 1
      x = 0;
      y = 1;
    } else if (Math.abs(d) > EPSILON) {
      // c == 0, so d*y = 0 => y = 0, x = 1
      x = 1;
      y = 0;
    } else {
      // A - λI is the zero matrix (e.g. Identity scaling) -> all vectors are eigenvectors
      x = 1;
      y = 0;
    }

    var vec = new Vector2D(x, y).normalize();
    return {
      vector: vec,
      lambda: lambda,
      spanAngle: Math.atan2(vec.y, vec.x)
    };
  }

  // ── SVD (Singular Value Decomposition) 2x2 Helper ────────────────────────
  // A^T * A is symmetric positive semi-definite
  // Singular values σ₁, σ₂ are the square roots of eigenvalues of A^T * A
  function computeSVD2x2(matrix) {
    var a = matrix.a, b = matrix.b, c = matrix.c, d = matrix.d;
    // M = A^T * A
    // [a  c] [a  b] = [a² + c²       ab + cd]
    // [b  d] [c  d]   [ab + cd       b² + d²]
    var m11 = a * a + c * c;
    var m12 = a * b + cd(a, b, c, d);
    var m22 = b * b + d * d;

    function cd(a, b, c, d) { return a * b + c * d; }

    var tr = m11 + m22;
    var det = m11 * m22 - m12 * m12;
    var disc = Math.max(0, tr * tr - 4 * det);
    var sqrtDisc = Math.sqrt(disc);

    var eig1 = Math.max(0, (tr + sqrtDisc) / 2);
    var eig2 = Math.max(0, (tr - sqrtDisc) / 2);

    return {
      sigma1: Math.sqrt(eig1),
      sigma2: Math.sqrt(eig2),
      conditionNumber: Math.sqrt(eig2) > EPSILON ? Math.sqrt(eig1) / Math.sqrt(eig2) : Infinity
    };
  }

  // ── Smooth Animation Controller ───────────────────────────────────────────

  function AnimationController(onUpdate, onComplete) {
    this.onUpdate = onUpdate;
    this.onComplete = onComplete;
    this.animId = null;
    this.startTime = 0;
    this.duration = 600; // ms
    this.fromMatrix = null;
    this.toMatrix = null;
    this.running = false;
  }

  AnimationController.prototype.start = function (fromM, toM, durationMs) {
    if (this.running) this.stop();
    this.fromMatrix = fromM.clone();
    this.toMatrix = toM.clone();
    this.duration = durationMs || 600;
    this.startTime = performance.now();
    this.running = true;

    var self = this;
    function loop(now) {
      var elapsed = now - self.startTime;
      var rawT = Math.min(1, elapsed / self.duration);
      // Cubic ease out
      var easeT = 1 - Math.pow(1 - rawT, 3);

      var current = Matrix2x2.lerp(self.fromMatrix, self.toMatrix, easeT);
      if (self.onUpdate) self.onUpdate(current, easeT);

      if (rawT < 1) {
        self.animId = requestAnimationFrame(loop);
      } else {
        self.running = false;
        if (self.onComplete) self.onComplete(self.toMatrix);
      }
    }

    this.animId = requestAnimationFrame(loop);
  };

  AnimationController.prototype.stop = function () {
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
    this.running = false;
  };

  // ── Public Export ─────────────────────────────────────────────────────────

  return {
    EPSILON: EPSILON,
    Vector2D: Vector2D,
    Matrix2x2: Matrix2x2,
    solveEigensystem: solveEigensystem,
    computeSVD2x2: computeSVD2x2,
    AnimationController: AnimationController
  };
});
