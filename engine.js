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

  // ── 3D Linear Algebra Math Core ──────────────────────────────────────────

  function Vector3D(x, y, z) {
    this.x = typeof x === 'number' && !isNaN(x) ? x : 0;
    this.y = typeof y === 'number' && !isNaN(y) ? y : 0;
    this.z = typeof z === 'number' && !isNaN(z) ? z : 0;
  }

  Vector3D.prototype.clone = function () {
    return new Vector3D(this.x, this.y, this.z);
  };

  Vector3D.prototype.add = function (v) {
    return new Vector3D(this.x + v.x, this.y + v.y, this.z + v.z);
  };

  Vector3D.prototype.sub = function (v) {
    return new Vector3D(this.x - v.x, this.y - v.y, this.z - v.z);
  };

  Vector3D.prototype.scale = function (s) {
    return new Vector3D(this.x * s, this.y * s, this.z * s);
  };

  Vector3D.prototype.dot = function (v) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  };

  Vector3D.prototype.cross = function (v) {
    return new Vector3D(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  };

  Vector3D.prototype.magnitude = function () {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  };

  Vector3D.prototype.normalize = function () {
    var mag = this.magnitude();
    if (mag < EPSILON) return new Vector3D(0, 0, 0);
    return new Vector3D(this.x / mag, this.y / mag, this.z / mag);
  };

  // 3x3 Matrix Class (Row-major)
  // [ m00, m01, m02 ]
  // [ m10, m11, m12 ]
  // [ m20, m21, m22 ]
  function Matrix3x3(arr) {
    if (Array.isArray(arr) && arr.length >= 9) {
      this.m = arr.slice(0, 9);
    } else {
      this.m = [
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
      ];
    }
  }

  Matrix3x3.prototype.clone = function () {
    return new Matrix3x3(this.m.slice());
  };

  Matrix3x3.identity = function () {
    return new Matrix3x3([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1
    ]);
  };

  Matrix3x3.prototype.apply = function (v) {
    var m = this.m;
    return new Vector3D(
      m[0] * v.x + m[1] * v.y + m[2] * v.z,
      m[3] * v.x + m[4] * v.y + m[5] * v.z,
      m[6] * v.x + m[7] * v.y + m[8] * v.z
    );
  };

  Matrix3x3.prototype.multiply = function (other) {
    var a = this.m;
    var b = other.m;
    var out = new Array(9);
    for (var r = 0; r < 3; r++) {
      for (var c = 0; c < 3; c++) {
        out[r * 3 + c] =
          a[r * 3 + 0] * b[0 * 3 + c] +
          a[r * 3 + 1] * b[1 * 3 + c] +
          a[r * 3 + 2] * b[2 * 3 + c];
      }
    }
    return new Matrix3x3(out);
  };

  Matrix3x3.prototype.determinant = function () {
    var m = this.m;
    return (
      m[0] * (m[4] * m[8] - m[5] * m[7]) -
      m[1] * (m[3] * m[8] - m[5] * m[6]) +
      m[2] * (m[3] * m[7] - m[4] * m[6])
    );
  };

  Matrix3x3.rotationX = function (rad) {
    var c = Math.cos(rad), s = Math.sin(rad);
    return new Matrix3x3([
      1, 0,  0,
      0, c, -s,
      0, s,  c
    ]);
  };

  Matrix3x3.rotationY = function (rad) {
    var c = Math.cos(rad), s = Math.sin(rad);
    return new Matrix3x3([
       c, 0, s,
       0, 1, 0,
      -s, 0, c
    ]);
  };

  Matrix3x3.rotationZ = function (rad) {
    var c = Math.cos(rad), s = Math.sin(rad);
    return new Matrix3x3([
      c, -s, 0,
      s,  c, 0,
      0,  0, 1
    ]);
  };

  // 3D Camera Projection (Isometric & Weak Perspective)
  function project3DTo2D(v3, cameraPitch, cameraYaw, fovScale) {
    var cosP = Math.cos(cameraPitch), sinP = Math.sin(cameraPitch);
    var cosY = Math.cos(cameraYaw), sinY = Math.sin(cameraYaw);

    // Yaw around Y
    var x1 = v3.x * cosY + v3.z * sinY;
    var y1 = v3.y;
    var z1 = -v3.x * sinY + v3.z * cosY;

    // Pitch around X
    var x2 = x1;
    var y2 = y1 * cosP - z1 * sinP;
    var z2 = y1 * sinP + z1 * cosP;

    // Perspective depth
    var distance = 5.0;
    var factor = fovScale / (distance + z2 * 0.4);

    return {
      x: x2 * factor,
      y: y2 * factor,
      depth: z2
    };
  }

  // ── LossLab Optimization Engine ───────────────────────────────────────────

  var LossFunctions = {
    bowl: {
      name: 'Convex Quadratic Bowl',
      evaluate: function (x, y) { return 0.5 * (x * x + 2 * y * y); },
      gradient: function (x, y) { return { dx: x, dy: 2 * y }; },
      domain: 3.0
    },
    saddle: {
      name: 'Saddle Point (Hyperbolic Paraboloid)',
      evaluate: function (x, y) { return 0.5 * (x * x - y * y); },
      gradient: function (x, y) { return { dx: x, dy: -y }; },
      domain: 2.5
    },
    rosenbrock: {
      name: "Rosenbrock's Banana Valley",
      evaluate: function (x, y) { return Math.pow(1 - x, 2) + 10 * Math.pow(y - x * x, 2); },
      gradient: function (x, y) {
        return {
          dx: -2 * (1 - x) - 40 * x * (y - x * x),
          dy: 20 * (y - x * x)
        };
      },
      domain: 2.0
    }
  };

  function OptimizerParticle(type, startX, startY, color) {
    this.type = type; // 'sgd' | 'momentum' | 'rmsprop' | 'adam'
    this.x = startX;
    this.y = startY;
    this.color = color;
    this.vx = 0;
    this.vy = 0;
    this.sx = 0; // 2nd moment
    this.sy = 0;
    this.stepCount = 0;
    this.history = [{ x: startX, y: startY }];
  }

  OptimizerParticle.prototype.step = function (lossFn, lr) {
    this.stepCount++;
    var grad = lossFn.gradient(this.x, this.y);
    var gx = grad.dx;
    var gy = grad.dy;

    // Gradient clipping
    var clip = 15;
    gx = Math.max(-clip, Math.min(clip, gx));
    gy = Math.max(-clip, Math.min(clip, gy));

    if (this.type === 'sgd') {
      this.x -= lr * gx;
      this.y -= lr * gy;
    } else if (this.type === 'momentum') {
      var beta = 0.85;
      this.vx = beta * this.vx + (1 - beta) * gx;
      this.vy = beta * this.vy + (1 - beta) * gy;
      this.x -= lr * this.vx;
      this.y -= lr * this.vy;
    } else if (this.type === 'rmsprop') {
      var gamma = 0.9;
      this.sx = gamma * this.sx + (1 - gamma) * (gx * gx);
      this.sy = gamma * this.sy + (1 - gamma) * (gy * gy);
      this.x -= (lr / (Math.sqrt(this.sx) + 1e-7)) * gx;
      this.y -= (lr / (Math.sqrt(this.sy) + 1e-7)) * gy;
    } else if (this.type === 'adam') {
      var b1 = 0.9, b2 = 0.999;
      this.vx = b1 * this.vx + (1 - b1) * gx;
      this.vy = b1 * this.vy + (1 - b1) * gy;
      this.sx = b2 * this.sx + (1 - b2) * (gx * gx);
      this.sy = b2 * this.sy + (1 - b2) * (gy * gy);

      // Bias correction
      var vHatX = this.vx / (1 - Math.pow(b1, this.stepCount));
      var vHatY = this.vy / (1 - Math.pow(b1, this.stepCount));
      var sHatX = this.sx / (1 - Math.pow(b2, this.stepCount));
      var sHatY = this.sy / (1 - Math.pow(b2, this.stepCount));

      this.x -= (lr / (Math.sqrt(sHatX) + 1e-7)) * vHatX;
      this.y -= (lr / (Math.sqrt(sHatY) + 1e-7)) * vHatY;
    }

    // Keep history (max 80 points)
    if (this.history.length > 80) this.history.shift();
    this.history.push({ x: this.x, y: this.y });
  };

  // ── MicroGraph: Pure JS Autograd DAG Engine ───────────────────────────────

  var _nodeId = 0;
  function AutogradValue(data, children, op, label) {
    this.id = ++_nodeId;
    this.data = typeof data === 'number' ? data : 0;
    this.grad = 0.0;
    this._backward = function () {};
    this._prev = children ? children.slice() : [];
    this._op = op || '';
    this.label = label || '';
  }

  AutogradValue.prototype.add = function (other, label) {
    other = other instanceof AutogradValue ? other : new AutogradValue(other);
    var out = new AutogradValue(this.data + other.data, [this, other], '+', label);
    out._backward = function () {
      this.grad += 1.0 * out.grad;
      other.grad += 1.0 * out.grad;
    }.bind(this);
    return out;
  };

  AutogradValue.prototype.mul = function (other, label) {
    other = other instanceof AutogradValue ? other : new AutogradValue(other);
    var out = new AutogradValue(this.data * other.data, [this, other], '×', label);
    out._backward = function () {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    }.bind(this);
    return out;
  };

  AutogradValue.prototype.relu = function (label) {
    var out = new AutogradValue(this.data > 0 ? this.data : 0, [this], 'ReLU', label);
    out._backward = function () {
      this.grad += (out.data > 0 ? 1.0 : 0.0) * out.grad;
    }.bind(this);
    return out;
  };

  AutogradValue.prototype.backward = function () {
    var topo = [];
    var visited = {};
    function buildTopo(v) {
      if (!visited[v.id]) {
        visited[v.id] = true;
        v._prev.forEach(buildTopo);
        topo.push(v);
      }
    }
    buildTopo(this);

    this.grad = 1.0;
    for (var i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
    return topo;
  };

  // ── Public Export ─────────────────────────────────────────────────────────

  return {
    EPSILON: EPSILON,
    Vector2D: Vector2D,
    Matrix2x2: Matrix2x2,
    Vector3D: Vector3D,
    Matrix3x3: Matrix3x3,
    project3DTo2D: project3DTo2D,
    LossFunctions: LossFunctions,
    OptimizerParticle: OptimizerParticle,
    AutogradValue: AutogradValue,
    solveEigensystem: solveEigensystem,
    computeSVD2x2: computeSVD2x2,
    AnimationController: AnimationController
  };
});

