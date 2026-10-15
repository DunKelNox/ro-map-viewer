"""
Utilidades mínimas de matrices 4x4 (numpy, row-vectors column-major como OpenGL).
Evita traer PyGLM como dependencia extra; solo lo que necesitamos.
"""
import numpy as np
import math


def identity():
    return np.identity(4, dtype=np.float32)


def translate(m, v):
    t = np.identity(4, dtype=np.float32)
    t[0, 3] = v[0]
    t[1, 3] = v[1]
    t[2, 3] = v[2]
    return m @ t


def scale(m, v):
    s = np.identity(4, dtype=np.float32)
    s[0, 0] = v[0]
    s[1, 1] = v[1]
    s[2, 2] = v[2]
    return m @ s


def rotate(m, angle_rad, axis):
    axis = np.array(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return m
    axis = axis / n
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1 - c
    r = np.array([
        [t*x*x+c,   t*x*y-s*z, t*x*z+s*y, 0],
        [t*x*y+s*z, t*y*y+c,   t*y*z-s*x, 0],
        [t*x*z-s*y, t*y*z+s*x, t*z*z+c,   0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return m @ r


def mat3_to_mat4(m3_9):
    """
    9 floats del .rsm → mat4, orden columna como BrowEdit/GLM:
      primer trío = columna 0, etc.
    """
    a = np.identity(4, dtype=np.float32)
    a[0, 0], a[1, 0], a[2, 0] = m3_9[0], m3_9[1], m3_9[2]
    a[0, 1], a[1, 1], a[2, 1] = m3_9[3], m3_9[4], m3_9[5]
    a[0, 2], a[1, 2], a[2, 2] = m3_9[6], m3_9[7], m3_9[8]
    return a


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def to_gl(m):
    """numpy row-major 4x4 -> lista column-major (formato que espera glUniformMatrix4fv con transpose=False)."""
    return m.T.astype(np.float32).flatten()


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_slerp(q0, q1, t):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return quat_normalize(s0 * q0 + s1 * q1)


def quat_to_mat4(q):
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = 1.0 - 2.0 * (yy + zz)
    m[0, 1] = 2.0 * (xy - wz)
    m[0, 2] = 2.0 * (xz + wy)
    m[1, 0] = 2.0 * (xy + wz)
    m[1, 1] = 1.0 - 2.0 * (xx + zz)
    m[1, 2] = 2.0 * (yz - wx)
    m[2, 0] = 2.0 * (xz - wy)
    m[2, 1] = 2.0 * (yz + wx)
    m[2, 2] = 1.0 - 2.0 * (xx + yy)
    return m
