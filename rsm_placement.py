# -*- coding: utf-8 -*-
"""
Colocación de modelos RSM (v1.x) idéntica a BrowEdit3.

Reproduce, sin OpenGL ni dependencias extra, lo que hacen en BrowEdit3
(Borf/BrowEdit3, browedit/components/Rsm.cpp y RsmRenderer.cpp):

  Rsm::updateMatrices()      -> bbox / bbrange / realbbmin / realbbmax / realbbrange
  Rsm::Mesh::calcMatrix1()   -> matrix1 (rama version < 0x202)
  Rsm::Mesh::calcMatrix2()   -> matrix2 (rama version < 0x202)
  RsmRenderer::render()      -> matriz de instancia del RSW

Fórmula completa (RSM 1.x):

  vertice_mundo =
      S(1,1,-1)
    @ T(5*gnd_w + pos.x, -pos.y, -10 - 5*gnd_h + pos.z)
    @ Rz(-rot.z) @ Rx(-rot.x) @ Ry(rot.y)
    @ S(scale.x, -scale.y, scale.z)
    @ T(-realbbrange.x, realbbmin.y, -realbbrange.z)      # <- centrado por bbox REAL
    @ matriz_malla                                         # matrix1 de ancestros @ matrix1 @ matrix2
    @ vertice_local

Lo importante: el modelo NO se planta por el origen del nodo raíz, sino por el
centro (XZ) de su bounding box real (con todos los hijos ya transformados, en
t=0) y apoyando su punto más bajo en y=0. Cualquier traslación global de la
raíz se cancela; solo importan rotaciones, escalas, offsets y la posición
relativa de cada hijo.
"""
import math
import numpy as np

import mat4


# ---------------------------------------------------------------------------
# Helpers de animación / nombres (antes en rsm_viewer_3d; aquí para evitar
# import circular: rsm_viewer → rsm_placement → rsm_viewer).
# ---------------------------------------------------------------------------

def _sample_rotation_quat(frames, anim_time):
    """Slerp de cuaternión entre keyframes. frames ordenados por time."""
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]['quat']
    t0 = frames[0]['time']
    t_last = frames[-1]['time']
    if anim_time <= t0:
        return frames[0]['quat']
    if anim_time >= t_last:
        return frames[-1]['quat']
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        if a['time'] <= anim_time <= b['time']:
            span = b['time'] - a['time']
            if span <= 0:
                return a['quat']
            u = (anim_time - a['time']) / float(span)
            return mat4.quat_slerp(a['quat'], b['quat'], u)
    return frames[-1]['quat']


def _sample_scale(scale_keyframes, static_scale, anim_time):
    """Lerp de escala entre keyframes; si no hay, escala estática."""
    if not scale_keyframes:
        return static_scale
    if len(scale_keyframes) == 1:
        return scale_keyframes[0]['scale']
    if anim_time <= scale_keyframes[0]['frame']:
        return scale_keyframes[0]['scale']
    if anim_time >= scale_keyframes[-1]['frame']:
        return scale_keyframes[-1]['scale']
    for i in range(len(scale_keyframes) - 1):
        a, b = scale_keyframes[i], scale_keyframes[i + 1]
        if a['frame'] <= anim_time <= b['frame']:
            span = b['frame'] - a['frame']
            if span <= 0:
                return a['scale']
            u = (anim_time - a['frame']) / float(span)
            sa, sb = a['scale'], b['scale']
            return (
                sa[0] + (sb[0] - sa[0]) * u,
                sa[1] + (sb[1] - sa[1]) * u,
                sa[2] + (sb[2] - sa[2]) * u,
            )
    return scale_keyframes[-1]['scale']


def _mesh_name_key(s):
    """Normaliza nombres de nodo RSM para emparejar padre/hijo."""
    if not s:
        return ''
    return s.strip().rstrip('\x00').lower()



# ----------------------------------------------------------------------------
# Árbol de mallas
# ----------------------------------------------------------------------------

def _build_tree(rsm):
    """Devuelve (roots, children_of). Empareja por nombre normalizado."""
    by_parent = {}
    for m in rsm.meshes:
        key = _mesh_name_key(m.get('parent_name') or '')
        by_parent.setdefault(key, []).append(m)

    def children_of(mesh):
        return by_parent.get(_mesh_name_key(mesh['name']), [])

    root_name = _mesh_name_key(getattr(rsm, 'root_node_name', '') or '')
    roots = []
    for m in rsm.meshes:
        pn = _mesh_name_key(m.get('parent_name') or '')
        if pn == '' or _mesh_name_key(m['name']) == root_name:
            if m not in roots:
                roots.append(m)
    if not roots and rsm.meshes:
        roots = [rsm.meshes[0]]
    return roots, children_of


def _offset_mat(mesh):
    return mat4.mat3_to_mat4(mesh['offset_matrix']).astype(np.float64)


def _t(v):
    m = np.identity(4, dtype=np.float64)
    m[0, 3], m[1, 3], m[2, 3] = float(v[0]), float(v[1]), float(v[2])
    return m


def _s(v):
    m = np.identity(4, dtype=np.float64)
    m[0, 0], m[1, 1], m[2, 2] = float(v[0]), float(v[1]), float(v[2])
    return m


def _rot_axis(angle, axis):
    return mat4.rotate(np.identity(4, dtype=np.float64), angle, axis).astype(np.float64)


# ----------------------------------------------------------------------------
# Rsm::Mesh::setBoundingBox  (bbox "de diseño", usada por matrix1 de la raíz)
# ----------------------------------------------------------------------------

def _model_bbox(roots, children_of):
    bbmin = np.array([9999999.0] * 3)
    bbmax = np.array([-9999999.0] * 3)

    def visit(mesh):
        nonlocal bbmin, bbmax
        has_parent = _mesh_name_key(mesh.get('parent_name') or '') != ''
        kids = children_of(mesh)
        off = _offset_mat(mesh)
        add = np.array(mesh['pos'], dtype=np.float64) + np.array(mesh['pos_'], dtype=np.float64)
        use_add = has_parent or len(kids) > 0
        # BrowEdit solo mira vértices usados por caras; usamos todos los de la
        # malla salvo que la cara los referencie (equivalente en modelos sanos).
        used = set()
        for f in mesh['faces']:
            for vi in f['vertex_ids']:
                used.add(vi)
        verts = mesh['vertices']
        for i in (used if used else range(len(verts))):
            v = verts[i]
            w = (off @ np.array([v[0], v[1], v[2], 1.0]))[:3]
            if use_add:
                w = w + add
            bbmin = np.minimum(bbmin, w)
            bbmax = np.maximum(bbmax, w)
        for c in kids:
            visit(c)

    for r in roots:
        visit(r)
    if np.any(bbmin > bbmax):
        bbmin = np.zeros(3)
        bbmax = np.zeros(3)
    return bbmin, bbmax


# ----------------------------------------------------------------------------
# calcMatrix1 / calcMatrix2 (version < 0x202)
# ----------------------------------------------------------------------------

def _prepare(rsm):
    """Árbol + bbox de diseño; se calcula una sola vez por modelo (cache en el objeto)."""
    cached = getattr(rsm, '_rp_cache', None)
    if cached is None:
        roots, children_of = _build_tree(rsm)
        bbmin, bbmax = _model_bbox(roots, children_of)
        cached = (roots, children_of, bbmin, bbmax, (bbmin + bbmax) / 2.0)
        rsm._rp_cache = cached
    return cached


def compute_mesh_matrices(rsm, anim_time=0.0):
    if getattr(rsm, 'is_rsm2', False):
        return compute_mesh_matrices_rsm2(rsm, anim_time)
    return _compute_mesh_matrices_v1(rsm, anim_time)


def _compute_mesh_matrices_v1(rsm, anim_time=0.0):
    """
    Matriz de cada malla en "espacio RSM" (Y hacia abajo, SIN flips):
        M_malla = matrix1(ancestros) @ matrix1 @ matrix2

    Returns: dict nombre -> np.float64 4x4
    """
    roots, children_of, bbmin, bbmax, bbrange = _prepare(rsm)

    results = {}

    def visit(mesh, parent_sub, is_root):
        kids = children_of(mesh)
        has_children = len(kids) > 0

        m1 = np.identity(4, dtype=np.float64)
        if is_root:
            if has_children:
                m1 = m1 @ _t((-bbrange[0], -bbmax[1], -bbrange[2]))
            else:
                m1 = m1 @ _t((0.0, -bbmax[1] + bbrange[1], 0.0))
        else:
            m1 = m1 @ _t(mesh['pos'])

        frames = mesh.get('frames')
        if frames:
            q = _sample_rotation_quat(frames, anim_time)
            if q is not None:
                m1 = m1 @ mat4.quat_to_mat4(q).astype(np.float64)
        elif abs(mesh.get('rot_angle', 0) or 0) > 0.01:
            m1 = m1 @ _rot_axis(mesh['rot_angle'], mesh['rot_axis'])

        sc = _sample_scale(mesh.get('scale_keyframes') or [],
                           mesh.get('scale', (1, 1, 1)), anim_time)
        m1 = m1 @ _s(sc)

        # calcMatrix2: SIEMPRE translate(pos_) (las dos ramas de BrowEdit lo hacen)
        m2 = _t(mesh['pos_']) @ _offset_mat(mesh)

        sub = parent_sub @ m1
        results[mesh['name']] = sub @ m2
        for c in kids:
            visit(c, sub, False)

    for r in roots:
        visit(r, np.identity(4, dtype=np.float64), True)
    for m in rsm.meshes:           # huérfanas
        if m['name'] not in results:
            visit(m, np.identity(4, dtype=np.float64), True)
    return results


def compute_real_bbox(rsm, mesh_matrices):
    """
    Rsm::updateMatrices -> setBoundingBox2: todos los vértices, con matriz
    S(1,-1,1) @ M_malla (t=0). Devuelve (realbbmin, realbbmax) en Y-arriba.
    """
    flip = _s((1.0, -1.0, 1.0))
    mn = np.array([999999.0] * 3)
    mx = np.array([-999999.0] * 3)
    for mesh in rsm.meshes:
        M = mesh_matrices.get(mesh['name'])
        verts = mesh['vertices']
        if M is None or not len(verts):
            continue
        V = np.asarray(verts, dtype=np.float64)
        H = np.hstack([V, np.ones((len(V), 1))])
        W = (flip @ M @ H.T).T[:, :3]
        mn = np.minimum(mn, W.min(axis=0))
        mx = np.maximum(mx, W.max(axis=0))
    if np.any(mn > mx):
        mn = np.zeros(3)
        mx = np.zeros(3)
    return mn, mx


def model_placement_data(rsm):
    """
    Datos estáticos de un modelo (calcular UNA vez, en t=0):
      mesh_matrices_t0, realbbmin, realbbmax, realbbrange
    """
    mm0 = compute_mesh_matrices(rsm, 0.0)
    if getattr(rsm, 'is_rsm2', False):
        rbmin, rbmax = _bbox_rsm2(rsm, mm0)
    else:
        rbmin, rbmax = compute_real_bbox(rsm, mm0)
    return mm0, rbmin, rbmax, (rbmin + rbmax) / 2.0


def display_matrices(rsm, anim_time=0.0, rb_min=None, rb_range=None):
    """Matrices de malla listas para dibujar el modelo en el origen.

    Misma convención que el mapa (BrowEdit3), sin traslación RSW de terreno:

      S(1,1,-1) @ S(1,-1,1) @ T(-realbbrange.x, realbbmin.y, -realbbrange.z) @ M_malla

    El bbox (rb_min / rb_range) se calcula una vez en t=0 y se reutiliza
    durante la animación para que el modelo no levite.
    """
    is_rsm2 = bool(getattr(rsm, 'is_rsm2', False))
    if rb_min is None or rb_range is None:
        if is_rsm2:
            mm0 = compute_mesh_matrices_rsm2(rsm, 0.0)
            rb_min, rb_max = _bbox_rsm2(rsm, mm0)
        else:
            mm0 = compute_mesh_matrices(rsm, 0.0)
            rb_min, rb_max = compute_real_bbox(rsm, mm0)
        rb_range = (rb_min + rb_max) / 2.0
    if is_rsm2:
        # RSM2: anim_time en frames
        mm = compute_mesh_matrices_rsm2(rsm, float(anim_time))
    else:
        mm = compute_mesh_matrices(rsm, float(anim_time))
    pre = _s((1.0, 1.0, -1.0)) @ _s((1.0, -1.0, 1.0)) @ _t(
        (-float(rb_range[0]), float(rb_min[1]), -float(rb_range[2]))
    )
    out = {name: pre @ M for name, M in mm.items()}
    return out, rb_min, rb_range


# ----------------------------------------------------------------------------
# Matriz de instancia del RSW  (RsmRenderer.cpp)
# ----------------------------------------------------------------------------

def instance_matrix(gnd_w, gnd_h, pos, rot, scale, realbbmin, realbbrange):
    """Matriz mundo completa de una instancia (sin la matriz de malla)."""
    m = np.identity(4, dtype=np.float64)
    m = m @ _s((1.0, 1.0, -1.0))
    m = m @ _t((5.0 * gnd_w + pos[0], -pos[1], -10.0 - 5.0 * gnd_h + pos[2]))
    m = m @ _rot_axis(-math.radians(rot[2]), (0, 0, 1))
    m = m @ _rot_axis(-math.radians(rot[0]), (1, 0, 0))
    m = m @ _rot_axis(math.radians(rot[1]), (0, 1, 0))
    m = m @ _s((scale[0], -scale[1], scale[2]))
    m = m @ _t((-realbbrange[0], realbbmin[1], -realbbrange[2]))
    return m


def instance_origin(gnd_w, gnd_h, pos):
    """Punto de anclaje de la instancia en el mundo (para culling/distancia)."""
    return (5.0 * gnd_w + pos[0], -pos[1], 10.0 + 5.0 * gnd_h - pos[2])


# ============================================================================
# RSM 2.x  (v2.2 / v2.3) — Rsm.cpp (version >= 0x202) + RsmRenderer.cpp
# ============================================================================
#
# Diferencias con RSM1 (según BrowEdit3):
#   - El modelo NO se centra por bbox: se usa el origen del archivo tal cual.
#   - Matriz de instancia:
#       S(1,1,-1) T(5gw+px, -py, -10-5gh+pz) Rz(-rz) Rx(-rx) Ry(ry)
#         S(sx,-sy,sz) S(1,-1,1)  @  matrix2(malla)  @ v
#     (S(sx,-sy,sz)·S(1,-1,1) = S(sx,sy,sz): el modelo ya viene "Y arriba").
#   - matrix2(malla) = matrix2(padre) @ matrix1(malla), con
#       matrix1 = T(posición) @ [R(quat) | inv(offset_padre)@offset] @ S(escala)
#     Sin animación equivale a T(pos_) @ offset.
#   - Las 9 floats de offset se leen por columnas (igual que en RSM1).

def _unify_textures_rsm2(raw):
    """v2.2: lista global + índices por malla. v2.3: nombres por malla → lista global."""
    global_tex = list(raw.textures) if getattr(raw, 'textures', None) else []
    tex_to_idx = {t: i for i, t in enumerate(global_tex)}
    out = []
    for m in raw.meshes:
        local_names = m.get('textures') or []
        if local_names:                       # v2.3
            idxs = []
            for name in local_names:
                if name not in tex_to_idx:
                    tex_to_idx[name] = len(global_tex)
                    global_tex.append(name)
                idxs.append(tex_to_idx[name])
        else:                                 # v2.2
            idxs = list(m.get('texture_indices') or [0])
        out.append(idxs)
    return global_tex, out


def load_rsm2(path):
    """Carga un .rsm2 y lo devuelve con el mismo "shape" que RSMParser (+ is_rsm2)."""
    from ro_map_viewer import RSM2Parser, RSMFileError
    raw = RSM2Parser(path)
    if raw.error:
        raise RSMFileError(raw.error)
    if not raw.meshes:
        raise RSMFileError("El .rsm2 no contiene ninguna malla")
    global_tex, per_mesh_idx = _unify_textures_rsm2(raw)

    class _Rsm2:
        pass
    r = _Rsm2()
    r.is_rsm2 = True
    r.error = None
    r.version = raw.version
    r.fps = float(getattr(raw, 'fps', 30.0) or 30.0)
    r.anim_len = float(raw.anim_len or 0)          # en frames
    r.textures = global_tex
    r.root_node_name = (raw.main_mesh_names[0] if getattr(raw, 'main_mesh_names', None) else '')
    meshes = []
    for m, idxs in zip(raw.meshes, per_mesh_idx):
        meshes.append({
            'name': m['name'],
            'parent_name': m.get('parent_name') or '',
            'offset_matrix': tuple(m.get('matrix') or (1, 0, 0, 0, 1, 0, 0, 0, 1)),
            'pos': (0.0, 0.0, 0.0),
            'pos_': tuple(m.get('position_') or (0.0, 0.0, 0.0)),
            'rot_angle': 0.0, 'rot_axis': (0.0, 1.0, 0.0), 'scale': (1.0, 1.0, 1.0),
            'vertices': m.get('vertices') or [],
            'texcoords': m.get('texcoords') or [],
            'faces': m.get('faces') or [],
            'texture_indices': idxs,
            'frames': [{'time': float(k['frame']), 'quat': k['quat']}
                       for k in (m.get('rot_keyframes') or [])],
            'scale_keyframes': [{'frame': float(k['frame']), 'scale': tuple(k['scale'])}
                                for k in (m.get('scale_keyframes') or [])],
            'pos_keyframes': [{'frame': float(k['frame']), 'pos': tuple(k['pos'])}
                              for k in (m.get('pos_keyframes') or [])],
            'texture_keyframe_groups': m.get('texture_keyframe_groups') or [],
        })
    r.meshes = meshes
    return r


def _sample_keys(keys, tick, anim_len, default, get_val, lerp):
    """Muestreo estilo BrowEdit: antes de la 1ª clave interpola desde `default`,
    después de la última mantiene el último valor."""
    if not keys:
        return None
    n = len(keys)
    nxt = 0
    while nxt < n and tick >= keys[nxt][0]:
        nxt += 1
    prv = nxt - 1
    prev_tick = 0.0 if prv < 0 else keys[prv][0]
    next_tick = anim_len if nxt == n else keys[nxt][0]
    prev_v = default if prv < 0 else get_val(keys[prv])
    next_v = get_val(keys[n - 1]) if nxt == n else get_val(keys[nxt])
    span = next_tick - prev_tick
    u = 0.0 if abs(span) < 1e-9 else (tick - prev_tick) / span
    return lerp(prev_v, next_v, u)


def compute_mesh_matrices_rsm2(rsm, frame=0.0):
    """matrix2 de cada malla (RSM 2.x) en el frame indicado (0 = pose estática)."""
    by_name = {_mesh_name_key(m['name']): m for m in rsm.meshes}
    anim_len = max(1.0, float(rsm.anim_len or 1.0))
    tick = float(frame) % anim_len
    offs = {m['name']: _offset_mat(m) for m in rsm.meshes}
    inv_offs = {}
    for k, o in offs.items():
        try:
            inv_offs[k] = np.linalg.inv(o)
        except np.linalg.LinAlgError:
            inv_offs[k] = np.identity(4)

    # orden: padres antes que hijos
    order, seen = [], set()
    def visit(m):
        if m['name'] in seen:
            return
        p = by_name.get(_mesh_name_key(m.get('parent_name') or ''))
        if p is not None and p is not m:
            visit(p)
        seen.add(m['name'])
        order.append(m)
    for m in rsm.meshes:
        visit(m)

    m2 = {}
    for mesh in order:
        parent = by_name.get(_mesh_name_key(mesh.get('parent_name') or ''))
        if parent is mesh:
            parent = None
        name = mesh['name']
        pos_ = np.array(mesh['pos_'], dtype=np.float64)

        # rotación
        rot_keys = [(float(k['time']), np.asarray(k['quat'], dtype=np.float64))
                    for k in (mesh.get('frames') or [])]
        q = _sample_keys(rot_keys, tick, anim_len, np.array([0.0, 0.0, 0.0, 1.0]),
                         lambda k: k[1], lambda a, b, u: mat4.quat_slerp(a, b, u))
        # (_sample_keys recibe tuplas (t, valor); get_val=k[1])
        if q is not None:
            m1 = mat4.quat_to_mat4(q).astype(np.float64)
        else:
            m1 = offs[name].copy()
            if parent is not None:
                m1 = inv_offs[parent['name']] @ m1

        # escala animada
        sc_keys = [(float(k['frame']), np.asarray(k['scale'], dtype=np.float64))
                   for k in (mesh.get('scale_keyframes') or [])]
        sc = _sample_keys(sc_keys, tick, anim_len, np.ones(3),
                          lambda k: k[1], lambda a, b, u: a + (b - a) * u)
        if sc is not None:
            m1 = m1 @ _s(sc)

        # posición
        pk = [(float(k['frame']), np.asarray(k['pos'], dtype=np.float64))
              for k in (mesh.get('pos_keyframes') or [])]
        pos = _sample_keys(pk, tick, anim_len, pos_,
                           lambda k: k[1], lambda a, b, u: a + (b - a) * u)
        if pos is None:
            if parent is not None:
                d = pos_ - np.array(parent['pos_'], dtype=np.float64)
                pos = (inv_offs[parent['name']] @ np.array([d[0], d[1], d[2], 0.0]))[:3]
            else:
                pos = pos_
        m1 = _t(pos) @ m1
        m2[name] = (m2[parent['name']] @ m1) if parent is not None else m1
    return m2


def _bbox_rsm2(rsm, mesh_matrices):
    """BBox (Y arriba, sin flips) de todos los vértices, solo para culling/diagnóstico."""
    mn = np.array([999999.0] * 3)
    mx = np.array([-999999.0] * 3)
    for mesh in rsm.meshes:
        V = mesh['vertices']
        M = mesh_matrices.get(mesh['name'])
        if M is None or not len(V):
            continue
        H = np.hstack([np.asarray(V, dtype=np.float64), np.ones((len(V), 1))])
        W = (M @ H.T).T[:, :3]
        mn = np.minimum(mn, W.min(axis=0))
        mx = np.maximum(mx, W.max(axis=0))
    if np.any(mn > mx):
        mn = np.zeros(3); mx = np.zeros(3)
    return mn, mx


def instance_matrix_rsm2(gnd_w, gnd_h, pos, rot, scale):
    """Matriz de instancia RSW para modelos RSM 2.x (sin centrado por bbox)."""
    m = np.identity(4, dtype=np.float64)
    m = m @ _s((1.0, 1.0, -1.0))
    m = m @ _t((5.0 * gnd_w + pos[0], -pos[1], -10.0 - 5.0 * gnd_h + pos[2]))
    m = m @ _rot_axis(-math.radians(rot[2]), (0, 0, 1))
    m = m @ _rot_axis(-math.radians(rot[0]), (1, 0, 0))
    m = m @ _rot_axis(math.radians(rot[1]), (0, 1, 0))
    m = m @ _s((scale[0], -scale[1], scale[2]))
    m = m @ _s((1.0, -1.0, 1.0))
    return m


def texture_offsets(anims, frame, anim_len):
    """Animación de textura RSM 2.3 (Rsm::Mesh::getTextureAnimation).

    anims: lista de {'type', 'frames': [{'frame', 'offset'}]}
    Devuelve (du, dv, unsupported): desplazamiento de UV (tipos 0 y 1) y True si
    el modelo usa escala/rotación de textura (tipos 2-4), que no se aplican.
    """
    tick = float(frame) % max(1.0, float(anim_len))
    du = dv = 0.0
    unsupported = False
    for anim in (anims or []):
        fr = anim.get('frames') or []
        if not fr:
            continue
        n = len(fr)
        nxt = 0
        while nxt < n and tick >= fr[nxt]['frame']:
            nxt += 1
        prv = nxt - 1
        prev_t = 0.0 if prv < 0 else float(fr[prv]['frame'])
        next_t = float(anim_len) if nxt == n else float(fr[nxt]['frame'])
        prev_v = 0.0 if prv < 0 else float(fr[prv]['offset'])
        next_v = float(fr[n - 1]['offset']) if nxt == n else float(fr[nxt]['offset'])
        span = next_t - prev_t
        u = 0.0 if abs(span) < 1e-9 else (tick - prev_t) / span
        off = prev_v + (next_v - prev_v) * u
        t = anim.get('type', 1)
        if t == 0:
            du += off
        elif t == 1:
            dv += off
        else:
            unsupported = True
    return du, dv, unsupported
