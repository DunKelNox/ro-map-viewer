"""
Visor 3D de modelos .rsm de Ragnarok Online, con OpenGL real.

Carga geometría y texturas desde un archivo .rsm (vía RSMParser, ya
validado byte a byte contra un archivo real) y las renderiza en 3D,
usando rsm_placement (misma matemática BrowEdit3 que el visor de mapa):
matrices de malla + centrado por bbox real. RSM1 y RSM2.

Controles:
  - Arrastrar con click izquierdo: orbitar la cámara
  - Rueda del mouse: zoom
  - ESC: salir
"""
import os
from rolog import info as log_info, warn as log_warn
import sys
import math
import ctypes
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mat4
from ro_map_viewer import RSMParser, RSM2Parser, RSMFileError
import rsm_placement as RP
_sample_rotation_quat = RP._sample_rotation_quat
_sample_scale = RP._sample_scale
_mesh_name_key = RP._mesh_name_key

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform vec2 u_uv_offset;

out vec2 v_uv;
out vec3 v_normal;

void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_uv = in_uv + u_uv_offset;
    v_normal = mat3(u_model) * in_normal;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
in vec3 v_normal;

uniform sampler2D u_texture;
uniform vec3 u_light_dir;
uniform vec3 u_diffuse_color;
uniform vec3 u_ambient_color;
uniform float u_intensity;
uniform bool u_use_texture;

out vec4 frag_color;

void main() {
    vec4 tex_color = u_use_texture ? texture(u_texture, v_uv) : vec4(1.0);
    if (tex_color.a < 0.1)
        discard;

    // Las texturas de RO (BMP sin canal alfa, como los árboles) usan
    // magenta puro (255,0,255) como color-key de transparencia, en vez
    // de un canal alfa real. Lo descartamos igual que hace el cliente.
    if (tex_color.a < 0.1)
        discard;
    // Color-key magenta RO (umbral un poco amplio: packs re-exportados)
    if (u_use_texture && tex_color.r > 0.80 && tex_color.g < 0.25 && tex_color.b > 0.80)
        discard;

    vec3 n = normalize(v_normal);
    float ndotl = max(dot(n, -u_light_dir), 0.0);
    vec3 lit = u_ambient_color + u_diffuse_color * ndotl * u_intensity;
    lit = clamp(lit, 0.0, 1.5);

    frag_color = vec4(tex_color.rgb * lit, tex_color.a);
}
"""








def diagnose_rsm_hierarchy(rsm):
    """
    Diagnóstico de jerarquía / nombres de un RSM ya parseado.
    Devuelve un dict con señales útiles para afinar el parser.
    """
    meshes = list(getattr(rsm, 'meshes', None) or [])
    root_hdr = getattr(rsm, 'root_node_name', '') or ''
    root_hdr_key = _mesh_name_key(root_hdr)

    name_keys = {}
    for m in meshes:
        n = m.get('name') or ''
        k = _mesh_name_key(n)
        name_keys.setdefault(k, []).append(n)

    orphans = []
    no_parent = []
    encoding_suspect = []
    for m in meshes:
        n = m.get('name') or ''
        pn = m.get('parent_name') or ''
        # U+FFFD = decode con errors='replace'
        if any(ord(ch) == 0xFFFD for ch in (n + pn)):
            encoding_suspect.append(n or ('parent:' + pn))
        pnk = _mesh_name_key(pn)
        if pnk == '':
            no_parent.append(n)
        elif pnk not in name_keys:
            orphans.append({'mesh': n, 'parent_name': pn})

    if root_hdr_key == '':
        root_hdr_status = 'empty'
        root_hdr_ok = False
    elif root_hdr_key in name_keys:
        root_hdr_status = 'ok'
        root_hdr_ok = True
    else:
        root_hdr_status = 'missing'
        root_hdr_ok = False

    roots = []
    for m in meshes:
        pn = _mesh_name_key(m.get('parent_name') or '')
        if pn == '' or _mesh_name_key(m.get('name') or '') == root_hdr_key:
            if m not in roots:
                roots.append(m)
    if not roots and meshes:
        roots = [meshes[0]]

    by_parent = {}
    for m in meshes:
        key = _mesh_name_key(m.get('parent_name') or '')
        by_parent.setdefault(key, []).append(m)
    reachable = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        k = _mesh_name_key(cur.get('name') or '')
        if k in reachable:
            continue
        reachable.add(k)
        for ch in by_parent.get(k, []):
            stack.append(ch)
    disconnected = []
    for m in meshes:
        k = _mesh_name_key(m.get('name') or '')
        if k not in reachable:
            disconnected.append(m.get('name') or '?')

    dup_keys = {k: v for k, v in name_keys.items() if k and len(v) > 1}

    issues = []
    if root_hdr_status == 'missing':
        issues.append('root_hdr_missing')
    if root_hdr_status == 'empty' and len(meshes) > 1:
        issues.append('root_hdr_empty')
    if len(roots) > 1:
        issues.append('multi_root')
    if orphans:
        issues.append('orphan_parents')
    if disconnected:
        issues.append('disconnected')
    if encoding_suspect:
        issues.append('encoding')
    if dup_keys:
        issues.append('dup_names')

    return {
        'n_meshes': len(meshes),
        'root_hdr': root_hdr,
        'root_hdr_status': root_hdr_status,
        'root_hdr_ok': root_hdr_ok,
        'n_roots': len(roots),
        'root_names': [r.get('name') or '?' for r in roots],
        'n_no_parent': len(no_parent),
        'orphans': orphans,
        'disconnected': disconnected,
        'encoding_suspect': encoding_suspect,
        'dup_keys': {k: v for k, v in list(dup_keys.items())[:8]},
        'issues': issues,
        'has_issues': len(issues) > 0,
    }


def format_rsm_diagnosis(diag, file_label=''):
    """Una o varias líneas de log legibles."""
    tag = file_label or '?'
    base = (
        f"[RSM-diag] {tag}  meshes={diag['n_meshes']}  "
        f"root_hdr={diag['root_hdr_status']!r}({diag['root_hdr']!r})  "
        f"roots={diag['n_roots']}{diag['root_names'][:3]}  "
        f"orphans={len(diag['orphans'])}  disc={len(diag['disconnected'])}  "
        f"enc={len(diag['encoding_suspect'])}"
    )
    if not diag['has_issues']:
        return base + "  OK"
    lines = [base + f"  ISSUES={diag['issues']}"]
    if diag['orphans'][:5]:
        sample = ', '.join(
            f"{o['mesh']!r}->parent={o['parent_name']!r}" for o in diag['orphans'][:5]
        )
        lines.append(f"         orphans: {sample}")
    if diag['disconnected'][:5]:
        lines.append(f"         disconnected: {diag['disconnected'][:5]}")
    if diag['encoding_suspect'][:5]:
        lines.append(f"         encoding: {diag['encoding_suspect'][:5]}")
    if diag['dup_keys']:
        lines.append(f"         dup_names: {list(diag['dup_keys'].items())[:3]}")
    return '\n'.join(lines)


def compute_model_transforms(rsm, anim_time=0.0, fixed_anchor=None, for_rsw_instance=False,
                            elong_horiz=1.55, elong_mean=1.70, elong_mirror_x=False,
                            elong_mirror_z=False, force_mirror_x=False, force_mirror_z=False):
    """
    BrowEdit calcMatrix1/2 + keyframes (RSM1).

    Sin hacks: offset columna, jerarquía fiel, espejo Y global.
    Soporta varias raíces y empareja hijos por nombre normalizado.
    """
    # Índice padre → hijos (nombre normalizado)
    by_parent = {}
    for m in rsm.meshes:
        key = _mesh_name_key(m.get('parent_name') or '')
        by_parent.setdefault(key, []).append(m)

    def children_of(mesh):
        return by_parent.get(_mesh_name_key(mesh['name']), [])

    # Raíces: sin padre o root_node_name
    root_name = _mesh_name_key(getattr(rsm, 'root_node_name', '') or '')
    roots = []
    for m in rsm.meshes:
        pn = _mesh_name_key(m.get('parent_name') or '')
        if pn == '' or _mesh_name_key(m['name']) == root_name:
            if m not in roots:
                roots.append(m)
    if not roots and rsm.meshes:
        roots = [rsm.meshes[0]]

    overall_min = np.array([9999999.0, 9999999.0, 9999999.0])
    overall_max = np.array([-9999999.0, -9999999.0, -9999999.0])

    def visit_bbox(mesh):
        offset_m4 = mat4.mat3_to_mat4(mesh['offset_matrix']).astype(np.float64)
        has_parent = _mesh_name_key(mesh.get('parent_name') or '') != ''
        kids = children_of(mesh)
        add_pos = has_parent or (len(kids) > 0)
        pos_sum = np.array(mesh['pos']) + np.array(mesh['pos_']) if add_pos else np.zeros(3)
        for v in mesh['vertices']:
            vh = np.array([v[0], v[1], v[2], 1.0])
            world_v = (offset_m4 @ vh)[:3] + pos_sum
            nonlocal overall_min, overall_max
            overall_min = np.minimum(overall_min, world_v)
            overall_max = np.maximum(overall_max, world_v)
        for child in kids:
            visit_bbox(child)

    for r in roots:
        visit_bbox(r)
    bbmin, bbmax = overall_min, overall_max
    if np.any(overall_min > overall_max):
        bbmin = np.zeros(3)
        bbmax = np.zeros(3)
    bbrange = (bbmin + bbmax) / 2.0

    results = {}

    def visit(mesh, parent_matrix_sub, is_tree_root):
        name = mesh['name']
        kids = children_of(mesh)
        has_children = len(kids) > 0
        is_root = is_tree_root

        matrix1 = mat4.identity()
        if is_root:
            if has_children:
                matrix1 = mat4.translate(matrix1, (-bbrange[0], -bbmax[1], -bbrange[2]))
            else:
                matrix1 = mat4.translate(matrix1, (0.0, -bbmax[1] + bbrange[1], 0.0))
        else:
            matrix1 = mat4.translate(matrix1, mesh['pos'])

        if mesh.get('frames'):
            quat = _sample_rotation_quat(mesh['frames'], anim_time)
            if quat is not None:
                matrix1 = matrix1 @ mat4.quat_to_mat4(quat)
        elif abs(mesh.get('rot_angle', 0) or 0) > 0.01:
            matrix1 = mat4.rotate(matrix1, mesh['rot_angle'], mesh['rot_axis'])

        sx, sy, sz = _sample_scale(mesh.get('scale_keyframes') or [], mesh.get('scale', (1,1,1)), anim_time)
        matrix1 = mat4.scale(matrix1, (sx, sy, sz))

        matrix2 = mat4.identity()
        if is_root and not has_children:
            matrix2 = mat4.translate(matrix2, (-bbrange[0], -bbrange[1], -bbrange[2]))
        if (not is_root) or has_children:
            matrix2 = mat4.translate(matrix2, mesh['pos_'])
        offset_m4 = mat4.mat3_to_mat4(mesh['offset_matrix'])
        matrix2 = matrix2 @ offset_m4

        world_matrix_sub = parent_matrix_sub @ matrix1
        world_matrix = world_matrix_sub @ matrix2
        results[name] = world_matrix

        for child in kids:
            visit(child, world_matrix_sub, False)

    for r in roots:
        visit(r, mat4.identity(), True)

    # Mallas huérfanas (padre no encontrado): tratar como raíz extra
    for m in rsm.meshes:
        if m['name'] not in results:
            visit(m, mat4.identity(), True)

    # y_flip RO→GL.
    # Ancla ÚNICA (diagnóstico hierarchy OK en mapa):
    #   XZ = origen del nodo raíz (tras jerarquía)
    #   Y  = minY del bbox (suelo) — no tocar
    # Ya no se usa centro-vs-raíz por proporción (Ctrl+{ } solo queda como legacy no-op).
    y_flip = mat4.scale(mat4.identity(), (1.0, -1.0, 1.0))
    for name in list(results.keys()):
        results[name] = y_flip @ results[name]

    anchor = fixed_anchor
    was_elongated = False
    # Metadatos de ancla para log (solo al calcular, no en animación)
    _anchor_meta = None
    if anchor is None:
        all_pts = []
        for mesh in rsm.meshes:
            mat = results.get(mesh['name'])
            if mat is None or not mesh['vertices']:
                continue
            verts = np.array(mesh['vertices'], dtype=np.float64)
            ones = np.ones((verts.shape[0], 1), dtype=np.float64)
            world = (mat.astype(np.float64) @ np.hstack([verts, ones]).T).T[:, :3]
            if len(world):
                all_pts.append(world)
        root_mesh = roots[0] if roots else (rsm.meshes[0] if rsm.meshes else None)
        rx = ry = rz = 0.0
        root_name = '?'
        if root_mesh is not None and root_mesh['name'] in results:
            o = results[root_mesh['name']].astype(np.float64) @ np.array(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float64
            )
            rx, ry, rz = float(o[0]), float(o[1]), float(o[2])
            root_name = root_mesh['name']
        if all_pts:
            pts = np.vstack(all_pts)
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            c = pts.mean(axis=0)
            min_y = float(mn[1])
            sx = float(mx[0] - mn[0])
            sy = float(mx[1] - mn[1])
            sz = float(mx[2] - mn[2])
            horiz_max = max(sx, sz)
            horiz_min = max(min(sx, sz), 1e-4)
            dims = sorted([sx, sy, sz])
            mean_d = max((sx + sy + sz) / 3.0, 1e-4)
            # Flag informativo
            was_elongated = (horiz_max / horiz_min) >= float(elong_horiz) or (dims[2] / mean_d) >= float(elong_mean)
            # Pivot XZ = origen del root (lo que planta el RSW/cliente).
            # Y = minY. Centro bbox solo se loguea para diagnóstico.
            ax, az = rx, rz
            anchor = np.array([ax, min_y, az], dtype=np.float64)
            _anchor_meta = {
                'root_name': root_name,
                'root_xz': (rx, rz),
                'bbox_center_xz': (float(c[0]), float(c[2])),
                'delta_xz': (float(c[0]) - rx, float(c[2]) - rz),
                'delta_len': float(np.hypot(c[0] - rx, c[2] - rz)),
                'min_y': min_y,
                'bbox_size': (sx, sy, sz),
                'elongated': was_elongated,
                'anchor_mode': 'root_xz+minY',
            }
        else:
            anchor = np.array([rx, 0.0, rz], dtype=np.float64)
            _anchor_meta = {
                'root_name': root_name,
                'root_xz': (rx, rz),
                'bbox_center_xz': (rx, rz),
                'delta_xz': (0.0, 0.0),
                'delta_len': 0.0,
                'min_y': 0.0,
                'bbox_size': (0.0, 0.0, 0.0),
                'elongated': False,
            }
    else:
        # Ancla fija (animación): no recalcular; espejos según flags
        was_elongated = bool(elong_mirror_x or elong_mirror_z)
        _anchor_meta = None

    recent = mat4.translate(
        mat4.identity(),
        (-float(anchor[0]), -float(anchor[1]), -float(anchor[2])),
    )
    for name in list(results.keys()):
        results[name] = recent @ results[name]

    # Espejo X local post-ancla (convención RO/BrowEdit tras y_flip).
    # Sin esto el molino queda con vástago/estructura al lado contrario
    # (validado antes en el visor RSM). Se aplica a TODOS los modelos.
    x_mirror = mat4.scale(mat4.identity(), (-1.0, 1.0, 1.0))
    for name in list(results.keys()):
        results[name] = x_mirror @ results[name]

    do_mx = force_mirror_x or (was_elongated and elong_mirror_x)
    do_mz = force_mirror_z or (was_elongated and elong_mirror_z)
    if do_mx:
        mx = mat4.scale(mat4.identity(), (-1.0, 1.0, 1.0))
        for name in list(results.keys()):
            results[name] = mx @ results[name]
    if do_mz:
        mz = mat4.scale(mat4.identity(), (1.0, 1.0, -1.0))
        for name in list(results.keys()):
            results[name] = mz @ results[name]

    if isinstance(anchor, np.ndarray):
        anchor = np.array(anchor, dtype=np.float64, copy=True)
        anchor.flags.writeable = True
    # Metadatos de ancla para log (None si animación con fixed_anchor)
    compute_model_transforms.last_anchor_meta = _anchor_meta
    return results, bbmin, bbmax, anchor, was_elongated


def compute_node_transform(mesh):
    """
    Reproduce Rsm::Mesh::calcMatrix1 / calcMatrix2 de BrowEdit, para el
    caso: nodo raíz (parent==None) SIN hijos y SIN animación (frames
    vacíos). Devuelve (matrix1, matrix2, bbmin, bbmax) como matrices
    numpy 4x4 listas para multiplicar.

    Se mantiene por compatibilidad con el visor de un solo modelo
    (rsm_viewer_3d.py la sigue usando para el caso simple). Para modelos
    con jerarquía padre-hijo, usar compute_model_transforms() en su lugar
    — esta función NO maneja esos casos.
    """
    offset_m4 = mat4.mat3_to_mat4(mesh['offset_matrix'])

    # --- Paso 1: bounding box en espacio local ---
    # setBoundingBox: v = offset * vertex; como parent==None y
    # children==[], NO se suma pos+pos_ para este cálculo.
    verts = np.array(mesh['vertices'], dtype=np.float64)
    ones = np.ones((verts.shape[0], 1))
    verts_h = np.hstack([verts, ones])  # Nx4
    transformed = (offset_m4.astype(np.float64) @ verts_h.T).T[:, :3]

    bbmin = transformed.min(axis=0)
    bbmax = transformed.max(axis=0)
    bbrange = (bbmin + bbmax) / 2.0

    # --- matrix1: raíz sin hijos -> translate(0, -bbmax.y+bbrange.y, 0) ---
    matrix1 = mat4.identity()
    matrix1 = mat4.translate(matrix1, (0.0, -bbmax[1] + bbrange[1], 0.0))
    if abs(mesh['rot_angle']) > 0.01:
        # rot_angle del .rsm está en radianes (ver calcMatrix1 de BrowEdit)
        matrix1 = mat4.rotate(matrix1, mesh['rot_angle'], mesh['rot_axis'])
    sx, sy, sz = mesh['scale']
    matrix1 = mat4.scale(matrix1, (sx, sy, sz))

    # --- matrix2: raíz sin hijos -> translate(-bbrange) * offset ---
    matrix2 = mat4.identity()
    matrix2 = mat4.translate(matrix2, (-bbrange[0], -bbrange[1], -bbrange[2]))
    matrix2 = matrix2 @ offset_m4

    return matrix1, matrix2, bbmin, bbmax


def build_vertex_buffers(mesh):
    """
    Desenrolla las caras indexadas en buffers de vértices planos
    (posición, uv, normal), agrupados por texture_id, ya que cada grupo
    necesita un bind de textura distinto para dibujarse.
    Devuelve: {texture_id: np.array de floats (N*8,)}  [3 pos + 2 uv + 3 normal]
    """
    vertices = mesh['vertices']
    texcoords = mesh['texcoords']
    local_to_global = mesh.get('texture_indices') or []
    groups = {}

    for face in mesh['faces']:
        vi = face['vertex_ids']
        ti = face['texcoord_ids']
        local_id = face['texture_id']
        tex_id = local_to_global[local_id] if local_id < len(local_to_global) else local_id

        p0 = np.array(vertices[vi[0]])
        p1 = np.array(vertices[vi[1]])
        p2 = np.array(vertices[vi[2]])
        normal = np.cross(p1 - p0, p2 - p0)
        n = np.linalg.norm(normal)
        normal = normal / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])

        buf = groups.setdefault(tex_id, [])
        for k in range(3):
            pos = vertices[vi[k]]
            uv = texcoords[ti[k]]
            buf.extend([pos[0], pos[1], pos[2], uv[0], uv[1],
                        normal[0], normal[1], normal[2]])

    return {tid: np.array(buf, dtype=np.float32) for tid, buf in groups.items()}


def apply_magenta_colorkey(img):
    """
    Convierte los píxeles magenta puro (255,0,255) — el color-key de
    transparencia clásico de las texturas BMP de RO, que no tienen canal
    alfa real — en transparencia de verdad (alpha=0). Además, copia el
    color de un vecino no-transparente hacia esos píxeles (en vez de
    dejarlos en magenta) para que el filtrado bilineal/mipmaps no mezcle
    magenta hacia adentro de los bordes de la hoja/rama.
    """
    arr = np.array(img)  # RGBA
    magenta = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 60) & (arr[:, :, 2] > 200)
    if not magenta.any():
        return img

    arr[magenta, 3] = 0

    # "Dilatar" el color de los píxeles opacos vecinos hacia adentro de
    # las zonas transparentes (unas pocas iteraciones alcanza para tapar
    # el halo que produce el filtrado bilineal en los bordes).
    rgb = arr[:, :, :3].astype(np.float32)
    opaque = ~magenta
    # Hay que cubrir TODO el relleno transparente: con mipmaps, un magenta
    # que quede a más de 2 px del borde se mezcla en los niveles bajos y da
    # un halo rosado en hojas/banderines vistos de lejos.
    for _ in range(32):
        if opaque.all():
            break
        filled = opaque.copy()
        src = rgb.copy()
        # desplazar en las 4 direcciones y copiar donde todavía es transparente
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_opaque = np.roll(opaque, (dy, dx), axis=(0, 1))
            shifted_rgb = np.roll(src, (dy, dx), axis=(0, 1))
            need_fill = (~filled) & shifted_opaque
            rgb[need_fill] = shifted_rgb[need_fill]
            filled |= need_fill
        opaque = filled

    # Lo que no alcanzó la dilatación: color medio de los píxeles opacos
    keep = ~magenta
    if keep.any() and not opaque.all():
        rgb[~opaque] = rgb[keep].mean(axis=0)

    arr[:, :, :3] = rgb.astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')


def load_texture_gl(path, generate_mipmaps=True):
    """Carga BMP/TGA/PNG/JPG con PIL y la sube como textura GL. Devuelve id GL o None.
    Deja en load_texture_gl.last_translucent si la textura tiene alfa parcial.

    generate_mipmaps=True (default): mipmaps al cargar (visor de un solo modelo).
    generate_mipmaps=False: solo nivel 0 + GL_LINEAR → carga más rápida; quien llama genera
    los mipmaps después (ver TerrainViewer3D._mipmap_pending).
    """
    load_texture_gl.last_translucent = False
    if not PIL_AVAILABLE:
        return None
    # Resolver ruta: exacta, case-insensitive, o misma base con otra extensión
    candidates = []
    if path:
        candidates.append(path)
        base, ext = os.path.splitext(path)
        for e in ('.bmp', '.BMP', '.tga', '.TGA', '.png', '.PNG', '.jpg', '.JPG', '.jpeg'):
            if e.lower() != ext.lower():
                candidates.append(base + e)
        # también buscar solo por nombre de archivo en el mismo dir
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        if dirname and os.path.isdir(dirname):
            lower_map = {f.lower(): f for f in os.listdir(dirname)}
            if basename.lower() in lower_map:
                candidates.append(os.path.join(dirname, lower_map[basename.lower()]))
            stem = os.path.splitext(basename)[0].lower()
            for f in os.listdir(dirname):
                if os.path.splitext(f)[0].lower() == stem:
                    candidates.append(os.path.join(dirname, f))

    for cand in candidates:
        if not cand or not os.path.isfile(cand):
            continue
        try:
            img = Image.open(cand).convert('RGBA')
            try:
                _a = np.asarray(img.getchannel('A'))
                load_texture_gl.last_translucent = bool(((_a > 8) & (_a < 248)).mean() > 0.02)
            except Exception:
                load_texture_gl.last_translucent = False
            img = apply_magenta_colorkey(img)
            img_data = img.tobytes()
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            if generate_mipmaps:
                glGenerateMipmap(GL_TEXTURE_2D)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            else:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            return tex_id
        except Exception as e:
            print(f"[3D] No se pudo cargar textura {cand}: {e}")
            continue
    return None




def _sample_tex_offset(anims, frame):
    """Interpola offset de animación de textura RSM2 (type 1=U, 2=V según fuente).

    En la práctica del fountain, type=1 desplaza V (caída de agua).
    Soporta varios anims y combina (u, v).
    """
    u_off, v_off = 0.0, 0.0
    for anim in (anims or []):
        frames = anim.get('frames') or []
        if not frames:
            continue
        atype = anim.get('type', 1)
        if frame <= frames[0]['frame']:
            val = float(frames[0]['offset'])
        elif frame >= frames[-1]['frame']:
            val = float(frames[-1]['offset'])
        else:
            val = float(frames[-1]['offset'])
            for i in range(len(frames) - 1):
                a, b = frames[i], frames[i + 1]
                if a['frame'] <= frame <= b['frame']:
                    span = b['frame'] - a['frame']
                    t = 0.0 if span <= 0 else (frame - a['frame']) / float(span)
                    val = a['offset'] + (b['offset'] - a['offset']) * t
                    break
        # type 1 → V (agua), type 2 → U, otros → V
        if atype == 2:
            u_off += val
        else:
            v_off += val
    return (u_off, v_off)


def _normalize_rsm2_mesh(mesh):
    """Adapta un dict de RSM2Parser al shape del renderer RSM1."""
    frames = []
    for kf in (mesh.get('rot_keyframes') or []):
        frames.append({'time': float(kf['frame']), 'quat': kf['quat']})
    scale_keyframes = []
    for kf in (mesh.get('scale_keyframes') or []):
        s = kf['scale']
        scale_keyframes.append({'frame': float(kf['frame']), 'scale': tuple(s)})
    matrix = mesh.get('matrix') or (1, 0, 0, 0, 1, 0, 0, 0, 1)
    pos_ = mesh.get('position_') or (0.0, 0.0, 0.0)
    return {
        'name': mesh['name'],
        'parent_name': mesh.get('parent_name') or '',
        'offset_matrix': tuple(matrix),
        'pos': (0.0, 0.0, 0.0),
        'pos_': tuple(pos_),
        'rot_angle': 0.0,
        'rot_axis': (0.0, 1.0, 0.0),
        'scale': (1.0, 1.0, 1.0),
        'vertices': mesh.get('vertices') or [],
        'texcoords': mesh.get('texcoords') or [],
        'faces': mesh.get('faces') or [],
        'texture_indices': mesh.get('texture_indices') or [],
        'frames': frames,
        'scale_keyframes': scale_keyframes,
        'texture_keyframe_groups': mesh.get('texture_keyframe_groups') or [],
    }


def compute_model_transforms_rsm2(rsm2, anim_time=0.0):
    """
    Matrices mundo RSM2 con animación suave.

    anim_time = índice de frame (0 .. anim_len), no milisegundos.
    Jerarquía como RSM2Parser.compute_world_matrices + keyframes.
    Sin espejo Y (RSM2 ya está en espacio del cliente).
    """
    anim_len = float(rsm2.anim_len) if rsm2.anim_len > 0 else 1.0
    frame = float(anim_time) % anim_len

    def sample_quat(keyframes, fr):
        if not keyframes:
            return None
        if len(keyframes) == 1 or fr <= keyframes[0]['frame']:
            return keyframes[0]['quat']
        if fr >= keyframes[-1]['frame']:
            return keyframes[-1]['quat']
        for i in range(len(keyframes) - 1):
            a, b = keyframes[i], keyframes[i + 1]
            if a['frame'] <= fr <= b['frame']:
                span = b['frame'] - a['frame']
                u = 0.0 if span <= 0 else (fr - a['frame']) / float(span)
                return mat4.quat_slerp(a['quat'], b['quat'], u)
        return keyframes[-1]['quat']

    def sample_scale_kf(keyframes, fr):
        if not keyframes:
            return (1.0, 1.0, 1.0)
        if len(keyframes) == 1 or fr <= keyframes[0]['frame']:
            return tuple(keyframes[0]['scale'])
        if fr >= keyframes[-1]['frame']:
            return tuple(keyframes[-1]['scale'])
        for i in range(len(keyframes) - 1):
            a, b = keyframes[i], keyframes[i + 1]
            if a['frame'] <= fr <= b['frame']:
                span = b['frame'] - a['frame']
                u = 0.0 if span <= 0 else (fr - a['frame']) / float(span)
                sa, sb = a['scale'], b['scale']
                return (
                    sa[0] + (sb[0] - sa[0]) * u,
                    sa[1] + (sb[1] - sa[1]) * u,
                    sa[2] + (sb[2] - sa[2]) * u,
                )
        return tuple(keyframes[-1]['scale'])

    def mat3_to_mat4_row(m9):
        a = np.identity(4, dtype=np.float64)
        a[0, 0:3] = m9[0:3]
        a[1, 0:3] = m9[3:6]
        a[2, 0:3] = m9[6:9]
        return a

    roots = [m for m in rsm2.meshes if not m.get('parent')]
    queue, seen, order = list(roots), set(), []
    while queue:
        m = queue.pop(0)
        if m['name'] in seen:
            continue
        seen.add(m['name'])
        order.append(m)
        queue.extend(m.get('children') or [])
    for m in rsm2.meshes:
        if m['name'] not in seen:
            order.append(m)

    local_matrix = {}
    world_matrix = {}

    for mesh in order:
        name = mesh['name']
        parent = mesh.get('parent')

        quat = sample_quat(mesh.get('rot_keyframes') or [], frame)
        if quat is not None:
            transform = mat4.quat_to_mat4(quat).astype(np.float64)
        else:
            transform = mat3_to_mat4_row(mesh['matrix'])

        sx, sy, sz = sample_scale_kf(mesh.get('scale_keyframes') or [], frame)
        transform = transform @ np.diag([sx, sy, sz, 1.0]).astype(np.float64)

        if parent is None:
            local_matrix[name] = transform.copy()
            position = np.array(mesh['position_'], dtype=np.float64)
        else:
            # matriz estática del padre para el offset relativo (como parser)
            parent_static = mat3_to_mat4_row(parent['matrix'])
            try:
                parent_inv = np.linalg.inv(parent_static)
            except np.linalg.LinAlgError:
                parent_inv = np.identity(4, dtype=np.float64)
            local_matrix[name] = transform @ parent_inv

            pos_diff = (
                np.array(mesh['position_'], dtype=np.float64)
                - np.array(parent['position_'], dtype=np.float64)
            )
            pos_diff_h = np.array([pos_diff[0], pos_diff[1], pos_diff[2], 0.0])
            position = (parent_inv @ pos_diff_h)[:3]

        mesh_self = local_matrix[name].copy()
        mesh_self[0:3, 3] = position

        result = mesh_self
        m = mesh
        while m.get('parent') is not None:
            m = m['parent']
            result = result @ local_matrix[m['name']]

        if parent is not None:
            result = result.copy()
            result[0:3, 3] = mesh_self[0:3, 3] + world_matrix[parent['name']][0:3, 3]

        world_matrix[name] = result

    all_min = np.array([1e9, 1e9, 1e9])
    all_max = np.array([-1e9, -1e9, -1e9])
    for mesh in rsm2.meshes:
        w = world_matrix[mesh['name']]
        for v in mesh['vertices']:
            wv = (w @ np.array([v[0], v[1], v[2], 1.0]))[:3]
            all_min = np.minimum(all_min, wv)
            all_max = np.maximum(all_max, wv)
    return world_matrix, all_min, all_max


class RSMViewer3D:
    def __init__(self, rsm_path, texture_dir=None, width=1024, height=768):
        self.rsm_path = rsm_path
        self.texture_dir = texture_dir or os.path.dirname(os.path.abspath(rsm_path))
        self.width = width
        self.height = height

        ext = os.path.splitext(rsm_path)[1].lower()
        self.is_rsm2 = (ext == '.rsm2')
        # Misma carga y matrices que el mapa (rsm_placement / BrowEdit3)
        if self.is_rsm2:
            self.rsm = RP.load_rsm2(rsm_path)
            self.rsm2_raw = None  # compat: ya no se usa el parser crudo en el draw
        else:
            self.rsm = RSMParser(rsm_path)
            if self.rsm.error:
                raise RSMFileError(self.rsm.error)
            if not self.rsm.meshes:
                raise RSMFileError("El .rsm no contiene ninguna malla")
            self.rsm2_raw = None

        self.anim_time = 0.0
        self.anim_playing = True
        self.anim_speed = 1.0
        self.has_animation = any(
            (m.get('frames') or m.get('scale_keyframes') or m.get('pos_keyframes')
             or m.get('texture_keyframe_groups'))
            for m in self.rsm.meshes
        ) and float(getattr(self.rsm, 'anim_len', 0) or 0) > 0

        # Bbox fijo en t=0; matrices de visualización = misma convención que el mapa
        self.world_matrices, self.rb_min, self.rb_range = RP.display_matrices(
            self.rsm, 0.0
        )
        self.bbmin = np.asarray(self.rb_min, dtype=np.float64)
        # Centro / radio desde vértices ya transformados (espacio de visualización)
        all_world_verts = []
        for mesh in self.rsm.meshes:
            world = self.world_matrices.get(mesh['name'])
            verts = np.array(mesh['vertices'], dtype=np.float64)
            if world is None or len(verts) == 0:
                continue
            ones = np.ones((verts.shape[0], 1), dtype=np.float64)
            world_verts = (world.astype(np.float64) @ np.hstack([verts, ones]).T).T[:, :3]
            all_world_verts.append(world_verts)
        if all_world_verts:
            all_world_verts = np.vstack(all_world_verts)
            world_min = all_world_verts.min(axis=0)
            world_max = all_world_verts.max(axis=0)
        else:
            world_min = np.zeros(3)
            world_max = np.ones(3)
        self.bbmax = world_max
        world_center = (world_min + world_max) / 2.0
        diag = world_max - world_min
        self.model_radius = max(float(np.linalg.norm(diag)) / 2.0, 1.0)

        self.mesh_vertex_groups = {
            mesh['name']: build_vertex_buffers(mesh) for mesh in self.rsm.meshes
        }

        self.cam_yaw = 45.0
        self.display_x_flip = False
        self.cam_pitch = 15.0
        self.cam_dist = self.model_radius * 2.6
        self.cam_target = tuple(float(c) for c in world_center)

        self._init_gl(width, height)
        self._upload_geometry()
        self._load_textures()

    def _init_gl(self, width, height):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                         pygame.GL_CONTEXT_PROFILE_CORE)
        import os
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        self._display_flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        pygame.display.set_mode((width, height), self._display_flags)
        pygame.display.set_caption("RSM 3D Viewer" + (" [RSM2]" if getattr(self, "is_rsm2", False) else " [RSM1]"))

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)  # espejo Y invierte winding
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.15, 0.16, 0.20, 1.0)

        vs = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fs = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vs, fs)

        self.u_mvp = glGetUniformLocation(self.program, "u_mvp")
        self.u_model = glGetUniformLocation(self.program, "u_model")
        self.u_uv_offset = glGetUniformLocation(self.program, "u_uv_offset")
        self.u_texture = glGetUniformLocation(self.program, "u_texture")
        self.u_light_dir = glGetUniformLocation(self.program, "u_light_dir")
        self.u_diffuse = glGetUniformLocation(self.program, "u_diffuse_color")
        self.u_ambient = glGetUniformLocation(self.program, "u_ambient_color")
        self.u_intensity = glGetUniformLocation(self.program, "u_intensity")
        self.u_use_texture = glGetUniformLocation(self.program, "u_use_texture")

    def _upload_geometry(self):
        # self.gl_objects: lista de (vao, vbo, count, tex_id, world_matrix)
        # — una entrada por (malla, grupo_de_textura), ya que cada malla
        # tiene su propia matriz de mundo.
        self.gl_objects = []
        stride = 8 * 4  # 8 floats por vértice
        for mesh in self.rsm.meshes:
            groups = self.mesh_vertex_groups[mesh['name']]
            for tex_id, buf in groups.items():
                vao = glGenVertexArrays(1)
                vbo = glGenBuffers(1)
                glBindVertexArray(vao)
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, buf.nbytes, buf, GL_STATIC_DRAW)

                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
                glEnableVertexAttribArray(2)

                glBindVertexArray(0)
                self.gl_objects.append((vao, vbo, len(buf) // 8, tex_id, mesh['name']))

    def _load_textures(self):
        self.gl_textures = {}
        self._translucent_tex = set()   # texturas con alfa parcial (TGA) → 2ª pasada
        for i, tex_name in enumerate(self.rsm.textures):
            # Nombres con separador Windows y subcarpetas; BMP y TGA.
            norm = tex_name.replace(chr(92), '/')
            base = norm.split('/')[-1]
            stem = os.path.splitext(base)[0]
            candidates = [
                os.path.join(self.texture_dir, base),
                os.path.join(self.texture_dir, norm),
                os.path.join(self.texture_dir, stem + '.tga'),
                os.path.join(self.texture_dir, stem + '.TGA'),
                os.path.join(self.texture_dir, stem + '.bmp'),
                os.path.join(self.texture_dir, stem + '.BMP'),
                os.path.join(self.texture_dir, stem + '.png'),
            ]
            tex_id_gl = None
            for c in candidates:
                tex_id_gl = load_texture_gl(c)
                if tex_id_gl is not None:
                    break
            self.gl_textures[i] = tex_id_gl
            if tex_id_gl is not None and getattr(load_texture_gl, 'last_translucent', False):
                self._translucent_tex.add(i)
            status = "OK" if tex_id_gl is not None else "NO ENCONTRADA"
            if tex_id_gl is None:
                log_warn(f"[3D] Textura {i} ('{base}'): {status}")

    def _camera_position(self):
        yaw = math.radians(self.cam_yaw)
        pitch = math.radians(self.cam_pitch)
        x = self.cam_dist * math.cos(pitch) * math.sin(yaw)
        y = self.cam_dist * math.sin(pitch)
        z = self.cam_dist * math.cos(pitch) * math.cos(yaw)
        return (x + self.cam_target[0], y + self.cam_target[1], z + self.cam_target[2])

    def render_frame(self):
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)

        proj = mat4.perspective(50.0, self.width / self.height, 0.1, max(self.model_radius * 50, 100))
        eye = self._camera_position()
        view = mat4.look_at(eye, self.cam_target, (0, 1, 0))

        # Luz: usamos una dirección fija razonable (arriba/adelante). Si
        # más adelante conectamos el RSW real, Longitude/Latitude de
        # Rsw.Light se pueden convertir a esta misma dirección.
        glUniform3f(self.u_light_dir, -0.4, -0.8, -0.4)
        glUniform3f(self.u_diffuse, 1.0, 1.0, 1.0)
        glUniform3f(self.u_ambient, 0.45, 0.45, 0.45)
        glUniform1f(self.u_intensity, 0.8)

        # mapa nombre → mesh para keyframes de textura
        mesh_by_name = {m['name']: m for m in self.rsm.meshes}
        tex_frame = float(self.anim_time) if getattr(self, 'is_rsm2', False) else 0.0

        # Dos pasadas: opaco primero; texturas con alfa parcial o animadas (ondas,
        # brillos) después, de lejos a cerca y SIN escribir profundidad. Así se ven
        # desde arriba y desde abajo, como una transparencia (igual que GRF Editor).
        def _is_trans(o):
            m = mesh_by_name.get(o[4])
            return (o[3] in self._translucent_tex) or bool(m and m.get('texture_keyframe_groups'))

        def _dist2(o):
            m = mesh_by_name.get(o[4])
            V = m['vertices'] if m else []
            c = np.append(np.mean(np.asarray(V, dtype=np.float64), axis=0), 1.0) if len(V) \
                else np.array([0.0, 0.0, 0.0, 1.0])
            w = self.world_matrices[o[4]] @ c
            return float((w[0] - eye[0]) ** 2 + (w[1] - eye[1]) ** 2 + (w[2] - eye[2]) ** 2)

        opaque_objs = [o for o in self.gl_objects if not _is_trans(o)]
        trans_objs = sorted((o for o in self.gl_objects if _is_trans(o)), key=lambda o: -_dist2(o))
        for _phase, _objs in ((0, opaque_objs), (1, trans_objs)):
            glDepthMask(GL_TRUE if _phase == 0 else GL_FALSE)
            for vao, vbo, count, tex_id, mesh_name in _objs:
                world = self.world_matrices[mesh_name]
                if getattr(self, 'display_x_flip', False):
                    xflip = mat4.scale(mat4.identity(), (-1.0, 1.0, 1.0))
                    # centrar el flip en el target de cámara para no desplazar el modelo
                    cx, cy, cz = self.cam_target
                    to_c = mat4.translate(mat4.identity(), (-cx, -cy, -cz))
                    fr_c = mat4.translate(mat4.identity(), (cx, cy, cz))
                    mvp = proj @ view @ fr_c @ xflip @ to_c @ world
                else:
                    mvp = proj @ view @ world
                glUniformMatrix4fv(self.u_mvp, 1, GL_FALSE, mat4.to_gl(mvp))
                glUniformMatrix4fv(self.u_model, 1, GL_FALSE, mat4.to_gl(world))

                # Animación de UV (RSM2 texture_keyframe_groups)
                u_off, v_off = 0.0, 0.0
                mesh = mesh_by_name.get(mesh_name)
                if mesh and mesh.get('texture_keyframe_groups'):
                    for grp in mesh['texture_keyframe_groups']:
                        # texture_id del grupo es local; si coincide con el del draw o hay uno solo
                        grp_local = grp.get('texture_id', 0)
                        # aplicar siempre el offset del grupo (un grupo por textura típico)
                        ou, ov = _sample_tex_offset(grp.get('anims'), tex_frame)
                        u_off += ou
                        v_off += ov
                glUniform2f(self.u_uv_offset, u_off, v_off)

                gl_tex = self.gl_textures.get(tex_id)
                if gl_tex is not None:
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, gl_tex)
                    glUniform1i(self.u_texture, 0)
                    glUniform1i(self.u_use_texture, 1)
                else:
                    glUniform1i(self.u_use_texture, 0)

                glBindVertexArray(vao)
                glDrawArrays(GL_TRIANGLES, 0, count)
                glBindVertexArray(0)

        glDepthMask(GL_TRUE)

        pygame.display.flip()

    def run(self):
        """
        Controles:
          - Click izquierdo + arrastrar: orbitar
          - Rueda: zoom
          - Espacio: pausar/reanudar animación
          - [ / ]: velocidad animación
          - R: reiniciar animación
          - ESC: salir
        """
        clock = pygame.time.Clock()
        dragging = False
        last_pos = (0, 0)
        running = True
        if getattr(self, 'has_animation', False):
            log_info(f"[3D] Animación: anim_len={self.rsm.anim_len}, "
                  f"speed={self.anim_speed}x (Espacio=pausa, [/]=vel, R=reset)")
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.width = max(320, int(event.w))
                    self.height = max(240, int(event.h))
                    try:
                        pygame.display.set_mode(
                            (self.width, self.height),
                            getattr(self, '_display_flags',
                                    pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE),
                        )
                    except Exception:
                        pass
                    from OpenGL.GL import glViewport
                    glViewport(0, 0, self.width, self.height)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and getattr(self, 'has_animation', False):
                        self.anim_playing = not self.anim_playing
                        print(f"[3D] {'PLAY' if self.anim_playing else 'PAUSE'} t={self.anim_time:.0f}")
                    elif event.key == pygame.K_LEFTBRACKET and getattr(self, 'has_animation', False):
                        self.anim_speed = max(0.1, self.anim_speed / 1.5)
                        print(f"[3D] speed={self.anim_speed:.2f}x")
                    elif event.key == pygame.K_RIGHTBRACKET and getattr(self, 'has_animation', False):
                        self.anim_speed = min(8.0, self.anim_speed * 1.5)
                        print(f"[3D] speed={self.anim_speed:.2f}x")
                    elif event.key == pygame.K_r and getattr(self, 'has_animation', False):
                        self.anim_time = 0.0
                        print("[3D] anim reset")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging = True
                        last_pos = event.pos
                    elif event.button == 4:
                        self.cam_dist = max(self.model_radius * 0.2, self.cam_dist * 0.9)
                    elif event.button == 5:
                        self.cam_dist *= 1.1
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = False
                elif event.type == pygame.MOUSEMOTION and dragging:
                    dx = event.pos[0] - last_pos[0]
                    dy = event.pos[1] - last_pos[1]
                    self.cam_yaw += dx * 0.4
                    self.cam_pitch = max(-89, min(89, self.cam_pitch - dy * 0.4))
                    last_pos = event.pos

            dt_ms = clock.tick(60)
            if getattr(self, 'has_animation', False) and self.anim_playing:
                if getattr(self, 'is_rsm2', False):
                    # RSM2: anim_len en FRAMES → avanzar a fps del archivo
                    fps = float(getattr(self.rsm, 'fps', 30.0) or 30.0)
                    self.anim_time += (dt_ms / 1000.0) * fps * self.anim_speed
                else:
                    # RSM1: anim_len en milisegundos
                    self.anim_time += dt_ms * self.anim_speed
                if self.rsm.anim_len > 0:
                    self.anim_time %= float(self.rsm.anim_len)
                self.world_matrices, _, _ = RP.display_matrices(
                    self.rsm, self.anim_time,
                    rb_min=getattr(self, 'rb_min', None),
                    rb_range=getattr(self, 'rb_range', None),
                )

            self.render_frame()
        pygame.quit()

    def save_screenshot(self, path):
        """Renderiza un frame y lo guarda a disco (útil para validar sin display real)."""
        self.render_frame()
        buf = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        img = Image.frombytes('RGB', (self.width, self.height), buf)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(path)
        print(f"[3D] Captura guardada en {path}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python rsm_viewer_3d.py <modelo.rsm> [carpeta_texturas]")
        sys.exit(1)
    rsm_path = sys.argv[1]
    tex_dir = sys.argv[2] if len(sys.argv) > 2 else None
    viewer = RSMViewer3D(rsm_path, texture_dir=tex_dir)
    viewer.run()


if __name__ == "__main__":
    main()
