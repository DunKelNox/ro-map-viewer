"""
Visor 3D de terreno de Ragnarok Online (.gnd + opcionalmente .rsw).
RO_TERRAIN_BUILD = 2026-10-08-deferred-tex-mesh-mip

Caras: top + north + east (paredes verticales del GND).
Controles:
  click izq + arrastrar = orbitar · rueda = zoom · ESC = salir
  C = mostrar/ocultar brújula (líneas)
  F = invertir dirección de caras north (debug geometría)
"""
import os
import time
from rolog import info as log_info, warn as log_warn, VERBOSE
import sys
import math
import ctypes
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mat4
from ro_map_viewer import GNDParser, RSWParser, GATParser
try:
    from ro_map_viewer import resolve_ro_texture_path
except ImportError:
    def resolve_ro_texture_path(tex_name, roots, basename_index=None):
        if not tex_name: return None
        base = tex_name.replace('\\','/').split('/')[-1]
        for root in (roots or []):
            for p in (os.path.join(root, base), os.path.join(root, tex_name.replace('\\','/'))):
                if os.path.isfile(p): return p
        return None

from rsm_viewer_3d import (
    apply_magenta_colorkey,
    RSMParser, RSMFileError, build_vertex_buffers,
    load_texture_gl, VERTEX_SHADER as RSM_VERTEX_SHADER,
    FRAGMENT_SHADER as RSM_FRAGMENT_SHADER,
    diagnose_rsm_hierarchy, format_rsm_diagnosis,
)
import rsm_placement as RP   # colocación RSM idéntica a BrowEdit3

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


TERRAIN_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_lm_uv;
uniform mat4 u_mvp;
uniform vec3 u_light_dir;
uniform vec3 u_diffuse_color;
uniform vec3 u_ambient_color;
uniform bool u_client_light;
out vec2 v_uv;
out vec2 v_lm_uv;
out vec3 v_normal;
out vec3 v_mult;
// Iluminación del cliente RO tal como la replica BrowEdit3 (gnd.vs): se calcula
// por vértice (interpolación lineal de color, como el cliente oficial) y mezcla
// los colores ambiente y difuso del RSW. La "intensidad" del RSW no interviene.
vec3 client_light_mult(vec3 nrm) {
    float NL = clamp(dot(normalize(nrm), -normalize(u_light_dir)), 0.0, 1.0);
    vec3 A = u_ambient_color;
    vec3 D = u_diffuse_color;
    vec3 af = (1.0 - A) * A;
    vec3 amb = A - af + af * D;
    vec3 df = (1.0 - D) * D;
    vec3 dif = D - df + df * A;
    vec3 mult1 = min(NL * dif + amb, 1.0);
    vec3 mult2 = min(max(D, A) + (1.0 - max(D, A)) * min(D, A), 1.0);
    return clamp(min(mult1, mult2), 0.0, 1.0);
}
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_uv = in_uv;
    v_lm_uv = in_lm_uv;
    v_normal = in_normal;
    v_mult = u_client_light ? client_light_mult(in_normal) : vec3(1.0);
}
"""

TERRAIN_FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
in vec2 v_lm_uv;
in vec3 v_normal;
in vec3 v_mult;
uniform bool u_client_light;
uniform sampler2D u_texture;
uniform sampler2D u_lightmap;
uniform bool u_use_texture;
uniform bool u_use_lightmap;
uniform vec4 u_flat_color;
uniform vec3 u_light_dir;
uniform vec3 u_diffuse_color;
uniform vec3 u_ambient_color;
uniform float u_intensity;
uniform bool u_lm_invert_ao;
out vec4 frag_color;
void main() {
    vec4 base = u_use_texture ? texture(u_texture, v_uv) : u_flat_color;
    // Color-key RO (magenta) + alpha
    if (u_use_texture && base.a < 0.1)
        discard;
    if (u_use_texture && base.r > 0.85 && base.g < 0.20 && base.b > 0.85)
        discard;
    if (u_client_light) {
        // Igual que gnd.fs de BrowEdit3: textura * luz(vértice) * sombra(alfa del
        // lightmap) + color del lightmap (aditivo).
        vec3 c = base.rgb * v_mult;
        if (u_use_lightmap) {
            vec4 lm = texture(u_lightmap, v_lm_uv);
            float ao = lm.a;
            if (u_lm_invert_ao) ao = 1.0 - ao;
            c *= clamp(ao, 0.0, 1.0);
            c += clamp(lm.rgb, 0.0, 1.0);
        }
        frag_color = vec4(c, base.a);
        return;
    }
    vec3 n = normalize(v_normal);
    float ndotl = max(dot(n, -normalize(u_light_dir)), 0.05);
    vec3 lit = u_ambient_color + u_diffuse_color * (ndotl * u_intensity);
    lit = clamp(lit, 0.0, 1.4);
    vec3 rgb = base.rgb * lit;
    if (u_use_lightmap) {
        vec4 lm = texture(u_lightmap, v_lm_uv);
        // Solo AO (sombra): ao=1 → factor 1.0 (misma luz que sin S)
        // pow >1 aumenta contraste/definición como en GRF
        float ao = lm.a;
        if (u_lm_invert_ao) ao = 1.0 - ao;
        ao = clamp(ao, 0.0, 1.0);
        // Suave como GRF: menos contraste, sin suelo tan negro
        float shadow = mix(0.28, 1.0, pow(ao, 1.15));
        rgb *= shadow;
        // Tint de color muy suave
        rgb *= mix(vec3(1.0), max(lm.rgb, vec3(0.88)), 0.20 * ao);
    }
    frag_color = vec4(rgb, base.a);
}
"""


# --- Agua animada (modo 3) -------------------------------------------------
WATER_ALPHA = 0.62            # opacidad del agua con texturas RO (BrowEdit3 usa 0.564)
WATER_OPAQUE_TYPES = (4,)     # tipos de agua que BrowEdit3 dibuja opacos
WATER_TEX_TILE = 40.0   # unidades de mundo que cubre una repetición de la textura de agua

WATER_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 in_xz;
uniform mat4 u_mvp;
uniform float u_level_y;
uniform float u_time;        // segundos
uniform float u_wave_h;
uniform float u_wave_speed;
uniform float u_wave_pitch;
out vec3 v_world;
void main() {
    float k = 6.2831853 / max(u_wave_pitch, 1.0);
    float ph = (in_xz.x + in_xz.y) * k + u_time * u_wave_speed * 2.0;
    float y = u_level_y + u_wave_h * sin(ph);
    v_world = vec3(in_xz.x, y, in_xz.y);
    gl_Position = u_mvp * vec4(v_world, 1.0);
}
"""

WATER_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_world;
uniform sampler2D u_tex_a;
uniform sampler2D u_tex_b;
uniform float u_mix;
uniform bool u_use_tex;
uniform float u_time;
uniform float u_tile;
uniform float u_alpha;
out vec4 frag_color;
void main() {
    vec3 rgb;
    if (u_use_tex) {
        vec2 uv = v_world.xz / u_tile;
        rgb = mix(texture(u_tex_a, uv).rgb, texture(u_tex_b, uv).rgb, u_mix);
    } else {
        // Sin texturas de agua: ondulación procedural
        vec2 p = v_world.xz;
        float t = u_time;
        float w1 = sin(p.x * 0.35 + t * 1.3) + sin(p.y * 0.29 - t * 1.1);
        float w2 = sin((p.x + p.y) * 0.21 + t * 0.8) + sin((p.x - p.y) * 0.43 - t * 1.7);
        float r = (w1 + w2) * 0.25;
        vec3 deep = vec3(0.10, 0.28, 0.58);
        vec3 shallow = vec3(0.32, 0.58, 0.85);
        rgb = mix(deep, shallow, r * 0.5 + 0.5);
        rgb += smoothstep(0.80, 1.0, r) * 0.30;
    }
    frag_color = vec4(rgb, u_alpha);
}
"""


def rsw_light_direction(rsw, client=True):
    """Dirección en que VIAJA la luz (sistema de mundo del visor, Y arriba).

    client=True: misma orientación que BrowEdit3 / cliente RO. Se comprobó con
    las sombras horneadas del mapa (ver README): las sombras caen en el
    sentido que da esta fórmula.
    client=False: fórmula anterior del visor (azimut distinto), solo para
    comparar con la tecla T.
    """
    if rsw is None or rsw.light is None:
        return (-0.4, -0.8, -0.4)
    lon = math.radians(rsw.light['longitude'])
    lat = math.radians(rsw.light['latitude'])
    if client:
        return (math.sin(lon) * math.sin(lat),
                -math.cos(lat),
                -math.cos(lon) * math.sin(lat))
    x = -math.cos(lon) * math.sin(lat)
    y = math.cos(lat)
    z = math.sin(lon) * math.sin(lat)
    return (x, -y, z)




def scene_light_params(rsw, use_rsw_light=True, for_models=False):
    """Devuelve (ambiente, difusa, intensidad) para los shaders.

    Con use_rsw_light=True se usan los valores reales del .rsw (colores
    ambiente y difuso incluidos), como GRF Editor / el cliente: un mapa con
    luz azulada se ve azulado/violeta. Con False (o sin RSW) se conserva la
    luz neutra anterior.
    """
    light = getattr(rsw, 'light', None) if rsw is not None else None
    if use_rsw_light and light:
        def _c(v):
            return tuple(max(0.0, min(1.0, float(x))) for x in v)
        return (_c(light['ambient']), _c(light['diffuse']),
                max(0.0, float(light.get('intensity', 0.5))))
    # --- modo anterior (luz neutra) ---
    if for_models:
        return ((0.45, 0.45, 0.45), (1.0, 1.0, 1.0), 0.9)
    intensity = 0.9
    if light:
        intensity = max(0.5, float(light.get('intensity', 0.5)) * 1.4)
    return ((0.35, 0.35, 0.35), (1.0, 1.0, 1.0), intensity)


def _scale3(x, y, z):
    s = mat4.identity()
    s[0, 0], s[1, 1], s[2, 2] = x, y, z
    return s



def instance_world_matrix(gnd_width, gnd_height, pos, rot, scale,
                          rb_min=None, rb_range=None, rsm2=False,
                          off_x=0.0, off_z=0.0):
    """Matriz de instancia RSW (BrowEdit3 / rsm_placement).

    RSM1: bbox real (rb_min / rb_range) centra XZ y apoya minY.
    RSM2: instance_matrix_rsm2.
    off_x / off_z: nudge global de diagnóstico (J/L I/K).
    """
    if rb_min is None:
        rb_min = (0.0, 0.0, 0.0)
    if rb_range is None:
        rb_range = (0.0, 0.0, 0.0)
    if rsm2:
        world = RP.instance_matrix_rsm2(gnd_width, gnd_height, pos, rot, scale)
    else:
        world = RP.instance_matrix(gnd_width, gnd_height, pos, rot, scale, rb_min, rb_range)
    if off_x or off_z:
        world = RP._t((float(off_x), 0.0, float(off_z))) @ world
    return world


class ModelInstanceSet:
    _missing_tex_all = set()   # texturas de modelos no encontradas (resumen al final)

    """Geometría + texturas de un .rsm reutilizable en muchas instancias."""

    def __init__(self, rsm_path, texture_roots):
        self._rsm_path = rsm_path
        self.is_rsm2 = rsm_path.lower().endswith('.rsm2')
        self.rsm = RP.load_rsm2(rsm_path) if self.is_rsm2 else RSMParser(rsm_path)
        if self.rsm.error:
            raise RSMFileError(self.rsm.error)
        if not self.rsm.meshes:
            raise RSMFileError("Sin mallas")
        try:
            self.hierarchy_diag = diagnose_rsm_hierarchy(self.rsm)
        except Exception as _e:
            self.hierarchy_diag = {'has_issues': True, 'issues': ['diag_error'], 'error': str(_e)}
        self.anim_time = 0.0
        # Animación de textura (RSM 2.3): {(nombre_malla, id_textura_global): anims}
        self.tex_anims = {}
        for _m in self.rsm.meshes:
            _idxs = _m.get('texture_indices') or []
            for _g in (_m.get('texture_keyframe_groups') or []):
                _loc = _g.get('texture_id', 0)
                _glob = _idxs[_loc] if 0 <= _loc < len(_idxs) else _loc
                if _g.get('anims'):
                    self.tex_anims[(_m['name'], _glob)] = _g['anims']
        # Colocación BrowEdit: matrices de malla en t=0 + bbox REAL del modelo.
        # El modelo se centra por rb_range (XZ) y apoya rb_min.y en y=0
        # (ver rsm_placement.instance_matrix); ya no hay "ancla" de nodo raíz.
        (self.world_matrices, self.rb_min,
         self.rb_max, self.rb_range) = RP.model_placement_data(self.rsm)
        # Atributos de compatibilidad con código antiguo
        self.bbmin, self.bbmax = self.rb_min, self.rb_max
        self._rsm_anchor = None
        self.is_elongated = False
        self.anchor_meta = None
        self.mesh_vertex_groups = {
            mesh['name']: build_vertex_buffers(mesh) for mesh in self.rsm.meshes
        }
        self.gl_objects = []
        self._upload()
        self.gl_textures = {}
        self.translucent_tex = set()   # texturas con alfa parcial (TGA) → 2ª pasada
        self._ctr = {}                 # caché: centro local de cada malla
        self._load_textures(texture_roots)
        n_mesh = len(self.rsm.meshes)
        n_anim = sum(1 for m in self.rsm.meshes
                     if m.get('frames') or m.get('scale_keyframes') or m.get('pos_keyframes'))
        self.has_anim = n_anim > 0
        # duración: anim_len del header o último keyframe
        self.anim_len = float(getattr(self.rsm, 'anim_len', 0) or 0)
        if self.is_rsm2 and self.anim_len > 0:
            self.anim_len = self.anim_len / float(self.rsm.fps) * 1000.0   # frames → ms
        if self.anim_len <= 0:
            tmax = 0.0
            for m in self.rsm.meshes:
                for fr in (m.get('frames') or []):
                    tmax = max(tmax, float(fr.get('time', 0) or 0))
                for fr in (m.get('scale_keyframes') or []):
                    tmax = max(tmax, float(fr.get('frame', 0) or 0))
            self.anim_len = tmax if tmax > 0 else 1000.0
        n_draw = len(self.gl_objects)
        if n_mesh > 1 or n_anim:
            log_info(f"[3D]   mallas={n_mesh} animadas={n_anim} draw_groups={n_draw} "
                  f"bbY=[{self.bbmin[1]:.1f},{self.bbmax[1]:.1f}] anim_len={self.anim_len:.0f}")

    def _upload(self):
        stride = 8 * 4
        for mesh in self.rsm.meshes:
            mesh_matrix = self.world_matrices[mesh['name']]
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
                # mesh_name para actualizar matriz en animación
                self.gl_objects.append((vao, vbo, len(buf) // 8, tex_id, mesh['name']))

    def mesh_center(self, name):
        """Centro local (homogéneo) de una malla; para ordenar capas translúcidas."""
        c = self._ctr.get(name)
        if c is None:
            V = next((m['vertices'] for m in self.rsm.meshes if m['name'] == name), [])
            c = (np.append(np.mean(np.asarray(V, dtype=np.float64), axis=0), 1.0) if len(V)
                 else np.array([0.0, 0.0, 0.0, 1.0]))
            self._ctr[name] = c
        return c

    def recompute_anchor(self, *args, **kwargs):
        """Legacy no-op: ya no existen ancla ni umbrales de elongación."""
        return

    def set_anim_time(self, t):
        """Recalcula las matrices de malla para el tiempo de animación t.

        Como en BrowEdit, la bbox real (rb_min / rb_range) queda fija en t=0:
        solo cambian las matrices de las mallas animadas, así que el modelo
        no 'levita' ni se corre mientras gira.
        """
        if not self.has_anim:
            return
        self.anim_time = t
        if self.is_rsm2:
            self.world_matrices = RP.compute_mesh_matrices(
                self.rsm, float(t) / 1000.0 * float(self.rsm.fps))   # ms → frames
        else:
            self.world_matrices = RP.compute_mesh_matrices(self.rsm, float(t))

    def _load_textures(self, texture_roots):
        """Misma resolución que texturas de mapa: roots + walk basename + encoding."""
        roots = list(texture_roots or [])
        rsm_dir = getattr(self, '_rsm_path', None)
        if rsm_dir:
            rsm_dir = os.path.dirname(os.path.abspath(rsm_dir))
        extra = []
        if rsm_dir:
            extra.extend([
                rsm_dir,
                os.path.join(rsm_dir, '..'),
                os.path.join(rsm_dir, '..', '..'),
                os.path.join(rsm_dir, 'texture'),
                os.path.join(rsm_dir, '..', 'texture'),
                os.path.join(rsm_dir, '..', 'data'),
                os.path.join(rsm_dir, '..', 'data', 'texture'),
                os.path.join(rsm_dir, '..', '..', 'data', 'texture'),
            ])
        for e in extra:
            if e and os.path.isdir(e):
                abs_e = os.path.abspath(e)
                if abs_e not in roots:
                    roots.append(abs_e)
        if not hasattr(ModelInstanceSet, '_shared_tex_index'):
            ModelInstanceSet._shared_tex_index = {}
        idx = ModelInstanceSet._shared_tex_index

        found = 0
        missing = []
        for i, tex_name in enumerate(self.rsm.textures):
            path = None
            names = [tex_name]
            for enc_from, enc_to in (
                ('euc-kr', 'latin-1'), ('euc-kr', 'cp1252'),
                ('cp949', 'latin-1'), ('utf-8', 'latin-1'),
                ('latin-1', 'euc-kr'), ('cp1252', 'euc-kr'),
                ('latin-1', 'cp949'), ('cp1252', 'cp949'),
            ):
                try:
                    alt = tex_name.encode(enc_from).decode(enc_to)
                    if alt not in names:
                        names.append(alt)
                except (UnicodeError, LookupError):
                    pass
                try:
                    alt = tex_name.encode(enc_to).decode(enc_from)
                    if alt not in names:
                        names.append(alt)
                except (UnicodeError, LookupError):
                    pass

            for name in names:
                path = resolve_ro_texture_path(name, roots, idx)
                if path:
                    break
            gl_id = load_texture_gl(path, generate_mipmaps=False) if path else None
            self.gl_textures[i] = gl_id
            if gl_id and getattr(load_texture_gl, 'last_translucent', False):
                self.translucent_tex.add(i)
            if gl_id:
                found += 1
                # mipmaps diferidos (cola compartida con el visor de terreno)
                q = getattr(ModelInstanceSet, '_mipmap_queue', None)
                if q is not None:
                    q.append(gl_id)
            else:
                missing.append(tex_name)
                ModelInstanceSet._missing_tex_all.add(tex_name)

        n = len(self.rsm.textures)
        log_info(f"[3D]   texturas RSM: {found}/{n}")
        if missing and found == 0:
            log_info(f"[3D]   no encontradas (muestra): {missing[:5]}")
        elif missing:
            log_info(f"[3D]   faltan {len(missing)}: {missing[:3]}")


class TerrainViewer3D:
    def __init__(self, gnd_path, rsw_path=None, texture_dir=None,
                 width=1024, height=768, max_dim=None, water_dir=None):
        self.gnd_path = gnd_path
        self.water_dir = water_dir     # carpeta elegida en el launcher (water000..031.jpg)
        map_dir = os.path.dirname(os.path.abspath(gnd_path))
        self.texture_dir = texture_dir or map_dir
        self.width = width
        self.height = height
        self.max_dim = max_dim

        self.texture_roots = []
        for candidate in (
            self.texture_dir, map_dir,
            os.path.join(map_dir, "data"),
            os.path.join(map_dir, "data", "texture"),
            os.path.join(map_dir, "texture"),
            os.path.join(map_dir, ".."),
            os.path.join(map_dir, "..", "data"),
            os.path.join(map_dir, "..", "data", "texture"),
            os.path.join(map_dir, "..", "texture"),
            os.path.join(self.texture_dir, "data"),
            os.path.join(self.texture_dir, "data", "texture"),
            os.path.join(self.texture_dir, "texture"),
        ):
            if candidate and os.path.isdir(candidate):
                abs_c = os.path.abspath(candidate)
                if abs_c not in self.texture_roots:
                    self.texture_roots.append(abs_c)
        self._tex_basename_index = {}
        # compartir índice con RSM
        ModelInstanceSet._shared_tex_index = self._tex_basename_index

        self.gnd = GNDParser(gnd_path)
        if self.gnd.error:
            raise RuntimeError(f"No se pudo cargar el GND: {self.gnd.error}")

        self.rsw = RSWParser(rsw_path) if rsw_path else None
        self._collect_water_zones()

        # GAT (caminable) — mismo nombre base que el .gnd
        self.gat = None
        gat_path = os.path.splitext(gnd_path)[0] + ".gat"
        if os.path.isfile(gat_path):
            try:
                self.gat = GATParser(gat_path, strict=False)
                if getattr(self.gat, 'error', None):
                    print(f"[3D] GAT no cargado: {self.gat.error}")
                    self.gat = None
                else:
                    log_info(f"[3D] GAT {self.gat.width}x{self.gat.height} cargado")
            except Exception as e:
                print(f"[3D] GAT error: {e}")
                self.gat = None
        else:
            print(f"[3D] No se encontró {gat_path}")

        self.north_dir = 1
        self.show_walls = True
        self.flip_z = True
        self.smooth_normals = True  # N — normales suaves por vértice (como el cliente RO)
        self.use_rsw_light = True   # T — luz ambiente/difusa/intensidad reales del .rsw
        self.show_compass = True
        self.show_walkable = False  # G
        self.show_models = False    # M — marcadores RSW
        self.show_rsm = True        # R — modelos 3D RSM
        self.rsm_animate = False    # A — animar RSM (off por defecto)
        self.rsm_draw_dist = 1400.0  # culling distancia (aprobado)
        self.rsm_cull_fov = 100.0   # cono precarga (~55° FOV + margen)
        self._rsm_anim_clock = 0.0  # tiempo acumulado ms
        # Espejo X va en el RSM (matrix2); instancia sin S(-)
        # RSM: sin presets Ctrl+N (legacy desactivado)
        # Nudge global XZ (J/L I/K; Home=reset) — solo medición, no solución por modelo
        self.model_off_x = 0.0
        self.model_off_z = 0.0
        self.show_shadows = True   # S — lightmaps GND (sombras horneadas)
        self.lm_invert_ao = False  # I — invertir canal AO si sale negro
        self.lm_flip_u = False     # confirmado (U x1 desde True)
        self.lm_flip_v = True      # confirmado (O x1)
        self.lm_wall_mode_north = 0  # confirmado: none (Shift+P x4 desde 2)
        self.lm_wall_mode_east = 2   # confirmado: flipU (P x1 desde 1)
        # GAT suele ir espejado respecto al GND en uno o ambos ejes
        # Mapeo GAT = versión de referencia (puente en el lado correcto)
        self.gat_flip_x = False
        self.gat_flip_y = True
        self.gat_flip_heights = False  # tecla 3
        self.gat_off_x = 0.0   # teclas 4/5
        self.gat_off_z = 10.0  # confirmado: tecla 7 x2

        self._init_gl(width, height)
        self._build_lightmap_atlas()
        self._build_terrain_mesh()
        self._build_water_mesh()
        self.water_mode = 2          # 1=sin agua, 2=agua simple (actual), 3=agua animada
        self._water_time_ms = 0.0
        self._init_water_animated()
        self._build_walkable_overlay()
        self._build_model_markers()
        # Texturas GND diferidas: el terreno se ve al toque (color plano) y
        # las texturas llegan por _pump_loading (ver _gnd_tex_iter).
        self._prepare_texture_loading()
        self._init_rsm_program()
        self._load_rsm_instances()
        self._init_compass_geometry()

        diag_x = self.gnd.width * 10
        diag_z = self.gnd.height * 10
        self.model_radius = max(diag_x, diag_z, 100) / 2.0 * 1.4
        self.cam_target = (diag_x / 2.0, 0.0, diag_z / 2.0)
        self.cam_yaw = 45.0
        self.cam_pitch = 35.0
        self.cam_dist = self.model_radius * 1.6

    def _init_gl(self, width, height):
        pygame.init()
        # Ventana normal del SO: se puede mover y redimensionar (no pantalla completa)
        import os
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                         pygame.GL_CONTEXT_PROFILE_CORE)
        self._display_flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        pygame.display.set_mode((width, height), self._display_flags)
        pygame.display.set_caption("RO Terrain 3D Viewer")
        self.width = width
        self.height = height
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.5, 0.65, 0.85, 1.0)

        vs = shaders.compileShader(TERRAIN_VERTEX_SHADER, GL_VERTEX_SHADER)
        fs = shaders.compileShader(TERRAIN_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vs, fs)
        self.u_mvp = glGetUniformLocation(self.program, "u_mvp")
        self.u_texture = glGetUniformLocation(self.program, "u_texture")
        self.u_use_texture = glGetUniformLocation(self.program, "u_use_texture")
        self.u_flat_color = glGetUniformLocation(self.program, "u_flat_color")
        self.u_light_dir = glGetUniformLocation(self.program, "u_light_dir")
        self.u_client_light = glGetUniformLocation(self.program, "u_client_light")
        self.u_diffuse = glGetUniformLocation(self.program, "u_diffuse_color")
        self.u_ambient = glGetUniformLocation(self.program, "u_ambient_color")
        self.u_intensity = glGetUniformLocation(self.program, "u_intensity")
        self.u_lightmap = glGetUniformLocation(self.program, "u_lightmap")
        self.u_use_lightmap = glGetUniformLocation(self.program, "u_use_lightmap")
        self.u_lm_invert_ao = glGetUniformLocation(self.program, "u_lm_invert_ao")

    def _cube_corner(self, x, y, corner, h):
        gh = self.gnd.height
        # flip_z: intercambia offset Z de bordes top/bottom (solo afecta eje verde)
        dz_bottom, dz_top = (10.0, 0.0) if getattr(self, 'flip_z', False) else (0.0, 10.0)
        if corner == 'bottom_left':
            return (10.0 * x, -h, 10.0 * gh - 10.0 * y + dz_bottom)
        elif corner == 'bottom_right':
            return (10.0 * x + 10.0, -h, 10.0 * gh - 10.0 * y + dz_bottom)
        elif corner == 'top_left':
            return (10.0 * x, -h, 10.0 * gh - 10.0 * y + dz_top)
        elif corner == 'top_right':
            return (10.0 * x + 10.0, -h, 10.0 * gh - 10.0 * y + dz_top)
        raise ValueError(corner)


    def _build_lightmap_atlas(self):
        """Empaqueta lightmaps en atlas con padding (bleed) para suavizar bordes.

        Cada slice 8x8 se coloca en una celda (lw+2pad)x(lh+2pad) duplicando
        el borde. Así el filtrado bilineal no cruza al tile vecino (cortes duros).
        """
        self.lm_atlas_tex = None
        self.lm_atlas_info = {}          # UV estilo BrowEdit3 (inset 1.0 px)
        self.lm_atlas_info_legacy = {}   # UV anterior del visor (inset 1.5 px)
        lms = getattr(self.gnd, 'lightmaps', None) or []
        if not lms:
            log_info("[3D] Sin lightmaps en el GND")
            return
        lw = getattr(self.gnd, 'lightmap_width', 8) or 8
        lh = getattr(self.gnd, 'lightmap_height', 8) or 8
        pad = 1  # px de bleed alrededor de cada slice
        cell_w, cell_h = lw + 2 * pad, lh + 2 * pad
        n = len(lms)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = max(1, int(math.ceil(n / float(cols))))
        aw, ah = cols * cell_w, rows * cell_h
        atlas = bytearray(aw * ah * 4)
        for i in range(0, len(atlas), 4):
            atlas[i:i+4] = bytes((255, 255, 255, 255))

        def put_px(ax, ay, rgba_src, src_off):
            if ax < 0 or ay < 0 or ax >= aw or ay >= ah:
                return
            if src_off + 3 >= len(rgba_src):
                return
            dst = (ay * aw + ax) * 4
            atlas[dst:dst+4] = rgba_src[src_off:src_off+4]

        for i, rgba in enumerate(lms):
            col, row = i % cols, i // cols
            # origen del contenido (sin pad)
            x0, y0 = col * cell_w + pad, row * cell_h + pad
            # copiar 8x8
            for y in range(lh):
                for x in range(lw):
                    src = (y * lw + x) * 4
                    put_px(x0 + x, y0 + y, rgba, src)
            # bleed: duplicar bordes hacia el pad
            for x in range(lw):
                put_px(x0 + x, y0 - 1, rgba, (0 * lw + x) * 4)          # top
                put_px(x0 + x, y0 + lh, rgba, ((lh - 1) * lw + x) * 4)  # bottom
            for y in range(lh):
                put_px(x0 - 1, y0 + y, rgba, (y * lw + 0) * 4)          # left
                put_px(x0 + lw, y0 + y, rgba, (y * lw + (lw - 1)) * 4)  # right
            # esquinas
            put_px(x0 - 1, y0 - 1, rgba, 0)
            put_px(x0 + lw, y0 - 1, rgba, (lw - 1) * 4)
            put_px(x0 - 1, y0 + lh, rgba, ((lh - 1) * lw) * 4)
            put_px(x0 + lw, y0 + lh, rgba, ((lh - 1) * lw + (lw - 1)) * 4)

            # Inset ~1.5px (como el arreglo de paredes) + pad exterior
            # (el pad evita cortes en piso; el inset evita manchas en bordes de muro)
            # BrowEdit3 (GndRenderer::Chunk::rebuild) mapea cada celda de +1 a
            # lw-1 px del bloque lw x lh: el borde de 1 px del lightmap trae los
            # valores de la celda vecina, y con filtro LINEAR eso da una
            # transición continua entre celdas (salto 0.0000 medido en Comodo).
            # Con el inset anterior (1.5 px) se muestreaba un texel distinto a
            # cada lado del borde: escalones en sombras y parches cuadrados.
            inset_bro = 1.0 if min(lw, lh) >= 4 else 0.5
            self.lm_atlas_info[i] = (
                (x0 + inset_bro) / float(aw), (y0 + inset_bro) / float(ah),
                (x0 + lw - inset_bro) / float(aw), (y0 + lh - inset_bro) / float(ah))
            inset = 1.5 if min(lw, lh) >= 4 else 0.75
            self.lm_atlas_info_legacy[i] = (
                (x0 + inset) / float(aw), (y0 + inset) / float(ah),
                (x0 + lw - inset) / float(aw), (y0 + lh - inset) / float(ah))

        if not PIL_AVAILABLE:
            # subir con OpenGL directo
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, aw, ah, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes(atlas))
            self.lm_atlas_tex = tex
        else:
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, aw, ah, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes(atlas))
            self.lm_atlas_tex = tex
        # Diagnóstico: media de AO y RGB del primer slice
        if lms:
            s0 = lms[0]
            n_pix = lw * lh
            ao_vals = [s0[i*4+3] for i in range(n_pix)]
            r_vals = [s0[i*4] for i in range(n_pix)]
            ao_mean = sum(ao_vals) / max(1, len(ao_vals))
            r_mean = sum(r_vals) / max(1, len(r_vals))
            log_info(f"[3D] Lightmap atlas {aw}x{ah} ({n} slices, {cols}x{rows}) | "
                  f"slice0 AO_mean={ao_mean:.0f} R_mean={r_mean:.0f}")
            if ao_mean < 64:
                log_info("[3D] AO muy bajo → probá tecla I (invertir sombra)")
        else:
            log_info(f"[3D] Lightmap atlas {aw}x{ah} ({n} slices)")

    def _build_terrain_mesh(self):
        """Top + paredes SOLO cuando el GND trae surface_north/east (sin auto-walls).

        Optimizado: caras TOP vectorizadas con NumPy; paredes con aritmética
        pura (sin np.asarray por triángulo). Mismo resultado visual.
        """
        t0 = time.perf_counter()
        gw, gh = self.gnd.width, self.gnd.height
        max_x = min(gw, self.max_dim) if self.max_dim else gw
        max_y = min(gh, self.max_dim) if self.max_dim else gh
        cubes = self.gnd.cubes
        surfaces = self.gnd.surfaces
        n_surfaces = len(surfaces)
        n_tex = len(self.gnd.textures)
        NO_TEX_KEY = -1
        groups = {}
        nd = self.north_dir
        flip_z = getattr(self, 'flip_z', False)
        show_walls = getattr(self, 'show_walls', True)
        if getattr(self, 'use_rsw_light', True):
            lm_info = getattr(self, 'lm_atlas_info', None) or {}
        else:
            lm_info = getattr(self, 'lm_atlas_info_legacy', None) or {}
        flip_u = getattr(self, 'lm_flip_u', True)
        flip_v = getattr(self, 'lm_flip_v', False)
        mode_n = getattr(self, 'lm_wall_mode_north', 2)
        mode_e = getattr(self, 'lm_wall_mode_east', 1)
        dz_bottom, dz_top = (10.0, 0.0) if flip_z else (0.0, 10.0)
        z_base = 10.0 * gh

        # --- Pre-extraer datos de cubos a arrays (1 pass) ---
        n_cells = max_y * max_x
        h_bl = np.empty(n_cells, dtype=np.float64)
        h_br = np.empty(n_cells, dtype=np.float64)
        h_tl = np.empty(n_cells, dtype=np.float64)
        h_tr = np.empty(n_cells, dtype=np.float64)
        sid_top = np.empty(n_cells, dtype=np.int32)
        sid_north = np.empty(n_cells, dtype=np.int32)
        sid_east = np.empty(n_cells, dtype=np.int32)
        for y in range(max_y):
            row = y * max_x
            src_row = y * gw
            for x in range(max_x):
                i = row + x
                c = cubes[src_row + x]
                h_bl[i] = c['h_bottom_left']
                h_br[i] = c['h_bottom_right']
                h_tl[i] = c['h_top_left']
                h_tr[i] = c['h_top_right']
                sid_top[i] = c['surface_top']
                sid_north[i] = c['surface_north']
                sid_east[i] = c['surface_east']

        def lm_uvs(surface):
            lm_id = surface.get('lightmap_id', -1)
            if lm_id is None or lm_id < 0:
                return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
            info = lm_info.get(lm_id)
            if info is None:
                return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
            u0, v0, u1, v1 = info
            if flip_u:
                u0, u1 = u1, u0
            if flip_v:
                v0, v1 = v1, v0
            return (u0, v1), (u1, v1), (u0, v0), (u1, v0)

        def add_tri(tex_id, p0, p1, p2, uv0, uv1, uv2, lm0, lm1, lm2):
            ax, ay, az = p0
            bx, by, bz = p1
            cx, cy, cz = p2
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            nlen2 = nx * nx + ny * ny + nz * nz
            if nlen2 < 1e-20:
                return
            inv = 1.0 / (nlen2 ** 0.5)
            nx, ny, nz = nx * inv, ny * inv, nz * inv
            buf = groups.get(tex_id)
            if buf is None:
                buf = []
                groups[tex_id] = buf
            buf.extend((
                ax, ay, az, uv0[0], uv0[1], nx, ny, nz, lm0[0], lm0[1],
                bx, by, bz, uv1[0], uv1[1], nx, ny, nz, lm1[0], lm1[1],
                cx, cy, cz, uv2[0], uv2[1], nx, ny, nz, lm2[0], lm2[1],
            ))

        def add_quad(tex_id, p0, p1, p2, p3, uv0, uv1, uv2, uv3, lm0, lm1, lm2, lm3):
            add_tri(tex_id, p0, p1, p2, uv0, uv1, uv2, lm0, lm1, lm2)
            add_tri(tex_id, p0, p2, p3, uv0, uv2, uv3, lm0, lm2, lm3)

        def resolve_tex(sid):
            if not (0 <= sid < n_surfaces):
                return None
            surface = surfaces[sid]
            tid = surface['texture_id']
            if not (0 <= tid < n_tex):
                tid = NO_TEX_KEY
            return tid, surface

        # ========== TOP: vectorizado ==========
        valid_top = (sid_top >= 0) & (sid_top < n_surfaces)
        idx = np.nonzero(valid_top)[0]
        n_top = int(idx.size)
        if n_top:
            ys = idx // max_x
            xs = idx % max_x
            # posiciones de las 4 esquinas (N,)
            px0 = 10.0 * xs
            px1 = px0 + 10.0
            pz_bot = z_base - 10.0 * ys + dz_bottom
            pz_top = z_base - 10.0 * ys + dz_top
            # (N, 3) por esquina
            p_bl = np.column_stack((px0, -h_bl[idx], pz_bot))
            p_br = np.column_stack((px1, -h_br[idx], pz_bot))
            p_tl = np.column_stack((px0, -h_tl[idx], pz_top))
            p_tr = np.column_stack((px1, -h_tr[idx], pz_top))
            # tri 0: bl, br, tl  |  tri 1: br, tr, tl
            # normales
            e1 = p_br - p_bl
            e2 = p_tl - p_bl
            n0 = np.cross(e1, e2)
            nlen0 = np.linalg.norm(n0, axis=1, keepdims=True)
            good0 = nlen0[:, 0] > 1e-10
            n0 = np.divide(n0, nlen0, out=np.zeros_like(n0), where=nlen0 > 1e-10)

            e1b = p_tr - p_br
            e2b = p_tl - p_br
            n1 = np.cross(e1b, e2b)
            nlen1 = np.linalg.norm(n1, axis=1, keepdims=True)
            good1 = nlen1[:, 0] > 1e-10
            n1 = np.divide(n1, nlen1, out=np.zeros_like(n1), where=nlen1 > 1e-10)

            # --- Normales por vértice ---------------------------------------
            # Por defecto cada triángulo usa su normal plana, lo que marca el
            # corte en diagonal de cada celda (rombos en la arena). El cliente
            # RO / BrowEdit3 promedian las normales de las celdas que comparten
            # la esquina. Para no suavizar acantilados, solo se mezclan celdas
            # cuya esquina compartida está a la MISMA altura.
            if getattr(self, 'smooth_normals', True):
                # Igual que Gnd::Cube::calcNormal de BrowEdit3: cada esquina suma
                # las normales de los triángulos que la tocan (bl: n0, br/tl:
                # n0+n1, tr: n1). Ojo: en una esquina sin vecinos el cliente usa
                # n0 también para tr ("it's just how it is in the client").
                nsum = n0 + n1
                nc_bl, nc_br, nc_tl, nc_tr = n0, nsum, nsum, n1
                k_bot = ys - int(round(dz_bottom / 10.0))      # línea Z borde bottom
                k_top = ys - int(round(dz_top / 10.0))         # línea Z borde top
                def _q(h):
                    return np.round(h[idx] * 100.0).astype(np.int64)
                keys = np.concatenate((
                    np.stack((xs,     k_bot, _q(h_bl)), axis=1),   # bl
                    np.stack((xs + 1, k_bot, _q(h_br)), axis=1),   # br
                    np.stack((xs,     k_top, _q(h_tl)), axis=1),   # tl
                    np.stack((xs + 1, k_top, _q(h_tr)), axis=1),   # tr
                ))
                _, inv = np.unique(keys, axis=0, return_inverse=True)
                inv = inv.reshape(-1)
                acc = np.zeros((int(inv.max()) + 1, 3), dtype=np.float64)
                np.add.at(acc, inv, np.concatenate((nc_bl, nc_br, nc_tl, nc_tr)))
                inv_all = inv.reshape(4, n_top)
                def _unit(v):
                    l = np.linalg.norm(v, axis=1, keepdims=True)
                    return np.divide(v, l, out=np.zeros_like(v), where=l > 1e-10)
                vn_bl = _unit(acc[inv_all[0]])
                vn_br = _unit(acc[inv_all[1]])
                vn_tl = _unit(acc[inv_all[2]])
                # esquina tr: la propia celda aporta n0 (valor por defecto del
                # cliente) en lugar de n1; los vecinos sí aportan su n1.
                vn_tr = _unit(acc[inv_all[3]] + (n0 - n1))
                # celdas degeneradas (normal 0): conservar la plana
                for _vn, _fb in ((vn_bl, n0), (vn_br, n0), (vn_tl, n0), (vn_tr, n1)):
                    bad = np.linalg.norm(_vn, axis=1) < 1e-6
                    if np.any(bad):
                        _vn[bad] = _fb[bad]
            else:
                vn_bl = vn_br = vn_tl = n0
                vn_tr = n1

            # Agrupar por tex_id (pocas texturas distintas → pocos grupos)
            tex_ids = np.empty(n_top, dtype=np.int32)
            # UVs y LM se leen por superficie (no vectorizable limpio sin atlas de UV)
            uv_bl = np.empty((n_top, 2), dtype=np.float64)
            uv_br = np.empty((n_top, 2), dtype=np.float64)
            uv_tl = np.empty((n_top, 2), dtype=np.float64)
            uv_tr = np.empty((n_top, 2), dtype=np.float64)
            lm_bl = np.zeros((n_top, 2), dtype=np.float64)
            lm_br = np.zeros((n_top, 2), dtype=np.float64)
            lm_tl = np.zeros((n_top, 2), dtype=np.float64)
            lm_tr = np.zeros((n_top, 2), dtype=np.float64)
            for k, si in enumerate(sid_top[idx]):
                surf = surfaces[int(si)]
                tid = surf['texture_id']
                tex_ids[k] = tid if 0 <= tid < n_tex else NO_TEX_KEY
                if tex_ids[k] != NO_TEX_KEY:
                    uv_bl[k] = surf['uv_bottom_left']
                    uv_br[k] = surf['uv_bottom_right']
                    uv_tl[k] = surf['uv_top_left']
                    uv_tr[k] = surf['uv_top_right']
                else:
                    uv_bl[k] = (0.0, 1.0)
                    uv_br[k] = (1.0, 1.0)
                    uv_tl[k] = (0.0, 0.0)
                    uv_tr[k] = (1.0, 0.0)
                l0, l1, l2, l3 = lm_uvs(surf)
                lm_bl[k] = l0
                lm_br[k] = l1
                lm_tl[k] = l2
                lm_tr[k] = l3

            # Empaquetar vértices: 2 tris × 3 verts × 10 floats
            for tid in np.unique(tex_ids):
                mask = tex_ids == tid
                m0 = mask & good0
                m1 = mask & good1
                chunks = []
                if np.any(m0):
                    # v0=bl, v1=br, v2=tl
                    v = np.empty((int(m0.sum()), 3, 10), dtype=np.float32)
                    v[:, 0, 0:3] = p_bl[m0]
                    v[:, 0, 3:5] = uv_bl[m0]
                    v[:, 0, 5:8] = vn_bl[m0]
                    v[:, 0, 8:10] = lm_bl[m0]
                    v[:, 1, 0:3] = p_br[m0]
                    v[:, 1, 3:5] = uv_br[m0]
                    v[:, 1, 5:8] = vn_br[m0]
                    v[:, 1, 8:10] = lm_br[m0]
                    v[:, 2, 0:3] = p_tl[m0]
                    v[:, 2, 3:5] = uv_tl[m0]
                    v[:, 2, 5:8] = vn_tl[m0]
                    v[:, 2, 8:10] = lm_tl[m0]
                    chunks.append(v.reshape(-1))
                if np.any(m1):
                    # v0=br, v1=tr, v2=tl
                    v = np.empty((int(m1.sum()), 3, 10), dtype=np.float32)
                    v[:, 0, 0:3] = p_br[m1]
                    v[:, 0, 3:5] = uv_br[m1]
                    v[:, 0, 5:8] = vn_br[m1]
                    v[:, 0, 8:10] = lm_br[m1]
                    v[:, 1, 0:3] = p_tr[m1]
                    v[:, 1, 3:5] = uv_tr[m1]
                    v[:, 1, 5:8] = vn_tr[m1]
                    v[:, 1, 8:10] = lm_tr[m1]
                    v[:, 2, 0:3] = p_tl[m1]
                    v[:, 2, 3:5] = uv_tl[m1]
                    v[:, 2, 5:8] = vn_tl[m1]
                    v[:, 2, 8:10] = lm_tl[m1]
                    chunks.append(v.reshape(-1))
                if chunks:
                    arr = np.concatenate(chunks)
                    # groups guarda list para que las paredes puedan extend(); al final → array
                    groups[int(tid)] = arr.tolist()

        # ========== NORTH / EAST: lógica original, aritmética pura ==========
        n_north = n_east = 0
        if show_walls:
            for y in range(max_y):
                for x in range(max_x):
                    i = y * max_x + x
                    # ---- NORTH ----
                    sn = int(sid_north[i])
                    ny = y + nd
                    if 0 <= sn < n_surfaces and 0 <= ny < max_y:
                        ni = ny * max_x + x
                        if 0 <= int(sid_top[ni]) < n_surfaces:
                            info = resolve_tex(sn)
                            if info is not None:
                                tex_id, surface = info
                                if nd > 0:
                                    if flip_z:
                                        ha1, ha2 = float(h_tl[i]), float(h_tr[i])
                                        hb1, hb2 = float(h_bl[ni]), float(h_br[ni])
                                        a = (10.0 * x, -ha1, z_base - 10.0 * y + dz_top)
                                        b = (10.0 * x + 10.0, -ha2, z_base - 10.0 * y + dz_top)
                                        c = (10.0 * x + 10.0, -hb2, z_base - 10.0 * ny + dz_bottom)
                                        d = (10.0 * x, -hb1, z_base - 10.0 * ny + dz_bottom)
                                    else:
                                        ha1, ha2 = float(h_bl[i]), float(h_br[i])
                                        hb1, hb2 = float(h_tl[ni]), float(h_tr[ni])
                                        a = (10.0 * x, -ha1, z_base - 10.0 * y + dz_bottom)
                                        b = (10.0 * x + 10.0, -ha2, z_base - 10.0 * y + dz_bottom)
                                        c = (10.0 * x + 10.0, -hb2, z_base - 10.0 * ny + dz_top)
                                        d = (10.0 * x, -hb1, z_base - 10.0 * ny + dz_top)
                                else:
                                    if flip_z:
                                        ha1, ha2 = float(h_bl[i]), float(h_br[i])
                                        hb1, hb2 = float(h_tl[ni]), float(h_tr[ni])
                                        a = (10.0 * x, -ha1, z_base - 10.0 * y + dz_bottom)
                                        b = (10.0 * x + 10.0, -ha2, z_base - 10.0 * y + dz_bottom)
                                        c = (10.0 * x + 10.0, -hb2, z_base - 10.0 * ny + dz_top)
                                        d = (10.0 * x, -hb1, z_base - 10.0 * ny + dz_top)
                                    else:
                                        ha1, ha2 = float(h_tl[i]), float(h_tr[i])
                                        hb1, hb2 = float(h_bl[ni]), float(h_br[ni])
                                        a = (10.0 * x, -ha1, z_base - 10.0 * y + dz_top)
                                        b = (10.0 * x + 10.0, -ha2, z_base - 10.0 * y + dz_top)
                                        c = (10.0 * x + 10.0, -hb2, z_base - 10.0 * ny + dz_bottom)
                                        d = (10.0 * x, -hb1, z_base - 10.0 * ny + dz_bottom)
                                if abs(ha1 - hb1) > 0.01 or abs(ha2 - hb2) > 0.01:
                                    if tex_id != NO_TEX_KEY:
                                        uv_a = surface['uv_bottom_left']
                                        uv_b = surface['uv_bottom_right']
                                        uv_c = surface['uv_top_right']
                                        uv_d = surface['uv_top_left']
                                    else:
                                        uv_a, uv_b, uv_c, uv_d = (0, 1), (1, 1), (1, 0), (0, 0)
                                    lm_bl_, lm_br_, lm_tl_, lm_tr_ = lm_uvs(surface)
                                    lm_a, lm_b, lm_c, lm_d = lm_bl_, lm_br_, lm_tr_, lm_tl_
                                    if mode_n == 1:
                                        lm_a, lm_b, lm_c, lm_d = lm_tl_, lm_tr_, lm_br_, lm_bl_
                                    elif mode_n == 2:
                                        lm_a, lm_b, lm_c, lm_d = lm_br_, lm_bl_, lm_tl_, lm_tr_
                                    elif mode_n == 3:
                                        lm_a, lm_b, lm_c, lm_d = lm_tl_, lm_bl_, lm_br_, lm_tr_
                                    elif mode_n == 4:
                                        lm_a, lm_b, lm_c, lm_d = lm_br_, lm_tr_, lm_tl_, lm_bl_
                                    elif mode_n == 5:
                                        lm_a, lm_b, lm_c, lm_d = lm_tr_, lm_tl_, lm_bl_, lm_br_
                                    add_quad(tex_id, a, b, c, d, uv_a, uv_b, uv_c, uv_d,
                                             lm_a, lm_b, lm_c, lm_d)
                                    n_north += 1

                    # ---- EAST ----
                    se = int(sid_east[i])
                    if 0 <= se < n_surfaces and (x + 1) < max_x:
                        ni = y * max_x + (x + 1)
                        if 0 <= int(sid_top[ni]) < n_surfaces:
                            info = resolve_tex(se)
                            if info is not None:
                                tex_id, surface = info
                                ha1, ha2 = float(h_br[i]), float(h_tr[i])
                                hb1, hb2 = float(h_bl[ni]), float(h_tl[ni])
                                a = (10.0 * x + 10.0, -ha1, z_base - 10.0 * y + dz_bottom)
                                b = (10.0 * x + 10.0, -ha2, z_base - 10.0 * y + dz_top)
                                c = (10.0 * (x + 1), -hb2, z_base - 10.0 * y + dz_top)
                                d = (10.0 * (x + 1), -hb1, z_base - 10.0 * y + dz_bottom)
                                if abs(ha1 - hb1) > 0.01 or abs(ha2 - hb2) > 0.01:
                                    if tex_id != NO_TEX_KEY:
                                        uv_a = surface['uv_bottom_left']
                                        uv_b = surface['uv_bottom_right']
                                        uv_c = surface['uv_top_right']
                                        uv_d = surface['uv_top_left']
                                    else:
                                        uv_a, uv_b, uv_c, uv_d = (0, 1), (1, 1), (1, 0), (0, 0)
                                    lm_bl_, lm_br_, lm_tl_, lm_tr_ = lm_uvs(surface)
                                    lm_a, lm_b, lm_c, lm_d = lm_bl_, lm_br_, lm_tr_, lm_tl_
                                    if mode_e == 1:
                                        lm_a, lm_b, lm_c, lm_d = lm_tl_, lm_tr_, lm_br_, lm_bl_
                                    elif mode_e == 2:
                                        lm_a, lm_b, lm_c, lm_d = lm_br_, lm_bl_, lm_tl_, lm_tr_
                                    elif mode_e == 3:
                                        lm_a, lm_b, lm_c, lm_d = lm_tl_, lm_bl_, lm_br_, lm_tr_
                                    elif mode_e == 4:
                                        lm_a, lm_b, lm_c, lm_d = lm_br_, lm_tr_, lm_tl_, lm_bl_
                                    elif mode_e == 5:
                                        lm_a, lm_b, lm_c, lm_d = lm_tr_, lm_tl_, lm_bl_, lm_br_
                                    add_quad(tex_id, a, b, c, d, uv_a, uv_b, uv_c, uv_d,
                                             lm_a, lm_b, lm_c, lm_d)
                                    n_east += 1

        self.terrain_groups = {tid: np.asarray(buf, dtype=np.float32) for tid, buf in groups.items()}
        self._upload_terrain(self.terrain_groups)
        n_tris = sum(len(b) for b in self.terrain_groups.values()) // 10 // 3
        ms = (time.perf_counter() - t0) * 1000.0
        log_info(f"[3D] Terreno: {n_tris} tris | top={n_top} north={n_north} east={n_east} | "
              f"north_dir={self.north_dir:+d} flip_z={flip_z} walls={show_walls} | {ms:.0f} ms")

    def _upload_terrain(self, groups):
        # liberar VAOs viejos si se reconstruye
        if hasattr(self, 'terrain_objects'):
            for vao, vbo, _ in self.terrain_objects.values():
                try:
                    glDeleteVertexArrays(1, [vao])
                    glDeleteBuffers(1, [vbo])
                except Exception:
                    pass
        self.terrain_objects = {}
        stride = 10 * 4  # pos3 + uv2 + n3 + lm_uv2
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
            glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
            glEnableVertexAttribArray(3)
            glBindVertexArray(0)
            self.terrain_objects[tex_id] = (vao, vbo, len(buf) // 10)

    def _collect_water_zones(self):
        """Zonas de agua del mapa (una o varias).

        Prioridad igual que BrowEdit3: el agua del GND (>= 1.8) manda sobre la del RSW.
        Cada zona: dict(level, type, amp, speed, pitch, cycling, x0, x1, y0, y1), con el
        rango de cubos GND [x0,x1) x [y0,y1) que cubre. Un mapa clásico tiene 1 zona."""
        g = self.gnd
        gw, gh = g.width, g.height

        def zone(level, wtype, amp, speed, pitch, cyc, rect):
            return dict(
                level=float(level),
                type=int(wtype if wtype is not None else 0),
                amp=float(amp if amp is not None else 1.0),
                speed=float(speed if speed is not None else 2.0),
                pitch=float(pitch if pitch else 50.0),
                cycling=int(cyc if cyc else 3),
                x0=rect[0], x1=rect[1], y0=rect[2], y1=rect[3])

        zones = []
        U, V = g.water_zone_count_x, g.water_zone_count_y
        if g.water_level is not None:
            complete = (U >= 1 and V >= 1 and len(g.water_zones) == V
                        and all(len(r) == U for r in g.water_zones))
            if complete:
                for zy in range(V):
                    for zx in range(U):
                        zz = g.water_zones[zy][zx]
                        rect = g.water_zone_rect(zx, zy)
                        if rect[0] >= rect[1] or rect[2] >= rect[3]:
                            continue
                        zones.append(zone(zz['level'], zz['type'], zz['wave_height'],
                                          zz['wave_speed'], zz['wave_pitch'],
                                          zz['texture_cycling'], rect))
            else:
                zones.append(zone(g.water_level, g.water_type, g.water_wave_height,
                                  g.water_wave_speed, g.water_wave_pitch,
                                  g.water_texture_cycling, (0, gw, 0, gh)))
        elif self.rsw and self.rsw.water_level is not None:
            r = self.rsw
            zones.append(zone(r.water_level, r.water_type, r.water_wave_height,
                              r.water_wave_speed, r.water_wave_pitch,
                              r.water_texture_cycling, (0, gw, 0, gh)))
        self.water_zones = zones
        # Atributos de compatibilidad (primera zona)
        z0 = zones[0] if zones else None
        self.water_level = z0['level'] if z0 else None
        self.water_type = z0['type'] if z0 else None
        self.water_wave_height = z0['amp'] if z0 else None
        self.water_wave_speed = z0['speed'] if z0 else None
        self.water_wave_pitch = z0['pitch'] if z0 else None
        self.water_texture_cycling = z0['cycling'] if z0 else None
        self._water_hmax = None
        if len(zones) > 1:
            lv = [z['level'] for z in zones]
            log_info(f"[3D] Agua multi-zona: {len(zones)} zonas, niveles {min(lv):.1f} a {max(lv):.1f}")

    def _water_cells(self):
        """Por zona: array (n, 2) con (cx, cy) de los cubos donde se dibuja agua.

        Regla de BrowEdit3: un cubo se omite si sus 4 esquinas quedan por encima de la ola
        más baja (altura GND <= nivel - amplitud; en el GND, mayor valor = más profundo)."""
        gw, gh = self.gnd.width, self.gnd.height
        cubes = self.gnd.cubes
        if self._water_hmax is None:
            if len(cubes) == gw * gh:
                hm = np.array([max(c['h_bottom_left'], c['h_bottom_right'],
                                   c['h_top_left'], c['h_top_right']) for c in cubes],
                              dtype=np.float64).reshape(gh, gw)
            else:
                hm = np.full((gh, gw), 1e9)          # sin datos de cubos: agua en todo
            self._water_hmax = hm
        out = []
        for z in self.water_zones:
            sub = self._water_hmax[z['y0']:z['y1'], z['x0']:z['x1']]
            ys, xs = np.nonzero(sub > z['level'] - z['amp'])
            out.append(np.stack([xs + z['x0'], ys + z['y0']], axis=1))
        return out

    def _build_water_mesh(self):
        """Agua simple: un quad por cubo con agua, a la altura de SU zona."""
        self.water_vao = None
        self.water_vert_count = 0
        if not self.water_zones:
            return
        gh = self.gnd.height
        chunks = []
        for z, c in zip(self.water_zones, self._water_cells()):
            n = len(c)
            if n == 0:
                continue
            x0 = c[:, 0].astype(np.float32) * 10.0
            x1 = x0 + 10.0
            z0 = 10.0 * gh - 10.0 * c[:, 1].astype(np.float32)   # misma fila que el terreno
            z1 = z0 + 10.0
            arr = np.zeros((n, 6, 8), dtype=np.float32)
            arr[:, :, 0] = np.stack([x0, x1, x1, x0, x1, x0], 1)
            arr[:, :, 1] = -z['level']
            arr[:, :, 2] = np.stack([z0, z0, z1, z0, z1, z1], 1)
            arr[:, :, 6] = 1.0                                    # normal (0, 1, 0)
            chunks.append(arr.reshape(-1, 8))
        if not chunks:
            return
        verts = np.ascontiguousarray(np.vstack(chunks), dtype=np.float32)
        self.water_vert_count = len(verts)
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        stride = 8 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)
        self.water_vao = vao
        self.water_vbo = vbo

    # ------------------------------------------------------------------
    # Agua animada (modo 3)
    # ------------------------------------------------------------------
    def _find_water_frames(self, water_type):
        """Busca water{tipo}{NN}.jpg (NN=00..31) bajo las raíces de texturas."""
        import re
        pat = re.compile(r'^water%d(\d\d)\.(jpg|jpeg|bmp|png|tga)$' % int(water_type), re.I)
        found = {}
        cands = []
        wd = getattr(self, 'water_dir', None)
        if wd and os.path.isdir(wd):
            cands.append(os.path.abspath(wd))     # la carpeta del launcher tiene prioridad
        for root in (getattr(self, 'texture_roots', None) or []):
            for sub in ('', 'texture', os.path.join('data', 'texture'), 'data'):
                d = os.path.join(root, sub) if sub else root
                if os.path.isdir(d) and d not in cands:
                    cands.append(d)
        # Solo carpetas candidatas y 2 niveles de profundidad (no recorre todo el disco)
        for d in cands:
            base_depth = d.rstrip(os.sep).count(os.sep)
            for dp, dn, fn in os.walk(d):
                if dp.count(os.sep) - base_depth >= 2:
                    dn[:] = []
                for f in fn:
                    m = pat.match(f)
                    if m:
                        found.setdefault(int(m.group(1)), os.path.join(dp, f))
            if len(found) >= 8:
                break
        return [found[k] for k in sorted(found)]

    def _init_water_animated(self):
        self.water_anim_ok = False
        self.water_type_frames = {}
        self._water_paths = {}
        self.water_prog = None
        if not self.water_zones:
            return
        try:
            vs = shaders.compileShader(WATER_VERTEX_SHADER, GL_VERTEX_SHADER)
            fs = shaders.compileShader(WATER_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            prog = shaders.compileProgram(vs, fs)
        except Exception as e:
            print(f"[3D] Agua animada: error de shader ({e}); modo 3 usará agua simple")
            return
        self.water_prog = prog
        self.wu = {n: glGetUniformLocation(prog, n) for n in (
            'u_mvp', 'u_level_y', 'u_time', 'u_wave_h', 'u_wave_speed', 'u_wave_pitch',
            'u_tex_a', 'u_tex_b', 'u_mix', 'u_use_tex', 'u_tile', 'u_alpha')}

        # Malla: grilla en XZ (celdas de 5 unidades, 4 por cubo GND), agrupada por zona
        gh = self.gnd.height
        sub = np.array([(0, 0), (5, 0), (0, 5), (5, 5)], dtype=np.float32)
        chunks, ranges, first = [], [], 0
        for z, c in zip(self.water_zones, self._water_cells()):
            n = len(c)
            if n == 0:
                ranges.append((first, 0))
                continue
            bx = c[:, 0].astype(np.float32) * 10.0
            bz = 10.0 * gh - 10.0 * c[:, 1].astype(np.float32)
            x0 = bx[:, None] + sub[None, :, 0]
            z0 = bz[:, None] + sub[None, :, 1]
            x1, z1 = x0 + 5.0, z0 + 5.0
            px = np.stack([x0, x1, x1, x0, x1, x0], 2)
            pz = np.stack([z0, z0, z1, z0, z1, z1], 2)
            arr = np.stack([px, pz], 3).reshape(-1, 2).astype(np.float32)
            chunks.append(arr)
            ranges.append((first, len(arr)))
            first += len(arr)
        if not chunks:
            return
        tris = np.ascontiguousarray(np.vstack(chunks), dtype=np.float32)
        vao = glGenVertexArrays(1); vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, tris.nbytes, tris, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
        self.water_grid_vao, self.water_grid_vbo, self.water_grid_count = vao, vbo, len(tris)
        self.water_zone_ranges = ranges

        # Texturas de agua RO (si existen), una lista por TIPO de agua: se cargan DESPUÉS,
        # una por tanda (_water_frames_iter). Mientras tanto el modo 3 usa la ondulación procedural.
        if PIL_AVAILABLE:
            for t in sorted({z['type'] for z in self.water_zones}):
                self._water_paths[t] = self._find_water_frames(t)
        self.water_anim_ok = True

    def _water_frames_iter(self):
        """Sube a GL los frames de agua de RO (por tipo), de a uno, para no congelar la ventana."""
        for wtype, paths in sorted((getattr(self, '_water_paths', None) or {}).items()):
            if len(paths) < 2:
                log_info(f"[3D] Agua RO: sin water{int(wtype)}NN.jpg → ondulación procedural para el tipo {wtype}")
                continue
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            frames = []
            for pth in paths:
                try:
                    img = Image.open(pth).convert('RGB')
                    tid = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, tid)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0,
                                 GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
                    glGenerateMipmap(GL_TEXTURE_2D)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                    frames.append(tid)
                except Exception as e:
                    log_warn(f"[3D] Agua: no pude cargar {pth}: {e}")
                yield
            self.water_type_frames[wtype] = frames     # se activa completo, de una vez
            log_info(f"[3D] Agua RO tipo {wtype}: {len(frames)} frames de textura")

    def _set_water_mode(self, mode):
        self.water_mode = mode
        names = {1: "SIN agua", 2: "agua simple", 3: "agua animada (texturas RO)",
                 4: "agua animada procedural"}
        if not self.water_zones:
            print("[3D] Este mapa no tiene agua")
            return
        if self.water_vert_count == 0:
            print("[3D] El agua de este mapa queda por debajo del terreno: no hay nada que mostrar")
            return
        if mode in (3, 4) and not self.water_anim_ok:
            print("[3D] Agua animada no disponible; se usa agua simple")
        else:
            print(f"[3D] Agua: {names[mode]}")

    def _draw_water_animated(self, mvp, procedural=False):
        """Agua animada: cada zona con SU nivel, ola, tipo de textura y ciclo."""
        t = self._water_time_ms / 1000.0
        glUseProgram(self.water_prog)
        wu = self.wu
        glUniformMatrix4fv(wu['u_mvp'], 1, GL_FALSE, mat4.to_gl(mvp))
        glUniform1f(wu['u_time'], t)
        glUniform1f(wu['u_tile'], WATER_TEX_TILE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self.water_grid_vao)
        for z, (first, count) in zip(self.water_zones, self.water_zone_ranges):
            if count <= 0:
                continue
            glUniform1f(wu['u_level_y'], -z['level'])
            glUniform1f(wu['u_wave_h'], z['amp'])
            glUniform1f(wu['u_wave_speed'], z['speed'])
            glUniform1f(wu['u_wave_pitch'], z['pitch'])
            frames = self.water_type_frames.get(z['type']) or []
            n = len(frames)
            if n >= 2 and not procedural:
                f = (self._water_time_ms / (16.667 * max(1, z['cycling']))) % n   # frame fraccionario
                i0 = int(f) % n; i1 = (i0 + 1) % n
                glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, frames[i0])
                glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, frames[i1])
                glUniform1i(wu['u_tex_a'], 4); glUniform1i(wu['u_tex_b'], 5)
                glUniform1f(wu['u_mix'], f - int(f))
                glUniform1i(wu['u_use_tex'], 1)
                glUniform1f(wu['u_alpha'], 1.0 if z['type'] in WATER_OPAQUE_TYPES else WATER_ALPHA)
            else:
                glUniform1i(wu['u_use_tex'], 0)
                glUniform1f(wu['u_alpha'], 0.62)
            glDrawArrays(GL_TRIANGLES, first, count)
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glActiveTexture(GL_TEXTURE0)
        glUseProgram(self.program)


    def _gat_corner(self, gx, gy, corner, h):
        """Esquina GAT (celda=5). Fórmula de la versión con espejado correcto del puente."""
        gat_h = self.gat.height
        flip = getattr(self, 'flip_z', False)
        off_x = getattr(self, 'gat_off_x', 0.0)
        off_z = getattr(self, 'gat_off_z', 0.0)
        x = gx * 5.0 + (5.0 if 'right' in corner else 0.0) + off_x

        if flip:
            if 'upper' in corner:
                z = 5.0 * gy
            else:
                z = 5.0 * gy + 5.0
        else:
            if 'upper' in corner:
                z = 5.0 * (gat_h - gy)
            else:
                z = 5.0 * (gat_h - gy - 1)

        return (x, -h, z + off_z)

    def _build_walkable_overlay(self):
        """Malla GAT: alturas reales del .gat (puentes flotantes) + color caminable.

        GRF Editor dibuja esta malla 'fantasma' cuando muestra GAT types:
        la geometría caminable puede estar en el aire (puentes), no solo
        pegada al terreno GND.
        """
        self.walkable_bufs = {}
        if self.gat is None:
            return

        WALKABLE = (0, 3)
        green, red = [], []
        # Ligeramente por encima de la altura GAT para evitar z-fight con GND
        EPS = 0.0

        def add_cell(buf, gx, gy, block, fx, fy, edge_buf=None):
            h_ul = block['upper_left']
            h_ur = block['upper_right']
            h_ll = block['lower_left']
            h_lr = block['lower_right']
            # Como en la versión de referencia: reorientar alturas al espejar
            if fx:
                h_ul, h_ur = h_ur, h_ul
                h_ll, h_lr = h_lr, h_ll
            if fy:
                h_ul, h_ll = h_ll, h_ul
                h_ur, h_lr = h_lr, h_ur
            if getattr(self, 'gat_flip_heights', False):
                h_ul, h_ll = h_ll, h_ul
                h_ur, h_lr = h_lr, h_ur
            ul = self._gat_corner(gx, gy, 'upper_left', h_ul)
            ur = self._gat_corner(gx, gy, 'upper_right', h_ur)
            ll = self._gat_corner(gx, gy, 'lower_left', h_ll)
            lr = self._gat_corner(gx, gy, 'lower_right', h_lr)
            # elevar un poco
            ul = (ul[0], ul[1] + EPS, ul[2])
            ur = (ur[0], ur[1] + EPS, ur[2])
            ll = (ll[0], ll[1] + EPS, ll[2])
            lr = (lr[0], lr[1] + EPS, lr[2])
            n = (0.0, 1.0, 0.0)
            # dos triángulos (ul-ur-lr, ul-lr-ll)
            for tri in ((ul, ur, lr), (ul, lr, ll)):
                for p in tri:
                    buf.extend([p[0], p[1], p[2], 0.0, 0.0, n[0], n[1], n[2]])
            # Borde violeta (pseudo-grilla): 4 aristas, un poco más arriba
            if edge_buf is not None:
                e = 0.05  # evitar z-fight con el fill
                for a, b in ((ul, ur), (ur, lr), (lr, ll), (ll, ul)):
                    for p in (a, b):
                        edge_buf.extend([
                            p[0], p[1] + e, p[2],
                            0.0, 0.0, 0.0, 1.0, 0.0,
                        ])

        flip_x = getattr(self, 'gat_flip_x', False)
        flip_y = getattr(self, 'gat_flip_y', True)
        edges = []  # líneas violeta por celda
        gw, gh = self.gat.width, self.gat.height
        for gy in range(gh):
            for gx in range(gw):
                block = self.gat.get_block(gx, gy)
                if block is None:
                    continue
                # Espejo respecto al GND (puentes alineados)
                px = (gw - 1 - gx) if flip_x else gx
                py = (gh - 1 - gy) if flip_y else gy
                buf = green if block['type'] in WALKABLE else red
                add_cell(buf, px, py, block, flip_x, flip_y, edge_buf=edges)

        for key, buf, color in (
            ('green', green, (0.1, 1.0, 0.2, 0.55)),
            ('red', red, (1.0, 0.12, 0.08, 0.50)),
        ):
            if not buf:
                continue
            arr = np.array(buf, dtype=np.float32)
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
            stride = 8 * 4
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
            glEnableVertexAttribArray(2)
            glBindVertexArray(0)
            self.walkable_bufs[key] = (vao, vbo, len(arr) // 8, color)

        self.walkable_edge_vao = None
        self.walkable_edge_count = 0
        if edges:
            arr = np.array(edges, dtype=np.float32)
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
            stride = 8 * 4
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
            glEnableVertexAttribArray(2)
            glBindVertexArray(0)
            self.walkable_edge_vao = vao
            self.walkable_edge_vbo = vbo
            self.walkable_edge_count = len(arr) // 8

        n_g = len(green) // 24
        n_r = len(red) // 24
        # histograma de tipos GAT (para depurar escaleras)
        from collections import Counter
        type_hist = Counter()
        for gy in range(gh):
            for gx in range(gw):
                b = self.gat.get_block(gx, gy)
                if b:
                    type_hist[b['type']] += 1
        log_info(f"[3D] Overlay GAT: {n_g} cam / {n_r} block | flip_x={flip_x} flip_y={flip_y}")
        log_info(f"[3D] GAT tipos: {dict(sorted(type_hist.items()))}")



    def _init_rsm_program(self):
        self.rsm_program = None
        try:
            vs = shaders.compileShader(RSM_VERTEX_SHADER, GL_VERTEX_SHADER)
            fs = shaders.compileShader(RSM_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            self.rsm_program = shaders.compileProgram(vs, fs)
        except Exception as e:
            print(f"[3D] Shader RSM no disponible: {e}")

    def _rsm_search_roots(self):
        roots = list(getattr(self, 'texture_roots', []) or [])
        map_dir = os.path.dirname(os.path.abspath(self.gnd_path))
        extras = [
            map_dir,
            getattr(self, 'texture_dir', None),
            os.path.join(map_dir, 'model'),
            os.path.join(map_dir, 'data', 'model'),
            os.path.join(map_dir, '..', 'model'),
            os.path.join(map_dir, '..', 'data', 'model'),
        ]
        if self.texture_dir:
            extras += [
                os.path.join(self.texture_dir, 'model'),
                os.path.join(self.texture_dir, 'data', 'model'),
                os.path.join(self.texture_dir, 'data'),
            ]
        for extra in extras:
            if extra and os.path.isdir(extra):
                a = os.path.abspath(extra)
                if a not in roots:
                    roots.append(a)
        return roots

    @staticmethod
    def _name_encoding_variants(name):
        """Variantes por extracción GRF con codepage incorrecto (coreano ↔ mojibake)."""
        if not name:
            return set()
        out = {name, name.lower()}
        # Unicode correcto (euc-kr) → bytes → reinterpretado como Latin-1/CP1252 (como en el Explorer)
        for enc in ('euc-kr', 'cp949'):
            try:
                raw = name.encode(enc)
            except (UnicodeEncodeError, LookupError):
                continue
            for dec in ('latin-1', 'cp1252', 'cp437'):
                try:
                    out.add(raw.decode(dec))
                    out.add(raw.decode(dec).lower())
                except Exception:
                    pass
        # Mojibake en disco → recuperar Unicode
        for enc in ('latin-1', 'cp1252', 'cp437'):
            try:
                raw = name.encode(enc)
            except (UnicodeEncodeError, LookupError):
                continue
            for dec in ('euc-kr', 'cp949'):
                try:
                    out.add(raw.decode(dec))
                    out.add(raw.decode(dec).lower())
                except Exception:
                    pass
        return out

    def _build_rsm_index(self):
        """Índice de .rsm en disco, con claves Unicode y mojibake."""
        if getattr(self, '_rsm_index_ready', False):
            return
        self._rsm_by_base = {}  # basename variant → path
        self._rsm_by_rel = {}   # relative path variant → path
        roots = self._rsm_search_roots()
        n = 0
        for root in roots:
            try:
                for dirpath, _dirs, files in os.walk(root):
                    for f in files:
                        fl = f.lower()
                        if not (fl.endswith('.rsm') or fl.endswith('.rsm2')):
                            continue
                        full = os.path.join(dirpath, f)
                        try:
                            rel = os.path.relpath(full, root).replace('\\', '/')
                        except ValueError:
                            rel = f
                        for vb in self._name_encoding_variants(f):
                            self._rsm_by_base.setdefault(vb, full)
                            self._rsm_by_base.setdefault(vb.lower(), full)
                        for vr in self._name_encoding_variants(rel):
                            self._rsm_by_rel.setdefault(vr, full)
                            self._rsm_by_rel.setdefault(vr.lower(), full)
                            # sin data/
                            if vr.lower().startswith('data/'):
                                self._rsm_by_rel.setdefault(vr[5:], full)
                        n += 1
            except OSError as e:
                print(f"[3D] walk RSM en {root}: {e}")
        self._rsm_index_ready = True
        log_info(f"[3D] Índice RSM: {n} archivos, {len(self._rsm_by_base)} claves basename bajo {len(roots)} raíces")
        if n == 0:
            log_info(f"[3D] Raíces buscadas: {roots[:6]}")
        else:
            # muestra 3 nombres en disco
            sample = list(self._rsm_by_base.keys())[:3]
            log_info(f"[3D] Muestra claves índice: {sample}")

    def _resolve_rsm_path(self, file_name):
        if not file_name:
            return None
        self._build_rsm_index()
        rel = file_name.replace('\\', '/').strip()
        base = rel.split('/')[-1]
        parent = rel.split('/')[-2] if '/' in rel else ''

        # 1) ruta relativa con variantes de encoding
        for v in self._name_encoding_variants(rel):
            p = self._rsm_by_rel.get(v) or self._rsm_by_rel.get(v.lower())
            if p:
                return p
            if not v.lower().startswith('data/'):
                p = self._rsm_by_rel.get('data/' + v) or self._rsm_by_rel.get('data/' + v.lower())
                if p:
                    return p

        # 2) basename + carpeta padre (evita cargar otro .rsm con el mismo nombre)
        parent_vars = self._name_encoding_variants(parent) if parent else set()
        base_vars = self._name_encoding_variants(base)
        candidates = []
        for vb in base_vars:
            p = self._rsm_by_base.get(vb) or self._rsm_by_base.get(vb.lower())
            if not p:
                continue
            # _rsm_by_base solo guarda uno; buscar todos los paths con ese basename
        # reconstruir lista completa desde el índice relativo
        for rel_key, full in self._rsm_by_rel.items():
            leaf = rel_key.replace('\\', '/').split('/')[-1]
            if leaf not in base_vars and leaf.lower() not in {x.lower() for x in base_vars}:
                continue
            score = 1
            if parent_vars:
                parts = rel_key.replace('\\', '/').lower().split('/')
                for pv in parent_vars:
                    if pv.lower() in parts:
                        score += 10
                        break
            candidates.append((score, full))
        if candidates:
            candidates.sort(key=lambda t: -t[0])
            best_score, best = candidates[0]
            # si hay empate de score bajo sin carpeta, avisar
            if parent and best_score < 10:
                alts = [c for c in candidates if c[0] == best_score]
                if len(alts) > 1:
                    log_info(f"[3D] RSM ambiguo '{base}' ({len(alts)} opciones), uso: {best}")
            return best

        # 3) basename simple
        for v in base_vars:
            p = self._rsm_by_base.get(v) or self._rsm_by_base.get(v.lower())
            if p:
                return p

        # 4) prueba directa
        for root in self._rsm_search_roots():
            for cand in (
                os.path.join(root, rel.replace('/', os.sep)),
                os.path.join(root, base),
            ):
                if os.path.isfile(cand):
                    return cand
        return None

    def _apply_rsm_elong_threshold(self):
        """Legacy no-op (ancla fija root+minY; sin umbral/espejos experimentales)."""
        log_info("[3D] ancla fija root_XZ+minY (umbral/espejos Ctrl desactivados)")



    def _rebuild_rsm_placements(self):
        """Reaplica matrices de instancia sin recargar archivos .rsm."""
        if not self.rsw or not getattr(self, 'rsm_instance_sets', None):
            return
        self.rsm_placements = []
        for m in self.rsw.models:
            fn = m.get('file_name') or m.get('name')
            iset = self.rsm_instance_sets.get(fn)
            if iset is None:
                continue
            world = instance_world_matrix(
                self.gnd.width, self.gnd.height, m['pos'], m['rot'], m['scale'],
                rb_min=iset.rb_min, rb_range=iset.rb_range, rsm2=iset.is_rsm2,
                off_x=getattr(self, 'model_off_x', 0.0),
                off_z=getattr(self, 'model_off_z', 0.0),
            )
            # Punto de anclaje (para culling/distancia): posición RSW en el mundo
            self.rsm_placements.append((iset, world, self._rsw_to_world(*m['pos']),
                                        float(m.get('anim_speed') or 1.0)))
        log_info(f"[3D] RSM instancias={len(self.rsm_placements)}")

    def _load_rsm_instances(self):
        """Prepara la carga PROGRESIVA: el terreno se ve enseguida y los modelos RSM
        aparecen a medida que se cargan (un modelo por tanda, ver _pump_loading)."""
        self.rsm_instance_sets = {}
        self.rsm_placements = []
        ModelInstanceSet._missing_tex_all = set()
        self._rsm_loader = self._load_all_iter()

    def _load_all_iter(self):
        yield                                   # primero que se dibuje el terreno (sin texturas)
        yield from self._gnd_tex_iter()         # texturas del suelo (rápido, sin mipmap)
        yield from self._rsm_load_iter()        # modelos 3D
        yield from self._water_frames_iter()    # frames de agua animada
        yield from self._mipmap_iter()          # mipmaps diferidos (GND + RSM)

    def _rsm_load_iter(self):
        if not self.rsw or not getattr(self.rsw, 'models', None):
            log_warn("[3D] Sin modelos RSW — no se cargan RSM (¿pasaste el .rsw?)")
            return
        if not self.rsm_program:
            log_warn("[3D] Shader RSM no compiló — no se cargan modelos")
            return

        needed = {}
        by_file = {}
        for m in self.rsw.models:
            fn = m.get('file_name') or m.get('name')
            if not fn:
                continue
            by_file.setdefault(fn, []).append(m)
            if fn not in needed:
                needed[fn] = self._resolve_rsm_path(fn)
        yield

        missing = [fn for fn, p in needed.items() if not p]
        if missing:
            log_warn(f"[3D] Aviso: {len(missing)} modelo(s) RSM no encontrado(s), ej: {missing[:3]}")

        # Orden de carga: primero los archivos que "rinden más", o sea los que tienen más
        # instancias cerca de la cámara inicial. Cada archivo se carga UNA sola vez y sirve
        # a todas sus instancias, así que el mapa se va llenando lo más rápido posible.
        tgt = self.cam_target
        near_r2 = 450.0 ** 2
        def _key(fn):
            near, best = 0, 1e18
            for m in by_file[fn]:
                p = self._rsw_to_world(*m['pos'])
                d2 = (p[0] - tgt[0]) ** 2 + (p[2] - tgt[2]) ** 2
                best = min(best, d2)
                if d2 <= near_r2:
                    near += 1
            return (-near, best)
        order = sorted((fn for fn, p in needed.items() if p), key=_key)

        loaded = matched = 0
        for fn in order:
            path = needed[fn]
            try:
                iset = ModelInstanceSet(path, self.texture_roots)
            except Exception as e:
                log_warn(f"[3D] RSM omitido {fn}: {e}")
                yield
                continue
            self.rsm_instance_sets[fn] = iset
            loaded += 1
            for m in by_file[fn]:
                world = instance_world_matrix(
                    self.gnd.width, self.gnd.height, m['pos'], m['rot'], m['scale'],
                    rb_min=iset.rb_min, rb_range=iset.rb_range, rsm2=iset.is_rsm2,
                    off_x=getattr(self, 'model_off_x', 0.0),
                    off_z=getattr(self, 'model_off_z', 0.0),
                )
                self.rsm_placements.append((iset, world, self._rsw_to_world(*m['pos']),
                                            float(m.get('anim_speed') or 1.0)))
                matched += 1
            yield

        if ModelInstanceSet._missing_tex_all:
            miss = sorted(ModelInstanceSet._missing_tex_all)
            log_warn(f"[3D] Aviso: faltan {len(miss)} textura(s) de modelos, ej: {miss[:3]}")
        log_info(f"[3D] RSM listos: {loaded} archivos, {matched}/{len(self.rsw.models)} instancias")
        if VERBOSE:
            self._log_rsm_diag()

    def _log_rsm_diag(self):
        n_ok = n_bad = 0
        issue_hist = {}
        bad_samples = []
        for fn, iset in (self.rsm_instance_sets or {}).items():
            d = getattr(iset, 'hierarchy_diag', None)
            if not d:
                continue
            if d.get('has_issues'):
                n_bad += 1
                for iss in d.get('issues') or []:
                    issue_hist[iss] = issue_hist.get(iss, 0) + 1
                if len(bad_samples) < 40:
                    bad_samples.append((fn, d))
            else:
                n_ok += 1
        log_info(f"[RSM-diag] resumen: OK={n_ok}  con_issues={n_bad}  total={n_ok+n_bad}")
        if issue_hist:
            log_info(f"[RSM-diag] issues: {dict(sorted(issue_hist.items(), key=lambda x: -x[1]))}")
        # Detalle de los problemáticos (y unos OK de muestra si no hay issues)
        to_print = bad_samples
        if not to_print:
            for fn, iset in list((self.rsm_instance_sets or {}).items())[:5]:
                d = getattr(iset, 'hierarchy_diag', None)
                if d:
                    to_print.append((fn, d))
        for fn, d in to_print:
            label = fn if isinstance(fn, str) else str(fn)
            # basename corto para legibilidad
            try:
                import os as _os
                label = _os.path.basename(label.replace('\\', '/').split('/')[-1])
            except Exception:
                pass
            log_info(format_rsm_diagnosis(d, label))
        log_info("[RSM-diag] fin log (hierarchy)")

    def _pump_loading(self, budget_ms=12.0):
        """Avanza la carga progresiva durante ~budget_ms (llamar una vez por frame)."""
        gen = getattr(self, '_rsm_loader', None)
        if gen is None:
            return
        t0 = time.perf_counter()
        try:
            while True:
                next(gen)
                if (time.perf_counter() - t0) * 1000.0 >= budget_ms:
                    break
        except StopIteration:
            self._rsm_loader = None
        except Exception as e:
            self._rsm_loader = None
            log_warn(f"[3D] Error cargando modelos: {type(e).__name__}: {e}")

    def finish_loading(self):
        """Completa la carga de golpe (capturas de pantalla, pruebas)."""
        while getattr(self, '_rsm_loader', None) is not None:
            self._pump_loading(budget_ms=1e9)

    def _rsw_to_world(self, x_3d, y_3d, z_3d):
        """Traslación alineada al terreno (+ model_off fine-tune)."""
        gw = self.gnd.width
        gh = self.gnd.height
        wx = 5.0 * gw + float(x_3d) + getattr(self, 'model_off_x', 0.0)
        wy = -float(y_3d)
        wz = 5.0 * gh - float(z_3d) + 10.0 + getattr(self, 'model_off_z', 0.0)
        return (wx, wy, wz)

    def _build_model_markers(self):
        """Puntos/cruz en posiciones RSW (debug de orientación antes de cargar RSM)."""
        self.model_marker_vao = None
        self.model_marker_count = 0
        if not self.rsw or not getattr(self.rsw, 'models', None):
            log_info("[3D] Sin modelos RSW para marcadores")
            return
        # Cada marcador: cruz 3 ejes (6 líneas = 12 vértices)
        half = 3.0
        verts = []
        n = 0
        for m in self.rsw.models:
            fn = m.get('file_name') or m.get('name')
            wx, wy, wz = self._rsw_to_world(*m['pos'])
            for a, b in (
                ((wx - half, wy, wz), (wx + half, wy, wz)),
                ((wx, wy - half, wz), (wx, wy + half, wz)),
                ((wx, wy, wz - half), (wx, wy, wz + half)),
            ):
                for p in (a, b):
                    verts.extend([p[0], p[1], p[2], 0.0, 0.0, 0.0, 1.0, 0.0])
            n += 1
        if not verts:
            return
        arr = np.array(verts, dtype=np.float32)
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
        stride = 8 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)
        self.model_marker_vao = vao
        self.model_marker_vbo = vbo
        self.model_marker_count = len(arr) // 8
        log_info(f"[3D] Marcadores RSM: {n} modelos")

    def _init_compass_geometry(self):
        """Ejes E/O/N/S/U/D (solo líneas, sin letras) en el centro del mapa. C = mostrar/ocultar."""
        cx = self.gnd.width * 5.0
        cz = self.gnd.height * 5.0
        cy = 5.0
        L = max(self.gnd.width, self.gnd.height) * 1.5
        Lu = L * 0.4
        # Convención de este visor (GND):
        #   +X = Este (E)   -X = Oeste (O)
        #   -Z = Norte (N)  +Z = Sur (S)   [y mapa+]
        #   +Y = Up (U)     -Y = Down (D)
        verts = np.array([
            # E (+X) rojo
            cx, cy, cz, 1, 0, 0, 1, 0,
            cx + L, cy, cz, 1, 0, 0, 1, 0,
            # O (-X) rojo oscuro
            cx, cy, cz, 1, 0, 0, 1, 0,
            cx - L, cy, cz, 1, 0, 0, 1, 0,
            # N (-Z) verde
            cx, cy, cz, 0, 1, 0, 1, 0,
            cx, cy, cz - L, 0, 1, 0, 1, 0,
            # S (+Z) verde oscuro
            cx, cy, cz, 0, 1, 0, 1, 0,
            cx, cy, cz + L, 0, 1, 0, 1, 0,
            # U (+Y) azul
            cx, cy, cz, 0, 0, 1, 1, 0,
            cx, cy + Lu, cz, 0, 0, 1, 1, 0,
            # D (-Y) azul oscuro
            cx, cy, cz, 0, 0, 1, 1, 0,
            cx, cy - Lu, cz, 0, 0, 1, 1, 0,
        ], dtype=np.float32)
        self.compass_vao = glGenVertexArrays(1)
        self.compass_vbo = glGenBuffers(1)
        glBindVertexArray(self.compass_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.compass_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        stride = 8 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)
        self.compass_colors = [
            (1.0, 0.2, 0.2, 1.0),  # E
            (0.6, 0.25, 0.25, 1.0),  # O
            (0.2, 1.0, 0.3, 1.0),  # N
            (0.25, 0.6, 0.3, 1.0),  # S
            (0.35, 0.55, 1.0, 1.0),  # U
            (0.25, 0.35, 0.7, 1.0),  # D
        ]

    def _find_texture_file(self, tex_name):
        return resolve_ro_texture_path(
            tex_name, self.texture_roots, self._tex_basename_index
        )

    def _prepare_texture_loading(self):
        """Prepara carga diferida de texturas GND (sin bloquear el primer frame).

        El terreno se dibuja de inmediato con color plano; las texturas se
        suben de a una en _gnd_tex_iter (vía _pump_loading). Los mipmaps se
        generan después en _mipmap_iter.
        """
        self.gl_textures = {}
        self._mipmap_pending = []
        ModelInstanceSet._mipmap_queue = self._mipmap_pending
        n = len(getattr(self.gnd, 'textures', None) or [])
        self._gnd_tex_queue = list(range(n))
        self._gnd_tex_found = 0
        self._gnd_tex_missing = []
        if not PIL_AVAILABLE:
            print("[3D] Pillow no disponible.")
            self._gnd_tex_queue = []
        log_info(f"[3D] Texturas GND diferidas: {n} pendientes (se cargan en background)")

    def _upload_gnd_texture(self, tex_index):
        """Carga una textura GND a GL sin mipmaps. Devuelve gl_id o None."""
        tex_name = self.gnd.textures[tex_index]
        path = self._find_texture_file(tex_name)
        if not path:
            self._gnd_tex_missing.append(tex_name)
            self.gl_textures[tex_index] = None
            return None
        try:
            img = Image.open(path).convert('RGBA')
            img = apply_magenta_colorkey(img)
            gl_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, gl_tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
            self.gl_textures[tex_index] = gl_tex
            self._mipmap_pending.append(gl_tex)
            self._gnd_tex_found += 1
            return gl_tex
        except Exception as e:
            print(f"[3D] Error cargando {path}: {e}")
            self.gl_textures[tex_index] = None
            return None

    def _gnd_tex_iter(self):
        """Carga texturas del terreno de a una (no congela la ventana)."""
        queue = getattr(self, '_gnd_tex_queue', None) or []
        total = len(queue)
        for i in queue:
            self._upload_gnd_texture(i)
            yield
        if total:
            log_info(f"[3D] Texturas GND: {self._gnd_tex_found}/{total}")
            if self._gnd_tex_missing:
                print(f"[3D] Faltan {len(self._gnd_tex_missing)}, ej: {self._gnd_tex_missing[:3]}")
        self._gnd_tex_queue = []

    def _mipmap_iter(self):
        """Genera mipmaps diferidos (GND + RSM) de a uno por tanda."""
        pending = getattr(self, '_mipmap_pending', None)
        if not pending:
            return
        n = 0
        while pending:
            tid = pending.pop(0)
            try:
                glBindTexture(GL_TEXTURE_2D, tid)
                glGenerateMipmap(GL_TEXTURE_2D)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            except Exception:
                pass
            n += 1
            yield
        log_info(f"[3D] Mipmaps generados: {n}")

    def _camera_position(self):
        yaw = math.radians(self.cam_yaw)
        pitch = math.radians(self.cam_pitch)
        x = self.cam_dist * math.cos(pitch) * math.sin(yaw)
        y = self.cam_dist * math.sin(pitch)
        z = self.cam_dist * math.cos(pitch) * math.cos(yaw)
        return (x + self.cam_target[0], y + self.cam_target[1], z + self.cam_target[2])

    def render_frame(self):
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Estado estable: sin mezcla de transparencias para terreno y agua simple (antes lo
        # dejaba así, sin querer, el dibujo de las letras de la brújula al final del frame)
        glDisable(GL_BLEND)
        glUseProgram(self.program)

        proj = mat4.perspective(55.0, self.width / self.height, 0.5, self.model_radius * 20)
        eye = self._camera_position()
        view = mat4.look_at(eye, self.cam_target, (0, 1, 0))
        mvp = proj @ view
        cam_mvp = mvp   # el bucle RSM reutiliza el nombre 'mvp'
        glUniformMatrix4fv(self.u_mvp, 1, GL_FALSE, mat4.to_gl(mvp))

        client_light = bool(getattr(self, 'use_rsw_light', True))
        light_dir = rsw_light_direction(self.rsw, client_light)
        glUniform3f(self.u_light_dir, *light_dir)
        glUniform1i(self.u_client_light, 1 if client_light else 0)
        amb_c, dif_c, intensity = scene_light_params(
            self.rsw, getattr(self, 'use_rsw_light', True))
        glUniform3f(self.u_diffuse, *dif_c)
        glUniform3f(self.u_ambient, *amb_c)
        glUniform1f(self.u_intensity, intensity)

        use_lm = bool(getattr(self, 'show_shadows', True) and getattr(self, 'lm_atlas_tex', None))
        glUniform1i(self.u_use_lightmap, 1 if use_lm else 0)
        glUniform1i(self.u_lm_invert_ao, 1 if getattr(self, 'lm_invert_ao', False) else 0)
        if use_lm:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.lm_atlas_tex)
            glUniform1i(self.u_lightmap, 1)

        for tex_id, (vao, vbo, count) in self.terrain_objects.items():
            gl_tex = self.gl_textures.get(tex_id)
            if gl_tex is not None:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, gl_tex)
                glUniform1i(self.u_texture, 0)
                glUniform1i(self.u_use_texture, 1)
            else:
                glUniform1i(self.u_use_texture, 0)
                glUniform4f(self.u_flat_color, 0.72, 0.70, 0.65, 1.0)
            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, count)
            glBindVertexArray(0)

        wmode = getattr(self, 'water_mode', 2)
        if wmode in (3, 4) and not getattr(self, 'water_anim_ok', False):
            wmode = 2           # sin shader de agua → cae a simple
        self._water_time_ms = getattr(self, '_water_time_ms', 0.0) + getattr(self, '_frame_dt_ms', 16.0)
        if wmode == 2 and self.water_vao is not None:
            glUniform1i(self.u_use_texture, 0)
            glUniform1i(self.u_use_lightmap, 0)
            glUniform4f(self.u_flat_color, 0.15, 0.35, 0.65, 0.55)
            glBindVertexArray(self.water_vao)
            glDrawArrays(GL_TRIANGLES, 0, self.water_vert_count)
            glBindVertexArray(0)

        # Animación RSM (A) — actualizar ~15 fps, no cada frame
        if getattr(self, 'rsm_animate', False) and getattr(self, 'rsm_instance_sets', None):
            dt = getattr(self, '_frame_dt_ms', 16.0)
            self._rsm_anim_clock = getattr(self, '_rsm_anim_clock', 0.0) + dt
            self._rsm_anim_acc = getattr(self, '_rsm_anim_acc', 0.0) + dt
            if self._rsm_anim_acc >= 66.0:  # ~15 Hz
                self._rsm_anim_acc = 0.0
                for iset in self.rsm_instance_sets.values():
                    if not getattr(iset, 'has_anim', False):
                        continue
                    alen = max(float(iset.anim_len), 1.0)
                    iset.set_anim_time(self._rsm_anim_clock % alen)

        if getattr(self, 'show_rsm', True) and getattr(self, 'rsm_placements', None) and self.rsm_program:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUseProgram(self.rsm_program)
            # cachear locations una sola vez
            cache = getattr(self, '_rsm_uniform_cache', None)
            if cache is None or cache.get('prog') is not self.rsm_program:
                cache = {
                    'prog': self.rsm_program,
                    'u_mvp': glGetUniformLocation(self.rsm_program, "u_mvp"),
                    'u_model': glGetUniformLocation(self.rsm_program, "u_model"),
                    'u_tex': glGetUniformLocation(self.rsm_program, "u_texture"),
                    'u_use': glGetUniformLocation(self.rsm_program, "u_use_texture"),
                    'u_light': glGetUniformLocation(self.rsm_program, "u_light_dir"),
                    'u_diff': glGetUniformLocation(self.rsm_program, "u_diffuse_color"),
                    'u_amb': glGetUniformLocation(self.rsm_program, "u_ambient_color"),
                    'u_int': glGetUniformLocation(self.rsm_program, "u_intensity"),
                    'u_client': glGetUniformLocation(self.rsm_program, "u_client_light"),
                }
                self._rsm_uniform_cache = cache
            u_mvp, u_model = cache['u_mvp'], cache['u_model']
            u_tex, u_use = cache['u_tex'], cache['u_use']
            _client = bool(getattr(self, 'use_rsw_light', True))
            if cache['u_light'] >= 0:
                glUniform3f(cache['u_light'], *rsw_light_direction(self.rsw, _client))
            if cache['u_client'] >= 0:
                glUniform1i(cache['u_client'], 1 if _client else 0)
            m_amb, m_dif, m_int = scene_light_params(
                self.rsw, getattr(self, 'use_rsw_light', True), for_models=True)
            if cache['u_diff'] >= 0:
                glUniform3f(cache['u_diff'], *m_dif)
            if cache['u_amb'] >= 0:
                glUniform3f(cache['u_amb'], *m_amb)
            if cache['u_int'] >= 0:
                glUniform1f(cache['u_int'], m_int)

            # Culling: distancia + cono frontal ampliado (precarga al girar)
            # FOV real ~55°; usamos ~200° (casi hemisferio delantero + margen)
            # para que al orbitar rápido no "aparezcan" modelos de golpe.
            eye = self._camera_position()
            ex, ey, ez = eye[0], eye[1], eye[2]
            tgt = self.cam_target
            fx = float(tgt[0]) - ex
            fy = float(tgt[1]) - ey
            fz = float(tgt[2]) - ez
            flen = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
            fx, fy, fz = fx / flen, fy / flen, fz / flen
            max_d = float(getattr(self, 'rsm_draw_dist', 1400.0))
            max_d2 = max_d * max_d
            # cos del semiángulo del cono (FOV total / 2)
            half_fov = math.radians(float(getattr(self, 'rsm_cull_fov', 100.0)) * 0.5)
            cos_lim = math.cos(half_fov)  # ángulo con forward debe ser <= half_fov
            view_proj = proj @ view
            last_tex = object()  # sentinel distinto de None e int
            drawn = 0
            deferred = []      # mallas con alfa parcial / animación de textura (2ª pasada)
            tex_ms = getattr(self, '_water_time_ms', 0.0)
            for item in self.rsm_placements:
                inst_speed = item[3] if len(item) > 3 else 1.0
                if len(item) >= 3:
                    iset, world, pos = item[0], item[1], item[2]
                else:
                    iset, world = item[0], item[1]
                    pos = (float(world[0, 3]), float(world[1, 3]), float(world[2, 3]))
                dx = pos[0] - ex
                dy = pos[1] - ey
                dz = pos[2] - ez
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > max_d2:
                    continue
                # delante del cono ampliado (dot(forward, dir) >= cos_lim)
                inv_d = 1.0 / math.sqrt(d2) if d2 > 1e-8 else 1.0
                if (dx * fx + dy * fy + dz * fz) * inv_d < cos_lim:
                    continue
                drawn += 1
                for vao, vbo, count, tex_id, mesh_name in iset.gl_objects:
                    mesh_matrix = iset.world_matrices.get(mesh_name)
                    if mesh_matrix is None:
                        continue
                    model_matrix = world @ mesh_matrix
                    if (mesh_name, tex_id) in iset.tex_anims or tex_id in iset.translucent_tex:
                        _w = model_matrix @ iset.mesh_center(mesh_name)
                        _dd = (_w[0] - ex) ** 2 + (_w[1] - ey) ** 2 + (_w[2] - ez) ** 2
                        deferred.append((_dd, iset, vao, count, tex_id, mesh_name,
                                         model_matrix, inst_speed))
                        continue
                    mvp = view_proj @ model_matrix
                    glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mat4.to_gl(mvp))
                    if u_model >= 0:
                        glUniformMatrix4fv(u_model, 1, GL_FALSE, mat4.to_gl(model_matrix))
                    gl_tex = iset.gl_textures.get(tex_id)
                    if gl_tex is not last_tex:
                        last_tex = gl_tex
                        if gl_tex is not None:
                            glActiveTexture(GL_TEXTURE0)
                            glBindTexture(GL_TEXTURE_2D, gl_tex)
                            if u_tex >= 0:
                                glUniform1i(u_tex, 0)
                            if u_use >= 0:
                                glUniform1i(u_use, 1)
                        else:
                            if u_use >= 0:
                                glUniform1i(u_use, 0)
                    glBindVertexArray(vao)
                    glDrawArrays(GL_TRIANGLES, 0, count)
            # 2ª pasada: texturas con alfa parcial y/o animadas (ondas, agua, brillos),
            # de lejos a cerca y sin escribir profundidad
            if deferred:
                u_uv = glGetUniformLocation(self.rsm_program, "u_uv_offset")
                deferred.sort(key=lambda d: -d[0])
                glDepthMask(GL_FALSE)
                for d2_, iset, vao, count, tex_id, mesh_name, model_matrix, spd in deferred:
                    glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mat4.to_gl(view_proj @ model_matrix))
                    if u_model >= 0:
                        glUniformMatrix4fv(u_model, 1, GL_FALSE, mat4.to_gl(model_matrix))
                    gl_tex = iset.gl_textures.get(tex_id)
                    if gl_tex is not None:
                        glActiveTexture(GL_TEXTURE0)
                        glBindTexture(GL_TEXTURE_2D, gl_tex)
                        if u_tex >= 0:
                            glUniform1i(u_tex, 0)
                        if u_use >= 0:
                            glUniform1i(u_use, 1)
                    elif u_use >= 0:
                        glUniform1i(u_use, 0)
                    anims = iset.tex_anims.get((mesh_name, tex_id))
                    if anims and u_uv >= 0:
                        frame = (tex_ms / 1000.0) * float(iset.rsm.fps) * spd
                        du, dv, _unsup = RP.texture_offsets(anims, frame, iset.rsm.anim_len)
                        glUniform2f(u_uv, du, dv)
                    glBindVertexArray(vao)
                    glDrawArrays(GL_TRIANGLES, 0, count)
                    if anims and u_uv >= 0:
                        glUniform2f(u_uv, 0.0, 0.0)
                glDepthMask(GL_TRUE)
            glBindVertexArray(0)
            glUseProgram(self.program)
            self._rsm_drawn_last = drawn

        # Agua animada: se dibuja DESPUÉS de los modelos, así lo sumergido se ve bajo el agua
        if wmode in (3, 4) and self.water_zones:
            self._draw_water_animated(cam_mvp, procedural=(wmode == 4))

        if getattr(self, 'show_walkable', False) and getattr(self, 'walkable_bufs', None):
            # Con depth test: las murallas ocultan el overlay (sin visión de rayos X)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-3.0, -3.0)
            glUniform1i(self.u_use_texture, 0)
            glUniform1i(self.u_use_lightmap, 0)
            for key, (vao, vbo, count, color) in self.walkable_bufs.items():
                glUniform4f(self.u_flat_color, *color)
                glBindVertexArray(vao)
                glDrawArrays(GL_TRIANGLES, 0, count)
                glBindVertexArray(0)
            glDisable(GL_POLYGON_OFFSET_FILL)
            # Pseudo-grilla violeta en bordes de cada tile GAT (~2px)
            if getattr(self, 'walkable_edge_vao', None) and self.walkable_edge_count:
                glUniform4f(self.u_flat_color, 0.72, 0.35, 0.95, 0.85)
                try:
                    glLineWidth(2.0)
                except Exception:
                    pass
                glBindVertexArray(self.walkable_edge_vao)
                glDrawArrays(GL_LINES, 0, self.walkable_edge_count)
                glBindVertexArray(0)

        if getattr(self, 'show_models', False) and getattr(self, 'model_marker_vao', None):
            glUniform1i(self.u_use_texture, 0)
            glUniform1i(self.u_use_lightmap, 0)
            glUniform4f(self.u_flat_color, 1.0, 0.85, 0.2, 1.0)
            glDisable(GL_DEPTH_TEST)
            glBindVertexArray(self.model_marker_vao)
            glDrawArrays(GL_LINES, 0, self.model_marker_count)
            glBindVertexArray(0)
            glEnable(GL_DEPTH_TEST)

        if self.show_compass and hasattr(self, 'compass_vao'):
            glDisable(GL_DEPTH_TEST)
            glUniform1i(self.u_use_texture, 0)
            glUniform1i(self.u_use_lightmap, 0)
            glBindVertexArray(self.compass_vao)
            for i, col in enumerate(self.compass_colors):
                glUniform4f(self.u_flat_color, *col)
                glDrawArrays(GL_LINES, i * 2, 2)
            glBindVertexArray(0)
            glEnable(GL_DEPTH_TEST)

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        panning = False
        orbiting = False
        last_pos = (0, 0)
        running = True
        font = pygame.font.SysFont("consolas", 16)
        log_info("[3D] Mouse izq=pan der=orbit | G caminable M marcadores R RSM A anim S sombras")
        log_info("[3D] Nudge RSM: J/L = X±1  I/K = Z±1  (Shift=±5)  Home=reset | { } draw_dist | ESC")
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Ajustar viewport al nuevo tamaño de ventana
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
                    glViewport(0, 0, self.width, self.height)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c and not (event.mod & pygame.KMOD_SHIFT):
                        self.show_compass = not self.show_compass
                        print(f"[3D] brújula={'ON' if self.show_compass else 'off'}")
                    elif event.key == pygame.K_g:
                        self.show_walkable = not self.show_walkable
                        print(f"[3D] walkable overlay={'ON' if self.show_walkable else 'off'}")
                    elif event.key == pygame.K_m:
                        self.show_models = not self.show_models
                        print(f"[3D] model markers={'ON' if self.show_models else 'off'}")
                    elif event.key == pygame.K_r:
                        self.show_rsm = not self.show_rsm
                        print(f"[3D] RSM models={'ON' if self.show_rsm else 'off'}")
                    elif event.key == pygame.K_a:
                        self.rsm_animate = not getattr(self, 'rsm_animate', False)
                        if not self.rsm_animate:
                            for iset in (getattr(self, 'rsm_instance_sets', {}) or {}).values():
                                if getattr(iset, 'has_anim', False):
                                    iset.set_anim_time(0.0)
                            self._rsm_anim_clock = 0.0
                        _sets = list((getattr(self, 'rsm_instance_sets', None) or {}).values())
                        _na = sum(1 for x in _sets if getattr(x, 'has_anim', False))
                        _n2 = sum(1 for x in _sets if getattr(x, 'has_anim', False) and getattr(x, 'is_rsm2', False))
                        if self.rsm_animate:
                            _msg = f"{_na} modelo(s) con animación de malla ({_n2} RSM2)" if _na \
                                else "ningún modelo tiene animación de malla (las texturas animadas van siempre)"
                            print(f"[3D] Animación de modelos: ON — {_msg}")
                        else:
                            print("[3D] Animación de modelos: off")
                    elif event.key == pygame.K_s:
                        self.show_shadows = not self.show_shadows
                        print(f"[3D] lightmaps GND={'ON' if self.show_shadows else 'off'}")
                    elif event.key in (pygame.K_1, pygame.K_KP1):
                        self._set_water_mode(1)
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        self._set_water_mode(2)
                    elif event.key in (pygame.K_3, pygame.K_KP3):
                        self._set_water_mode(3)
                    elif event.key in (pygame.K_4, pygame.K_KP4):
                        self._set_water_mode(4)
                    elif event.key == pygame.K_w:
                        self.show_walls = not self.show_walls
                        print(f"[3D] walls={self.show_walls} — reconstruyendo...")
                        self._build_terrain_mesh()
                        self._build_walkable_overlay()
                    # --- Nudge posición RSM (XZ) ---
                    elif event.key == pygame.K_j:
                        step = 5.0 if (event.mod & pygame.KMOD_SHIFT) else 1.0
                        self.model_off_x -= step
                        print(f"[3D] model_off xz=({self.model_off_x:.1f}, {self.model_off_z:.1f})")
                        self._rebuild_rsm_placements()
                        self._build_model_markers()
                    elif event.key == pygame.K_l:
                        step = 5.0 if (event.mod & pygame.KMOD_SHIFT) else 1.0
                        self.model_off_x += step
                        print(f"[3D] model_off xz=({self.model_off_x:.1f}, {self.model_off_z:.1f})")
                        self._rebuild_rsm_placements()
                        self._build_model_markers()
                    elif event.key == pygame.K_i:
                        step = 5.0 if (event.mod & pygame.KMOD_SHIFT) else 1.0
                        self.model_off_z -= step
                        print(f"[3D] model_off xz=({self.model_off_x:.1f}, {self.model_off_z:.1f})")
                        self._rebuild_rsm_placements()
                        self._build_model_markers()
                    elif event.key == pygame.K_k:
                        step = 5.0 if (event.mod & pygame.KMOD_SHIFT) else 1.0
                        self.model_off_z += step
                        print(f"[3D] model_off xz=({self.model_off_x:.1f}, {self.model_off_z:.1f})")
                        self._rebuild_rsm_placements()
                        self._build_model_markers()
                    elif event.key == pygame.K_HOME:
                        self.model_off_x = 0.0
                        self.model_off_z = 0.0
                        print("[3D] model_off RESET (0, 0)")
                        self._rebuild_rsm_placements()
                        self._build_model_markers()
                    # --- Draw distance
                    # --- Draw distance RSM ---
                    elif event.unicode in ('{', '[') or event.key == pygame.K_LEFTBRACKET:
                        self.rsm_draw_dist = max(80.0, getattr(self, 'rsm_draw_dist', 1400.0) - 50.0)
                        print(f"[3D] RSM draw_dist={self.rsm_draw_dist:.0f}")
                    elif event.unicode in ('}', ']') or event.key == pygame.K_RIGHTBRACKET:
                        self.rsm_draw_dist = min(2000.0, getattr(self, 'rsm_draw_dist', 1400.0) + 50.0)
                        print(f"[3D] RSM draw_dist={self.rsm_draw_dist:.0f}")
                    # --- Terreno (dejar por si hace falta) ---
                    elif event.key == pygame.K_f and not (event.mod & pygame.KMOD_SHIFT):
                        self.north_dir = -self.north_dir
                        print(f"[3D] north_dir={self.north_dir:+d} — reconstruyendo...")
                        self._build_terrain_mesh()
                        self._build_walkable_overlay()
                    elif event.key == pygame.K_t:
                        self.use_rsw_light = not self.use_rsw_light
                        print(f"[3D] modo BrowEdit3 (luz + lightmap UV)={self.use_rsw_light} — reconstruyendo...")
                        self._build_terrain_mesh()
                    elif event.key == pygame.K_n:
                        self.smooth_normals = not self.smooth_normals
                        print(f"[3D] normales suaves={self.smooth_normals} — reconstruyendo...")
                        self._build_terrain_mesh()
                    elif event.key == pygame.K_z:
                        self.flip_z = not self.flip_z
                        print(f"[3D] flip_z={self.flip_z} — reconstruyendo...")
                        self._build_terrain_mesh()
                        self._build_walkable_overlay()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # izquierdo = pan
                        panning = True
                        last_pos = event.pos
                    elif event.button == 3:  # derecho = orbitar
                        orbiting = True
                        last_pos = event.pos
                    elif event.button == 4:  # zoom in
                        self.cam_dist = max(self.model_radius * 0.05, self.cam_dist * 0.9)
                    elif event.button == 5:  # zoom out (tope 1500)
                        self.cam_dist = min(1500.0, self.cam_dist * 1.1)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        panning = False
                    elif event.button == 3:
                        orbiting = False
                elif event.type == pygame.MOUSEMOTION:
                    dx = event.pos[0] - last_pos[0]
                    dy = event.pos[1] - last_pos[1]
                    if orbiting:
                        self.cam_yaw += dx * 0.4
                        self.cam_pitch = max(-89, min(89, self.cam_pitch - dy * 0.4))
                        last_pos = event.pos
                    elif panning:
                        # Desplazar target en el plano XZ según yaw de cámara
                        yaw = math.radians(self.cam_yaw)
                        scale = self.cam_dist * 0.0015
                        # derecha de la cámara en XZ
                        rx, rz = math.cos(yaw), -math.sin(yaw)
                        # adelante horizontal (hacia donde mira, proyectado)
                        fx, fz = -math.sin(yaw), -math.cos(yaw)
                        tx, ty, tz = self.cam_target
                        # dx>0 → mover escena a la derecha del view = target a la izquierda
                        tx += (-dx * rx + dy * fx) * scale
                        tz += (-dx * rz + dy * fz) * scale
                        self.cam_target = (tx, ty, tz)
                        last_pos = event.pos
            self.render_frame()
            self._pump_loading()
            self._frame_dt_ms = float(clock.tick(60))
        pygame.quit()

    def save_screenshot(self, path):
        self.finish_loading()
        self.render_frame()
        buf = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        img = Image.frombytes('RGB', (self.width, self.height), buf)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(path)
        print(f"[3D] Captura guardada en {path}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python terrain_viewer_3d.py <mapa.gnd> [mapa.rsw] [carpeta_texturas]")
        sys.exit(1)
    gnd_path = sys.argv[1]
    rsw_path = sys.argv[2] if len(sys.argv) > 2 else None
    tex_dir = sys.argv[3] if len(sys.argv) > 3 else None
    viewer = TerrainViewer3D(gnd_path, rsw_path=rsw_path, texture_dir=tex_dir)
    viewer.run()


if __name__ == "__main__":
    main()
