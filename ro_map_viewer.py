#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ragnarok Online Map Viewer — Versión Completa
RO_MAP_VIEWER_BUILD = 2026-03-18-silhouette
Uso: python ro_map_viewer.py <mapa.gat> [mapa.rsw] [mapa.gnd]

Soporta:
  - GAT: terreno + tipos de bloque (caminable, no caminable, agua, acantilado)
  - RSW: modelos 3D (posición/nombre), nivel de agua (versiones antiguas)
  - GND: texturas del suelo + nivel de agua (versiones 2.6+)
  - Renderizado combinado: texturas GND + overlay de colores GAT
  - Modelos 3D: indicadores visuales en el mapa
  - Controles: WASD/arrastrar, zoom, click derecho para centrar
"""

import os
import sys
import struct
import math
import io
from rolog import info as log_info, warn as log_warn   # (también oculta el banner de pygame)
import pygame
from pygame.locals import *

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ============================================================
# CONSTANTES
# ============================================================

# Colores por tipo de bloque (GAT)
BLOCK_COLORS = {
    0: (76, 175, 80),    # Caminable
    1: (244, 67, 54),    # No caminable
    2: (33, 150, 243),   # Agua no caminable (no snip)
    3: (3, 169, 244),    # Agua caminable
    4: (156, 39, 176),   # Agua no caminable (snip)
    5: (255, 152, 0),    # Acantilado (snip)
    6: (121, 85, 72),    # Acantilado (no snip)
}

WATER_BELOW_COLOR = (0, 188, 212)  # Color cuando está bajo agua

WATER_STATE_UNKNOWN = 0
WATER_STATE_ABOVE = 1
WATER_STATE_UNDER = 2

UNDERWATER_COLOR_REMAP = {
    0: 'WATER_BELOW',
    1: 2,
    3: 'WATER_BELOW',
    5: 4,
    6: 2,
}

BLOCK_NAMES = {
    0: "Caminable",
    1: "No caminable",
    2: "Agua no caminable (no snip)",
    3: "Agua caminable",
    4: "Agua no caminable (snip)",
    5: "Acantilado (snip)",
    6: "Acantilado (no snip)",
}

# ============================================================
# GAT PARSER
# ============================================================

class GATFileError(Exception):
    """Error al parsear un archivo .gat inválido, truncado o corrupto."""
    pass


class GATParser:
    """Parser para archivos .gat (Ground Altitude File)."""
    
    MAGIC = b"GRAT"
    
    def __init__(self, filepath, strict=True):
        """
        strict=True (default): si el archivo está truncado (menos bloques
            de los que Width*Height indican), se lanza GATFileError.
        strict=False: se recorta el mapa a las filas completas disponibles
            y se continúa, emitiendo una advertencia (sin crashear más tarde
            por índices fuera de rango).
        """
        self.filepath = filepath
        self.strict = strict
        self.magic = None
        self.version = None
        self.width = 0
        self.height = 0
        self.declared_width = 0
        self.declared_height = 0
        self.blocks = []
        self._parse()
    
    def _parse(self):
        with open(self.filepath, 'rb') as f:
            data = f.read()
        
        offset = 0
        
        # Magic (4 bytes) + versión (2 bytes). ANTES el magic incluía la
        # versión como parte de una constante fija de 6 bytes
        # (b"GRAT\x01\x02"), lo que rechazaba cualquier archivo que no
        # fuera EXACTAMENTE v1.2 — un bug real que apareció al probar un
        # archivo v1.3 real. La versión ahora se lee dinámicamente en vez
        # de compararse contra un valor fijo.
        self.magic = data[offset:offset+4]
        offset += 4
        if self.magic != self.MAGIC:
            raise GATFileError(
                f"Magic inválido: {self.magic!r}. Se esperaba {self.MAGIC!r}. "
                f"¿Es este un archivo .gat válido de Ragnarok Online?"
            )
        self.version = (data[offset], data[offset+1])
        offset += 2
        
        # Width y Height
        self.width = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        self.height = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        self.declared_width = self.width
        self.declared_height = self.height
        
        expected = self.width * self.height
        actual = (len(data) - offset) // 20
        
        if actual != expected:
            msg = (
                f"Bloques esperados: {expected} ({self.width}x{self.height}), "
                f"bloques encontrados: {actual}. El archivo podría estar "
                f"truncado o corrupto."
            )
            if self.strict:
                raise GATFileError(msg)
            print(f"[ADVERTENCIA] {msg} Se usará solo la porción disponible.")
            # Recortamos a FILAS COMPLETAS para no romper la indexación
            # (x, y) -> y * width + x que usa get_block(). Si dejáramos
            # self.height igual pero con menos bloques de los declarados,
            # get_block() podría devolver un índice fuera de rango.
            self.height = (actual // self.width) if self.width else 0
            expected = self.width * self.height
        
        # Leer bloques
        for i in range(expected):
            h1 = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            h2 = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            h3 = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            h4 = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            type_raw = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            
            self.blocks.append({
                'upper_left': h1,
                'upper_right': h2,
                'lower_left': h3,
                'lower_right': h4,
                'type': type_raw & 0xFF,
                'type_raw': type_raw,
                'avg_height': (h1 + h2 + h3 + h4) / 4.0,
            })
        
        log_info(f"[GAT] {self.width}x{self.height} = {len(self.blocks)} bloques")
    
    def get_block(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.blocks[y * self.width + x]
        return None
    
    @staticmethod
    def get_water_state(block, water_level):
        if water_level is None:
            return WATER_STATE_UNKNOWN
        return WATER_STATE_UNDER if block['avg_height'] > water_level else WATER_STATE_ABOVE
    
    @staticmethod
    def get_block_color(block, water_state=WATER_STATE_UNKNOWN):
        block_type = block['type']
        if water_state == WATER_STATE_UNDER:
            remap = UNDERWATER_COLOR_REMAP.get(block_type)
            if remap == 'WATER_BELOW':
                return WATER_BELOW_COLOR
            elif remap is not None:
                return BLOCK_COLORS.get(remap, (128, 128, 128))
        return BLOCK_COLORS.get(block_type, (128, 128, 128))

# ============================================================
# RSW PARSER (COMPLETO)
# ============================================================

class RSWParser:
    """Parser completo para archivos .rsw (Resource World File)."""
    
    MAGIC = b"GRSW"
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.version = None
        self.water_level = None
        self.water_color = None
        self.water_type = None
        self.water_wave_height = None
        self.water_wave_speed = None
        self.water_wave_pitch = None
        self.water_texture_cycling = None
        self.light = None
        self.ground_bbox = None
        self.models = []
        self.lights = []
        self.sounds = []
        self.effects = []
        self.objects_total = None
        self.objects_parsed = 0
        self.camera = {}
        self.error = None
        self._parse()
    
    def _parse(self):
        try:
            with open(self.filepath, 'rb') as f:
                data = f.read()
        except OSError as e:
            self.error = f"No se pudo leer: {e}"
            return
        
        if len(data) < 14:
            self.error = "Archivo demasiado pequeño"
            return
        
        # Magic
        magic = data[0:4]
        if magic != self.MAGIC:
            self.error = f"Magic inválido: {magic!r}"
            return
        
        # Versión
        self.version = (data[4], data[5])
        log_info(f"[RSW] Versión: {self.version[0]}.{self.version[1]}")

        # --- Offset base del header, según versión ---
        # CONFIRMADO contra TRES archivos reales de versiones distintas:
        # para v2.1 el header trae 4 campos de nombre de archivo de 40
        # bytes (ini/gnd/gat/un 4to campo) empezando en offset 6. Para
        # v2.2 hay 1 byte extra antes de esos campos. Para v2.6 hay 5
        # bytes extra (probablemente un BuildNumber de 4 bytes + el mismo
        # byte extra de v2.2, aunque no identificado con certeza exacta —
        # lo que sí está confirmado empíricamente es el desplazamiento en
        # sí: el nombre del .gnd aparece en offset 51 en vez de 46/47, y
        # con base=11 tanto la Luz como el ObjectCount dan valores
        # sensatos). Versiones intermedias (2.3-2.5) no probadas — se
        # usa el umbral más cercano confirmado como mejor esfuerzo.
        #
        # Además, a partir de v2.6 el bloque WaterConfiguration (24 bytes)
        # DESAPARECE del RSW por completo: el agua pasó a vivir solo en
        # el GND (coherente con que GND v1.8+ ya trae su propio
        # water_level, que confirmamos en un archivo real). Antes de este
        # hallazgo, asumíamos que el bloque de agua siempre estaba
        # presente — eso hacía que v2.6 leyera basura total (luz con
        # NaN, nivel de agua absurdo, 0 objetos). Confirmado leyendo
        # directo la Luz en el offset donde antes iba el agua, y
        # obteniendo valores sensatos (longitud=45, difusa en rango 0-1).
        if self.version >= (2, 6):
            header_base = 11
            has_water_block = False
        elif self.version >= (2, 2):
            header_base = 7
            has_water_block = True
        else:
            header_base = 6
            has_water_block = True

        cursor = header_base + 160  # tras los 4 campos de nombre de archivo

        if has_water_block:
            WATER_LEVEL_OFFSET = cursor
            if len(data) >= WATER_LEVEL_OFFSET + 24:
                self.water_level = struct.unpack_from('<f', data, WATER_LEVEL_OFFSET)[0]
                self.water_type = struct.unpack_from('<i', data, WATER_LEVEL_OFFSET + 4)[0]
                self.water_wave_height = struct.unpack_from('<f', data, WATER_LEVEL_OFFSET + 8)[0]
                self.water_wave_speed = struct.unpack_from('<f', data, WATER_LEVEL_OFFSET + 12)[0]
                self.water_wave_pitch = struct.unpack_from('<f', data, WATER_LEVEL_OFFSET + 16)[0]
                self.water_texture_cycling = struct.unpack_from('<i', data, WATER_LEVEL_OFFSET + 20)[0]
                log_info(f"[RSW] Nivel de agua: {self.water_level:.2f}")
            cursor += 24
        else:
            log_info("[RSW] Esta versión no trae agua en el RSW (buscar en el .gnd)")

        # --- Luz de escena (36 bytes) ---
        # CONFIRMADO byte a byte contra la salida de GRF Editor (v2.1) y
        # contra valores sensatos en archivos reales v2.2 y v2.6.
        LIGHT_OFFSET = cursor
        if len(data) >= LIGHT_OFFSET + 36:
            self.light = {
                'longitude': struct.unpack_from('<i', data, LIGHT_OFFSET)[0],
                'latitude': struct.unpack_from('<i', data, LIGHT_OFFSET + 4)[0],
                'diffuse': struct.unpack_from('<3f', data, LIGHT_OFFSET + 8),
                'ambient': struct.unpack_from('<3f', data, LIGHT_OFFSET + 20),
                'intensity': struct.unpack_from('<f', data, LIGHT_OFFSET + 32)[0],
            }
            log_info(
                f"[RSW] Luz: longitud={self.light['longitude']} "
                f"latitud={self.light['latitude']} "
                f"intensidad={self.light['intensity']:.2f}"
            )

        # --- Bounding box de terreno (16 bytes) ---
        GROUND_BBOX_OFFSET = LIGHT_OFFSET + 36
        if len(data) >= GROUND_BBOX_OFFSET + 16:
            top, bottom, left, right = struct.unpack_from('<4i', data, GROUND_BBOX_OFFSET)
            self.ground_bbox = {'top': top, 'bottom': bottom, 'left': left, 'right': right}

        offset = GROUND_BBOX_OFFSET + 16

        # A partir de v2.7 (confirmado contra un archivo real) hay un
        # bloque extra de 68 bytes entre el Ground bbox y ObjectCount,
        # relleno con 0xFF (parece un campo nuevo sin usar/sin inicializar
        # en este archivo — no se sabe qué es, así que solo se salta).
        # Sin este salto, ObjectCount se leía de un lugar equivocado y
        # el primer objeto salía con un type basura.
        if self.version >= (2, 7):
            offset += 68

        # ahí empieza ObjectCount

        # --- Objetos de escena (modelos, luces, sonidos, efectos) ---
        # Descifrado contra un archivo real (usando el código fuente de
        # BrowEdit como guía para el ORDEN de campos, y validación
        # empírica byte a byte para los offsets exactos):
        #
        #   ObjectCount (int32) va JUSTO DESPUÉS del bounding box de
        #   terreno (offset 242 en nuestro archivo real), NO después del
        #   QuadTree como habíamos supuesto — el QuadTree en realidad va
        #   DESPUÉS de la lista de objetos, no antes.
        #
        #   Cada objeto empieza con un 'type' (int32):
        #     1 = Model  (struct de 252 bytes, ver abajo)
        #     2 = Light  (struct de 112 bytes, ver abajo)
        #     3 = Sound  (struct de 196 bytes, ver abajo)
        #     4 = Effect (struct aún no descifrado)
        #
        #   MODEL (252 bytes), validado con 235 objetos reales de
        #   new_zone01.rsw (un muro de castillo hecho de segmentos en
        #   fila, con volteos de escala X alternados — coherente):
        #     +0   type (int32) = 1
        #     +4   name (40 bytes, string)       — nombre de instancia
        #     +44  animType (int32)
        #     +48  animSpeed (float)
        #     +52  blockType (int32)
        #     +56  fileName (40 bytes, string)    — ruta al .rsm
        #     +96  (120 bytes sin descifrar — no se necesitan para
        #          ubicar el modelo en la escena)
        #     +216 position (3 floats)
        #     +228 rotation (3 floats, grados)
        #     +240 scale (3 floats)
        #
        #   LIGHT (112 bytes), validado con 100 objetos reales de
        #   pay_dun01.rsw (antorchas de dungeon: posiciones distribuidas
        #   por los pasillos, colores rojizo/violeta variables entre
        #   antorchas — coherente):
        #     +0   type (int32) = 2
        #     +4   name (40 bytes, string)        — p.ej. "광원1" (Luz1)
        #     +44  (16 bytes sin descifrar)
        #     +60  (12 bytes: constante 0.7,0.7,0.7 en todas las luces
        #          probadas — probablemente NO es el color real, sino
        #          algún parámetro fijo; no lo usamos)
        #     +72  (12 bytes sin descifrar, contenido tipo texto)
        #     +84  position (3 floats)
        #     +96  color RGB (3 floats, 0.0-1.0)   — SÍ varía entre
        #          instancias de forma coherente con "antorcha", se usa
        #          como el color real de la luz
        #     +108 range (float)                   — radio de alcance
        #
        #   SOUND (196 bytes), validado con 83 objetos reales de
        #   new_zone01.rsw:
        #     +0   type (int32) = 3
        #     +4   name (aprox. 40 bytes, string)
        #     +~56 fileName (.wav, no perfectamente delimitado aún)
        #     +120 position (3 floats)             — único campo
        #          verificado con confianza
        #     (volumen/rango/ciclo: sin descifrar, no usados)
        #
        #   EFFECT (120 bytes), validado con 3 objetos reales de
        #   job4_bio.rsw (nombres "이펙터1/2/3" = "Efector 1/2/3",
        #   posiciones distintas y sensatas para cada uno — coherente).
        #   Después de las 3 instancias, el resto del archivo cuadra
        #   exacto con el QuadTree (1365 nodos × 48 bytes), buena
        #   confirmación adicional de que el tamaño de struct es correcto:
        #     +0   type (int32) = 4
        #     +4   name (28 bytes, string)          — p.ej. "이펙터1"
        #     +32  (52 bytes: valores CONSTANTES en las 3 instancias
        #          probadas — casi seguro basura de buffer sin
        #          inicializar, mismo patrón visto en otros structs; no
        #          se usan)
        #     +84  position (3 floats)               — varía por
        #          instancia de forma sensata, es el único campo que
        #          usamos
        #     +100 (4 bytes: constante 1.0 en las 3 instancias — podría
        #          ser un flag "loop"/"enabled" o una escala default; no
        #          confirmado, no se usa)
        #     +104 (16 bytes finales, todo cero en las 3 instancias — sin
        #          descifrar)
        self.models = []
        self.lights = []
        self.sounds = []
        self.effects = []
        self.objects_total = None
        self.objects_parsed = 0

        OBJECT_TYPE_MODEL = 1
        OBJECT_TYPE_LIGHT = 2
        OBJECT_TYPE_SOUND = 3
        OBJECT_TYPE_EFFECT = 4

        # El struct de Modelo cambió de tamaño DOS VECES más, confirmado
        # contra archivos reales:
        #   < 2.6:  252 bytes, fileName@56,  pos@216, rot@228, scale@240
        #   == 2.6: 253 bytes, fileName@57,  pos@217, rot@229, scale@241
        #   >= 2.7: 257 bytes, fileName@60,  pos@221, rot@233, scale@245
        # Light/Sound/Effect NO cambiaron en ninguna de estas versiones
        # (confirmado: Light sigue dando valores sensatos con los mismos
        # offsets de siempre en v2.7). No sabemos en qué sub-versión
        # exacta aparece cada cambio (sin muestras entre 2.2-2.6 ni entre
        # 2.6-2.7), así que se usan los umbrales confirmados como mejor
        # esfuerzo.
        if self.version >= (2, 7):
            MODEL_STRUCT_SIZE = 257
            MODEL_FILENAME_OFFSET = 60
            MODEL_POS_OFFSET = 221
            MODEL_ROT_OFFSET = 233
            MODEL_SCALE_OFFSET = 245
        elif self.version >= (2, 6):
            MODEL_STRUCT_SIZE = 253
            MODEL_FILENAME_OFFSET = 57
            MODEL_POS_OFFSET = 217
            MODEL_ROT_OFFSET = 229
            MODEL_SCALE_OFFSET = 241
        else:
            MODEL_STRUCT_SIZE = 252
            MODEL_FILENAME_OFFSET = 56
            MODEL_POS_OFFSET = 216
            MODEL_ROT_OFFSET = 228
            MODEL_SCALE_OFFSET = 240

        STRUCT_SIZES = {
            OBJECT_TYPE_MODEL: MODEL_STRUCT_SIZE, OBJECT_TYPE_LIGHT: 112,
            OBJECT_TYPE_SOUND: 196, OBJECT_TYPE_EFFECT: 120,
        }

        if offset + 4 <= len(data):
            n_objects = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            if 0 <= n_objects <= 100000:  # guardrail: evita un conteo absurdo
                self.objects_total = n_objects
                for i in range(n_objects):
                    if offset + 4 > len(data):
                        break
                    obj_type = struct.unpack_from('<i', data, offset)[0]
                    struct_size = STRUCT_SIZES.get(obj_type)

                    if struct_size is None:
                        print(
                            f"[RSW] Objeto {i}/{n_objects} tiene type={obj_type} "
                            f"(struct aún no descifrado) — se detiene el parseo "
                            f"de objetos aquí. Objetos leídos: {self.objects_parsed}."
                        )
                        break
                    if offset + struct_size > len(data):
                        break

                    if obj_type == OBJECT_TYPE_MODEL:
                        name = data[offset+4:offset+44].split(b'\x00')[0].decode('euc-kr', errors='replace')
                        anim_type = struct.unpack_from('<i', data, offset+44)[0]
                        anim_speed = struct.unpack_from('<f', data, offset+48)[0]
                        block_type = struct.unpack_from('<i', data, offset+52)[0]
                        file_name = data[offset+MODEL_FILENAME_OFFSET:offset+MODEL_FILENAME_OFFSET+40].split(b'\x00')[0].decode('euc-kr', errors='replace')
                        pos = struct.unpack_from('<3f', data, offset+MODEL_POS_OFFSET)
                        rot = struct.unpack_from('<3f', data, offset+MODEL_ROT_OFFSET)
                        scale = struct.unpack_from('<3f', data, offset+MODEL_SCALE_OFFSET)
                        self.models.append({
                            'name': name, 'file_name': file_name,
                            'anim_type': anim_type, 'anim_speed': anim_speed,
                            'block_type': block_type,
                            'pos': pos, 'rot': rot, 'scale': scale,
                        })

                    elif obj_type == OBJECT_TYPE_LIGHT:
                        name = data[offset+4:offset+44].split(b'\x00')[0].decode('euc-kr', errors='replace')
                        pos = struct.unpack_from('<3f', data, offset+84)
                        color = struct.unpack_from('<3f', data, offset+96)
                        light_range = struct.unpack_from('<f', data, offset+108)[0]
                        self.lights.append({
                            'name': name, 'pos': pos, 'color': color, 'range': light_range,
                        })

                    elif obj_type == OBJECT_TYPE_SOUND:
                        name = data[offset+4:offset+44].split(b'\x00')[0].decode('euc-kr', errors='replace')
                        pos = struct.unpack_from('<3f', data, offset+120)
                        self.sounds.append({'name': name, 'pos': pos})

                    elif obj_type == OBJECT_TYPE_EFFECT:
                        name = data[offset+4:offset+32].split(b'\x00')[0].decode('euc-kr', errors='replace')
                        pos = struct.unpack_from('<3f', data, offset+84)
                        self.effects.append({'name': name, 'pos': pos})

                    self.objects_parsed += 1
                    offset += struct_size

                log_info(
                    f"[RSW] {self.objects_parsed}/{n_objects} objetos parseados "
                    f"({len(self.models)} modelos, {len(self.lights)} luces, "
                    f"{len(self.sounds)} sonidos, {len(self.effects)} efectos)."
                )

    def get_model_positions(self):
        return [(m['name'], m['pos'][0], m['pos'][1], m['pos'][2]) 
                for m in self.models]

# ============================================================
# GND PARSER
# ============================================================

class GNDParser:
    """
    Parser para archivos .gnd (Ground File), según la especificación
    documentada en rdw-archive/RagnarokFileFormats (GND.MD).

    Layout real (v1.7+):
        Header (4)  = "GRGN"           <- OJO: NO es "GRND"
        Version (2) = major, minor byte
        Width (4), Height (4)          <- dimensiones de la GRILLA DE CUBOS
                                           (la mitad del ancho/alto en tiles GAT)
        Scale (4, float)                <- factor de escala de geometría
        TextureCount (4)
        TexturePathLength (4)           <- normalmente 80
        TexturePaths (TextureCount * TexturePathLength)
        LightmapSlices (variable)       <- no se usa aquí, solo se salta
        SurfaceCount (4)
        SurfaceDefinitions (SurfaceCount * 56 bytes) <- aquí vive el texture_id real
        GroundMeshCubes (Width*Height * 28 bytes)     <- cada cubo referencia
                                                          una superficie (o -1)
        [v1.8+] WaterPlaneConfiguration (al final del archivo)

    Un tile GAT (x, y) pertenece al cubo (x // 2, y // 2), ya que cada
    superficie de cubo cubre 2x2 tiles GAT.

    Esta clase es defensiva: si el archivo no calza exactamente con lo
    documentado (variantes de versión, etc.), se detiene con self.error
    en vez de lanzar una excepción o dejar datos corruptos a medio leer;
    el visualizador simplemente sigue sin texturas GND en ese caso.
    """

    MAGIC = b"GRGN"

    def __init__(self, filepath):
        self.filepath = filepath
        self.version = (0, 0)
        self.width = 0    # ancho en CUBOS (no en tiles GAT)
        self.height = 0   # alto en CUBOS
        self.scale = 10.0
        self.water_level = None
        self.water_type = None
        self.water_wave_height = None
        self.water_wave_speed = None
        self.water_wave_pitch = None
        self.water_texture_cycling = None
        self.water_zones = []
        self.water_zone_count_x = 0
        self.water_zone_count_y = 0
        self.textures = []
        self.surfaces = []   # cada uno: {'texture_id', 'lightmap_id', uvs...}
        self.lightmaps = []  # cada uno: bytes RGBA (w*h*4) sombra+color
        self.lightmap_width = 8
        self.lightmap_height = 8
        self.cubes = []      # cada uno: {'surface_top': int, ...}
        self.error = None
        self._parse()

    def _parse(self):
        try:
            with open(self.filepath, 'rb') as f:
                data = f.read()
        except OSError as e:
            self.error = f"No se pudo leer: {e}"
            return

        try:
            offset = 0

            magic = data[offset:offset + 4]
            offset += 4
            if magic != self.MAGIC:
                self.error = (
                    f"Magic inválido: {magic!r}. Se esperaba {self.MAGIC!r} "
                    f"('GRGN', no 'GRND' a pesar de la extensión .gnd)."
                )
                return

            major, minor = data[offset], data[offset + 1]
            offset += 2
            self.version = (major, minor)
            log_info(f"[GND] Versión: {major}.{minor}")

            self.width = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            self.height = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            self.scale = struct.unpack_from('<f', data, offset)[0]
            offset += 4

            log_info(f"[GND] Grilla de cubos: {self.width}x{self.height}, escala={self.scale:.2f}")

            # --- Texturas ---
            n_textures = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            texture_path_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            for _ in range(n_textures):
                raw = data[offset:offset + texture_path_len]
                tex_name = raw.split(b'\x00')[0].decode('latin-1', errors='replace')
                offset += texture_path_len
                self.textures.append(tex_name)

            log_info(f"[GND] {n_textures} texturas")

            # --- Lightmap slices ---
            # Orden confirmado: count, width, height, pixel_format
            # Cada slice: shadow (w*h bytes AO) + color RGB (w*h*3)
            lm_count = struct.unpack_from('<I', data, offset)[0]
            lm_width = struct.unpack_from('<I', data, offset + 4)[0]
            lm_height = struct.unpack_from('<I', data, offset + 8)[0]
            lm_pixel_format = struct.unpack_from('<I', data, offset + 12)[0]
            offset += 16
            self.lightmap_width = max(1, lm_width)
            self.lightmap_height = max(1, lm_height)
            self.lightmaps = []
            shadow_size = lm_width * lm_height
            color_size = shadow_size * 3
            for _ in range(lm_count):
                shadow = data[offset:offset + shadow_size]
                offset += shadow_size
                color = data[offset:offset + color_size]
                offset += color_size
                # RGBA: RGB del lightmap color, A = AO/sombra (255=sin sombra)
                rgba = bytearray(shadow_size * 4)
                for i in range(shadow_size):
                    rgba[i*4 + 0] = color[i*3 + 0] if i*3 + 2 < len(color) else 255
                    rgba[i*4 + 1] = color[i*3 + 1] if i*3 + 2 < len(color) else 255
                    rgba[i*4 + 2] = color[i*3 + 2] if i*3 + 2 < len(color) else 255
                    rgba[i*4 + 3] = shadow[i] if i < len(shadow) else 255
                self.lightmaps.append(bytes(rgba))
            log_info(f"[GND] {lm_count} lightmaps ({lm_width}x{lm_height})")

            # --- Surface Definitions ---
            # Estructura confirmada contra un archivo real: 40 bytes por
            # superficie (no 56), con texture_id como int16 (no int32) en
            # el offset relativo 32. Verificado: con estos valores, el
            # array de superficies termina exactamente donde deben
            # empezar los cubos (sin bytes sobrantes ni faltantes).
            # Los primeros 32 bytes (8 floats) son las coordenadas UV de
            # las 4 esquinas — las texturas GND suelen ser atlas con
            # varias sub-imágenes, así que un UV ingenuo (0,0)-(1,1)
            # muestrea el atlas completo estirado en vez de la sub-imagen
            # correcta. Orden asumido por analogía con las alturas
            # (bottom_left, bottom_right, top_left, top_right); no
            # verificado con la misma profundidad que el resto, pero el
            # resultado visual (ver visor 3D) confirma que es razonable.
            surface_count = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            SURFACE_STRUCT_SIZE = 40
            for _ in range(surface_count):
                # UVs GND: u0,u1,u2,u3, v0,v1,v2,v3
                # +32 texture_id (i16), +34 lightmap_id (i16), +36 color BGRA
                uv = struct.unpack_from('<8f', data, offset)
                texture_id = struct.unpack_from('<h', data, offset + 32)[0]
                lightmap_id = struct.unpack_from('<h', data, offset + 34)[0]
                b, g, r, a = struct.unpack_from('<4B', data, offset + 36)
                self.surfaces.append({
                    'texture_id': texture_id,
                    'lightmap_id': lightmap_id,
                    'vertex_color': (r / 255.0, g / 255.0, b / 255.0, a / 255.0),
                    'uv_bottom_left':  (uv[0], uv[4]),
                    'uv_bottom_right': (uv[1], uv[5]),
                    'uv_top_left':     (uv[2], uv[6]),
                    'uv_top_right':    (uv[3], uv[7]),
                })
                offset += SURFACE_STRUCT_SIZE

            log_info(f"[GND] {surface_count} superficies")

            # --- Ground Mesh Cubes (Width x Height cubos, 28 bytes c/u) ---
            # Orden de alturas confirmado contra el código fuente de BrowEdit
            # (MapView.Heightmode.cpp): h1=bottomleft, h2=bottomright,
            # h3=topleft, h4=topright.
            CUBE_STRUCT_SIZE = 28
            n_cubes = self.width * self.height
            for _ in range(n_cubes):
                h_bottom_left = struct.unpack_from('<f', data, offset)[0]
                h_bottom_right = struct.unpack_from('<f', data, offset + 4)[0]
                h_top_left = struct.unpack_from('<f', data, offset + 8)[0]
                h_top_right = struct.unpack_from('<f', data, offset + 12)[0]
                surface_top = struct.unpack_from('<i', data, offset + 16)[0]
                surface_north = struct.unpack_from('<i', data, offset + 20)[0]
                surface_east = struct.unpack_from('<i', data, offset + 24)[0]
                self.cubes.append({
                    'h_bottom_left': h_bottom_left,
                    'h_bottom_right': h_bottom_right,
                    'h_top_left': h_top_left,
                    'h_top_right': h_top_right,
                    'surface_top': surface_top,
                    'surface_north': surface_north,
                    'surface_east': surface_east,
                })
                offset += CUBE_STRUCT_SIZE

            log_info(f"[GND] {len(self.cubes)} cubos leídos")

            # --- Water Plane Configuration (v1.8+), al FINAL del archivo ---
            #
            # CORREGIDO: antes leíamos un solo float (el nivel de agua
            # "por defecto") e ignorábamos el resto del bloque. Con un
            # archivo real de agua multi-zona (mapa "icecastle") se
            # confirmó que el bloque completo es mucho más grande:
            #
            #   Config global (32 bytes): level, type, wave_height,
            #     wave_speed, wave_pitch, texture_cycling (mismos 6
            #     campos que el WaterConfiguration del RSW), más DOS
            #     campos extra: zone_count_x, zone_count_y (int32 cada
            #     uno) — la grilla de zonas de agua.
            #   Luego zone_count_x * zone_count_y zonas, cada una con
            #     los mismos 6 campos (24 bytes) que la config global,
            #     pero con su PROPIO nivel de agua (y demás parámetros).
            #
            # Validado: con esta estructura, el parser consume el
            # archivo completo exacto (sin sobrar ni faltar un byte), y
            # los niveles de agua por zona varían de verdad entre sí
            # (de 20.0 a -105.0 en el archivo de prueba) — confirmando
            # que es agua multi-zona real, no ruido.
            #
            # NO VALIDADO: a qué región del terreno corresponde cada
            # zona exactamente (qué rango de cubos cubre la zona [x,y]
            # de la grilla). Se guarda todo el dato crudo en
            # self.water_zones; self.water_level sigue siendo el nivel
            # GLOBAL/por defecto, para no romper el uso existente (2D
            # viewer, terrain_viewer_3d.py) que solo necesita un valor
            # único.
            self.water_zones = []
            self.water_zone_count_x = 0
            self.water_zone_count_y = 0
            if (major, minor) >= (1, 8) and offset + 32 <= len(data):
                (self.water_level, self.water_type, self.water_wave_height,
                 self.water_wave_speed, self.water_wave_pitch,
                 self.water_texture_cycling, zone_count_x, zone_count_y) = \
                    struct.unpack_from('<fifffiii', data, offset)
                offset += 32
                log_info(f"[GND] Nivel de agua (global/default): {self.water_level:.2f}")

                if 0 <= zone_count_x * zone_count_y <= 100000:
                    self.water_zone_count_x = zone_count_x
                    self.water_zone_count_y = zone_count_y
                    # GND 1.8: cada zona trae SOLO su nivel (float); el resto de los
                    # parámetros se hereda de la config global. GND 1.9: 24 bytes completos.
                    zone_size = 4 if (major, minor) == (1, 8) else 24
                    for zy in range(zone_count_y):
                        row = []
                        for zx in range(zone_count_x):
                            if offset + zone_size > len(data):
                                break
                            if zone_size == 4:
                                z_level = struct.unpack_from('<f', data, offset)[0]
                                z_type, z_wh, z_ws, z_wp, z_tc = (
                                    self.water_type, self.water_wave_height,
                                    self.water_wave_speed, self.water_wave_pitch,
                                    self.water_texture_cycling)
                            else:
                                (z_level, z_type, z_wh, z_ws, z_wp, z_tc) = \
                                    struct.unpack_from('<fifffi', data, offset)
                            row.append({
                                'level': z_level, 'type': z_type,
                                'wave_height': z_wh, 'wave_speed': z_ws,
                                'wave_pitch': z_wp, 'texture_cycling': z_tc,
                            })
                            offset += zone_size
                        self.water_zones.append(row)
                    if zone_count_x * zone_count_y > 1:
                        levels = [z['level'] for row in self.water_zones for z in row]
                        log_info(
                            f"[GND] Agua multi-zona: {zone_count_x}x{zone_count_y} "
                            f"zonas, niveles de {min(levels):.1f} a {max(levels):.1f}"
                        )

        except (struct.error, IndexError, ZeroDivisionError) as e:
            self.error = f"Error al parsear estructura GND: {e}"
            print(f"[GND] {self.error}")

    # ------------------------------------------------------------------
    # Zonas de agua (GND 1.8 / 1.9)
    # ------------------------------------------------------------------
    def water_zone_rect(self, zx, zy):
        """Rango de cubos (x0, x1, y0, y1) que cubre la zona (zx, zy); x1/y1 exclusivos.

        Regla de BrowEdit3: el terreno se parte en zonas de width//U x height//V cubos y
        la última columna/fila absorbe el resto."""
        U = max(1, self.water_zone_count_x)
        V = max(1, self.water_zone_count_y)
        per_w = self.width // U
        per_h = self.height // V
        x0 = per_w * zx
        x1 = self.width if zx == U - 1 else per_w * (zx + 1)
        y0 = per_h * zy
        y1 = self.height if zy == V - 1 else per_h * (zy + 1)
        return x0, x1, y0, y1

    def water_zone_at(self, cube_x, cube_y):
        """Zona de agua (dict) que corresponde al cubo (cube_x, cube_y), o None."""
        U, V = self.water_zone_count_x, self.water_zone_count_y
        if U < 1 or V < 1 or len(self.water_zones) != V:
            return None
        per_w = max(1, self.width // U)
        per_h = max(1, self.height // V)
        zx = min(max(0, cube_x // per_w), U - 1)
        zy = min(max(0, cube_y // per_h), V - 1)
        row = self.water_zones[zy]
        return row[zx] if zx < len(row) else None

    def water_level_at(self, cube_x, cube_y):
        """Nivel de agua en un cubo: el de su zona, o el global si no hay zonas."""
        z = self.water_zone_at(cube_x, cube_y)
        return z['level'] if z is not None else self.water_level

    def get_texture_name(self, gat_x, gat_y):
        """
        Obtener el nombre de textura para un tile GAT (gat_x, gat_y).
        Cada cubo del GND cubre 2x2 tiles GAT, así que primero convertimos
        a coordenadas de cubo.
        """
        if self.width <= 0 or self.height <= 0:
            return None

        cube_x = gat_x // 2
        cube_y = gat_y // 2
        if not (0 <= cube_x < self.width and 0 <= cube_y < self.height):
            return None

        cube_idx = cube_y * self.width + cube_x
        if not (0 <= cube_idx < len(self.cubes)):
            return None

        surface_id = self.cubes[cube_idx]['surface_top']
        if not (0 <= surface_id < len(self.surfaces)):
            return None  # -1 = superficie sin textura (no se renderiza)

        texture_id = self.surfaces[surface_id]['texture_id']
        if not (0 <= texture_id < len(self.textures)):
            return None

        return self.textures[texture_id]

# ============================================================
# RSM PARSER
# ============================================================

class RSMFileError(Exception):
    """Error al parsear un archivo .rsm inválido, truncado o corrupto."""
    pass


class RSMParser:
    """
    Parser para archivos .rsm (modelos 3D: árboles, edificios, props).

    Implementado siguiendo el código fuente real de BrowEdit (Rsm.cpp,
    Borf/browedit en GitHub), el editor de mapas más usado por la
    comunidad RO, y validado 100% contra un archivo .rsm real: los
    conteos parseados (vértices, coordenadas de textura, caras, mallas,
    keyframes) coinciden exactamente con lo reportado por GRF Editor
    para ese mismo archivo, y el parser consume el archivo completo sin
    sobrar ni faltar un solo byte.

    LIMITACIONES CONOCIDAS:
    - Solo soporta RSM v1.x (extensión .rsm). RSM2 (.rsm2, usado en
      clientes más nuevos) tiene un formato distinto y no está soportado.
    - Solo se probó contra un archivo real con 1 sola malla, sin
      jerarquía de nodos hijos ni animación. La lectura de nodos hijos
      (mesh.fetch_children en BrowEdit) y de keyframes de animación
      sigue la lógica del código fuente de BrowEdit pero no se ha
      podido validar contra un archivo real que las use.
    - No importa el bounding box (se recalcula a partir de la geometría,
      igual que hace BrowEdit; no es un campo real del archivo).
    """

    MAGIC = b"GRSM"

    def __init__(self, filepath):
        self.filepath = filepath
        self.version = None
        self.anim_len = 0
        self.shade_type = 0
        self.alpha = 255
        self.textures = []
        self.root_node_name = ""
        self.meshes = []  # lista de dicts (ver _read_mesh)
        self.error = None
        self._parse()

    def _parse(self):
        try:
            with open(self.filepath, 'rb') as f:
                data = f.read()
        except OSError as e:
            self.error = f"No se pudo leer: {e}"
            return

        try:
            off = 0
            magic = data[off:off + 4]
            off += 4
            if magic != self.MAGIC:
                raise RSMFileError(f"Magic inválido: {magic!r}. Se esperaba {self.MAGIC!r}.")

            # Versión: 2 bytes BIG-ENDIAN (confirmado: BrowEdit usa
            # readShort() y compara contra constantes como 0x0104 = v1.4;
            # el byte alto es el major, el bajo el minor).
            version_raw = struct.unpack_from('>H', data, off)[0]
            off += 2
            self.version = (version_raw >> 8, version_raw & 0xFF)

            self.anim_len = struct.unpack_from('<I', data, off)[0]
            off += 4
            self.shade_type = struct.unpack_from('<I', data, off)[0]
            off += 4

            if version_raw >= 0x0104:
                self.alpha = data[off]
                off += 1
            else:
                self.alpha = 255

            off += 16  # 16 bytes reservados, siempre en cero (no se usan)

            texture_count = struct.unpack_from('<I', data, off)[0]
            off += 4
            if texture_count > 1000:
                raise RSMFileError(f"texture_count implausible: {texture_count} (archivo corrupto)")

            for _ in range(texture_count):
                raw = data[off:off + 40]
                off += 40
                name = raw.split(b'\x00')[0].decode('euc-kr', errors='replace')
                self.textures.append(name)

            self.root_node_name = data[off:off + 40].split(b'\x00')[0].decode(
                'euc-kr', errors='replace')
            off += 40

            mesh_count = struct.unpack_from('<I', data, off)[0]
            off += 4
            if mesh_count > 1000:
                raise RSMFileError(f"mesh_count implausible: {mesh_count} (archivo corrupto)")

            for _ in range(mesh_count):
                mesh, off = self._read_mesh(data, off)
                self.meshes.append(mesh)

            # Campos finales a nivel de modelo (no de malla).
            #
            # CORREGIDO tras contrastar con un informe independiente
            # (parser de referencia de GRF Editor / Tokeiburu): el conteo
            # de "scale keyframes" a nivel de archivo completo (ubicación
            # histórica, distinta de los scale keyframes por malla de
            # v1.6+) SOLO existe si version < 1.6. Antes lo leíamos sin
            # condición — en nuestro único archivo de prueba real (v1.4)
            # no se notaba porque ahí sí corresponde leerlo, pero para
            # cualquier .rsm v1.6+ esto desalineaba silenciosamente la
            # lectura de VolumeBoxes. Sin un archivo real v1.6+ a mano
            # para confirmar el fix con la misma certeza que el resto del
            # parser, así que queda marcado como mejor-esfuerzo.
            self.top_level_scale_keyframes = []
            if self.version < (1, 6):
                if off + 4 <= len(data):
                    n = struct.unpack_from('<I', data, off)[0]
                    off += 4
                    for _ in range(n):
                        if off + 20 > len(data):
                            break
                        frame = struct.unpack_from('<i', data, off)[0]
                        sx, sy, sz, _unused = struct.unpack_from('<4f', data, off + 4)
                        self.top_level_scale_keyframes.append({
                            'frame': frame, 'scale': (sx, sy, sz),
                        })
                        off += 20

            # VolumeBoxes: cajas de colisión/volumen. Estructura tomada
            # del mismo informe de referencia; no la teníamos antes.
            self.volume_boxes = []
            if off + 4 <= len(data):
                n_boxes = struct.unpack_from('<I', data, off)[0]
                off += 4
                if 0 <= n_boxes <= 10000:
                    for _ in range(n_boxes):
                        box_size = 40 if self.version >= (1, 3) else 36
                        if off + box_size > len(data):
                            break
                        size = struct.unpack_from('<3f', data, off)
                        position = struct.unpack_from('<3f', data, off + 12)
                        rotation = struct.unpack_from('<3f', data, off + 24)
                        flag = None
                        if self.version >= (1, 3):
                            flag = struct.unpack_from('<i', data, off + 36)[0]
                        self.volume_boxes.append({
                            'size': size, 'position': position,
                            'rotation': rotation, 'flag': flag,
                        })
                        off += box_size

            log_info(
                f"[RSM] v{self.version[0]}.{self.version[1]} — "
                f"{len(self.textures)} texturas, {mesh_count} malla(s), "
                f"nodo raíz='{self.root_node_name}'"
            )

        except RSMFileError as e:
            self.error = str(e)
            print(f"[RSM] {self.error}")
        except (struct.error, IndexError, UnicodeDecodeError) as e:
            self.error = f"Error al parsear estructura RSM: {e}"
            print(f"[RSM] {self.error}")

    def _read_mesh(self, data, off):
        """Lee una malla (nodo) completa. Devuelve (dict_malla, nuevo_offset)."""
        name = data[off:off + 40].split(b'\x00')[0].decode('euc-kr', errors='replace')
        off += 40
        parent_name = data[off:off + 40].split(b'\x00')[0].decode('euc-kr', errors='replace')
        off += 40

        tex_idx_count = struct.unpack_from('<I', data, off)[0]
        off += 4
        texture_indices = []
        for _ in range(tex_idx_count):
            texture_indices.append(struct.unpack_from('<i', data, off)[0])
            off += 4

        # Matriz de rotación/orientación local 3x3 (9 floats)
        offset_matrix = struct.unpack_from('<9f', data, off)
        off += 36

        pos_ = struct.unpack_from('<3f', data, off); off += 12
        pos = struct.unpack_from('<3f', data, off); off += 12
        rot_angle = struct.unpack_from('<f', data, off)[0]; off += 4
        rot_axis = struct.unpack_from('<3f', data, off); off += 12
        scale = struct.unpack_from('<3f', data, off); off += 12

        vertex_count = struct.unpack_from('<I', data, off)[0]
        off += 4
        vertices = []
        for _ in range(vertex_count):
            vertices.append(struct.unpack_from('<3f', data, off))
            off += 12

        texcoord_count = struct.unpack_from('<I', data, off)[0]
        off += 4
        texcoords = []
        for _ in range(texcoord_count):
            # Desde v1.2, cada entrada tiene un campo extra antes de u,v.
            # BrowEdit lo llama "3d texture coordinate" (float, siempre 0
            # en nuestro archivo de prueba). Un informe independiente
            # (parser de GRF Editor/Tokeiburu) lo describe distinto: un
            # uint32 "Color" por vértice de textura. Ambas lecturas son
            # compatibles a nivel de bytes (mismos 4 bytes, interpretación
            # distinta) — no lo usamos para nada en el render todavía, así
            # que lo saltamos sin decidir cuál interpretación es la
            # correcta; si en el futuro se quiere tinte de vértice, hay
            # que resolver esta ambigüedad contra un archivo real que use
            # un valor distinto de 0.
            if self.version >= (1, 2):
                off += 4
            u, v = struct.unpack_from('<2f', data, off)
            off += 8
            texcoords.append((u, v))

        face_count = struct.unpack_from('<I', data, off)[0]
        off += 4
        faces = []
        for _ in range(face_count):
            v_idx = struct.unpack_from('<3H', data, off); off += 6
            tv_idx = struct.unpack_from('<3H', data, off); off += 6
            tex_id = struct.unpack_from('<H', data, off)[0]; off += 2
            off += 2  # padding, sin uso
            two_sided = struct.unpack_from('<i', data, off)[0]; off += 4
            smooth_group = struct.unpack_from('<i', data, off)[0]; off += 4
            faces.append({
                'vertex_ids': v_idx,
                'texcoord_ids': tv_idx,
                'texture_id': tex_id,
                'two_sided': bool(two_sided),
                'smooth_group': smooth_group,
            })

        # Scale keyframes POR MALLA (v1.6+ únicamente — para v<1.6 el
        # conteo de scale keyframes vive una sola vez al final del
        # archivo completo, no acá; ver _parse). Antes esto faltaba por
        # completo: para cualquier .rsm v1.6+ con esto presente, la
        # lectura se desalinearía a partir de acá. Corregido tras
        # contrastar con un informe independiente (parser de GRF
        # Editor/Tokeiburu); sin un archivo real v1.6+ con scale
        # keyframes para confirmarlo con la misma certeza que el resto.
        scale_keyframes = []
        if self.version >= (1, 6):
            n_scale = struct.unpack_from('<I', data, off)[0]
            off += 4
            for _ in range(n_scale):
                frame = struct.unpack_from('<i', data, off)[0]
                sx, sy, sz, _unused = struct.unpack_from('<4f', data, off + 4)
                scale_keyframes.append({'frame': frame, 'scale': (sx, sy, sz)})
                off += 20

        # Keyframes de animación por nodo (posición/rotación). No se
        # pudieron validar contra un archivo real que los use; se leen
        # para mantener el offset correcto y quedan disponibles, pero el
        # visualizador no los usa todavía.
        frame_count = struct.unpack_from('<I', data, off)[0]
        off += 4
        frames = []
        for _ in range(frame_count):
            time = struct.unpack_from('<I', data, off)[0]; off += 4
            qx, qy, qz, qw = struct.unpack_from('<4f', data, off); off += 16
            frames.append({'time': time, 'quat': (qx, qy, qz, qw)})

        mesh = {
            'name': name,
            'parent_name': parent_name,
            'texture_indices': texture_indices,
            'offset_matrix': offset_matrix,
            'pos_': pos_,
            'pos': pos,
            'rot_angle': rot_angle,
            'rot_axis': rot_axis,
            'scale': scale,
            'vertices': vertices,
            'texcoords': texcoords,
            'faces': faces,
            'scale_keyframes': scale_keyframes,
            'frames': frames,
        }
        return mesh, off

    def get_root_mesh(self):
        """Devuelve el dict de la malla raíz (por nombre), o la primera si no se encuentra."""
        for mesh in self.meshes:
            if mesh['name'] == self.root_node_name:
                return mesh
        return self.meshes[0] if self.meshes else None


# ============================================================
# RSM2 PARSER (v2.2+, múltiples raíces, jerarquía, texturas por malla)
# ============================================================

class RSM2Parser:
    """
    Parser para archivos .rsm2 (RSM v2.2+): formato nuevo de modelos 3D,
    con múltiples nodos raíz, texturas por malla en vez de una lista
    global, y strings con prefijo de longitud en vez de campos fijos de
    40 bytes.

    Implementado y validado contra un archivo real (v2.3, un prop de
    flores con 2 mallas en jerarquía padre-hijo — la primera vez que
    probamos jerarquía real, no solo el caso raíz-sin-hijos de RSM1):
    el parser consume el archivo completo sin sobrar ni faltar un solo
    byte, y las 18 caras totales (12+6) tienen 0 índices inválidos.

    Referencia de estructura: informe técnico basado en el parser de
    referencia de GRF Editor (Tokeiburu) y RagnarokFileFormats.

    LIMITACIONES:
    - Solo probado contra v2.3. La rama v2.2 (con lista global de
      texturas, a diferencia de v2.3 que las mueve a cada malla) se
      implementa siguiendo el mismo informe pero no se validó contra un
      archivo real v2.2.
    - Solo probado con mallas SIN animación (keyframes en 0 en el
      archivo de prueba). Las animation keyframes (escala/rotación/
      posición/textura) se leen para no romper el offset, pero no hay
      garantía de que la interpretación de sus valores sea correcta.
    - El cálculo de la matriz de transformación final para jerarquías
      con padre (sección 5 del informe, que usa inversas de matrices de
      padres) NO está implementado todavía — este parser solo expone los
      datos crudos (matrix, position_, jerarquía padre/hijo), no calcula
      la transformación final. Ver `resolve_hierarchy()`.
    """

    MAGIC = b"GRSM"

    def __init__(self, filepath):
        self.filepath = filepath
        self.version = None
        self.anim_len = 0
        self.shade_type = 0
        self.alpha = 255
        self.fps = 30.0
        self.textures = []  # solo se llena en v2.2 (lista global)
        self.main_mesh_names = []
        self.meshes = []
        self.error = None
        self._parse()

    def _read_len_string(self, data, off):
        slen = struct.unpack_from('<i', data, off)[0]
        off += 4
        s = data[off:off + slen].decode('utf-8', errors='replace')
        off += slen
        return s, off

    def _parse(self):
        try:
            with open(self.filepath, 'rb') as f:
                data = f.read()
        except OSError as e:
            self.error = f"No se pudo leer: {e}"
            return

        try:
            off = 0
            magic = data[off:off + 4]
            off += 4
            if magic != self.MAGIC:
                raise RSMFileError(f"Magic inválido: {magic!r}. Se esperaba {self.MAGIC!r}.")

            major, minor = data[off], data[off + 1]
            off += 2
            self.version = (major, minor)
            if self.version < (2, 2):
                raise RSMFileError(
                    f"Versión {major}.{minor} no es RSM2 (v2.2+). "
                    f"Usá RSMParser para archivos .rsm clásicos."
                )

            self.anim_len = struct.unpack_from('<i', data, off)[0]
            off += 4
            self.shade_type = struct.unpack_from('<i', data, off)[0]
            off += 4
            self.alpha = data[off]
            off += 1

            self.fps = struct.unpack_from('<f', data, off)[0]
            off += 4

            if self.version >= (2, 3):
                # v2.3: sin lista global de texturas (se movieron a cada malla)
                main_count = struct.unpack_from('<i', data, off)[0]
                off += 4
            else:
                # v2.2: sí hay lista global de texturas
                tex_count = struct.unpack_from('<i', data, off)[0]
                off += 4
                for _ in range(tex_count):
                    tex, off = self._read_len_string(data, off)
                    self.textures.append(tex)
                main_count = struct.unpack_from('<i', data, off)[0]
                off += 4

            for _ in range(main_count):
                name, off = self._read_len_string(data, off)
                self.main_mesh_names.append(name)

            mesh_count = struct.unpack_from('<i', data, off)[0]
            off += 4
            if mesh_count > 10000:
                raise RSMFileError(f"mesh_count implausible: {mesh_count}")

            for _ in range(mesh_count):
                mesh, off = self._read_mesh(data, off)
                self.meshes.append(mesh)

            # VolumeBoxes (no hay scale-keyframes a nivel de archivo en
            # RSM2 — esos campos, si existen, viven en <1.6 solamente)
            self.volume_boxes = []
            if off + 4 <= len(data):
                n_boxes = struct.unpack_from('<I', data, off)[0]
                off += 4
                if 0 <= n_boxes <= 10000:
                    for _ in range(n_boxes):
                        if off + 40 > len(data):
                            break
                        size = struct.unpack_from('<3f', data, off)
                        position = struct.unpack_from('<3f', data, off + 12)
                        rotation = struct.unpack_from('<3f', data, off + 24)
                        flag = struct.unpack_from('<i', data, off + 36)[0]
                        self.volume_boxes.append({
                            'size': size, 'position': position,
                            'rotation': rotation, 'flag': flag,
                        })
                        off += 40

            self.resolve_hierarchy()

            log_info(
                f"[RSM2] v{self.version[0]}.{self.version[1]} — "
                f"{mesh_count} malla(s), {len(self.main_mesh_names)} "
                f"raíz(ces): {self.main_mesh_names}"
            )

        except RSMFileError as e:
            self.error = str(e)
            print(f"[RSM2] {self.error}")
        except (struct.error, IndexError, UnicodeDecodeError) as e:
            self.error = f"Error al parsear estructura RSM2: {e}"
            print(f"[RSM2] {self.error}")

    def _read_mesh(self, data, off):
        name, off = self._read_len_string(data, off)
        parent_name, off = self._read_len_string(data, off)

        tex_count = struct.unpack_from('<i', data, off)[0]
        off += 4
        textures = []
        texture_indices = []
        if self.version >= (2, 3):
            for _ in range(tex_count):
                tex, off = self._read_len_string(data, off)
                textures.append(tex)
            texture_indices = list(range(tex_count))
        else:
            for _ in range(tex_count):
                texture_indices.append(struct.unpack_from('<i', data, off)[0])
                off += 4

        matrix = struct.unpack_from('<9f', data, off)
        off += 36

        # v2.2+: solo Position_ (3 floats). offset/rotangle/rotaxis/scale
        # NO existen en el archivo para esta versión (se manejan vía
        # keyframes o quedan bakeados en la matriz).
        position_ = struct.unpack_from('<3f', data, off)
        off += 12

        vertex_count = struct.unpack_from('<i', data, off)[0]
        off += 4
        vertices = []
        for _ in range(vertex_count):
            vertices.append(struct.unpack_from('<3f', data, off))
            off += 12

        texcoord_count = struct.unpack_from('<i', data, off)[0]
        off += 4
        texcoords = []
        texcoord_colors = []
        for _ in range(texcoord_count):
            color = struct.unpack_from('<I', data, off)[0]
            off += 4
            u, v = struct.unpack_from('<2f', data, off)
            off += 8
            texcoords.append((u, v))
            texcoord_colors.append(color)

        face_count = struct.unpack_from('<i', data, off)[0]
        off += 4
        faces = []
        for _ in range(face_count):
            face_len = struct.unpack_from('<i', data, off)[0]
            off += 4
            face_start = off
            v_idx = struct.unpack_from('<3H', data, off); off += 6
            tv_idx = struct.unpack_from('<3H', data, off); off += 6
            tex_id = struct.unpack_from('<H', data, off)[0]; off += 2
            off += 2  # padding
            two_sided = struct.unpack_from('<i', data, off)[0]; off += 4
            # SmoothGroup: puede variar en tamaño según face_len (hasta
            # 3 valores en v2.2+, uno por vértice de la cara).
            remaining = face_len - (off - face_start)
            n_smooth = max(0, remaining // 4)
            smooth_groups = struct.unpack_from(f'<{n_smooth}i', data, off) if n_smooth else ()
            off += n_smooth * 4
            faces.append({
                'vertex_ids': v_idx,
                'texcoord_ids': tv_idx,
                'texture_id': tex_id,
                'two_sided': bool(two_sided),
                'smooth_groups': smooth_groups,
            })

        scale_keyframes = []
        if self.version >= (1, 6):
            n = struct.unpack_from('<I', data, off)[0]
            off += 4
            for _ in range(n):
                frame = struct.unpack_from('<i', data, off)[0]
                sx, sy, sz, _unused = struct.unpack_from('<4f', data, off + 4)
                scale_keyframes.append({'frame': frame, 'scale': (sx, sy, sz)})
                off += 20

        rot_keyframes = []
        n_rot = struct.unpack_from('<I', data, off)[0]
        off += 4
        for _ in range(n_rot):
            frame = struct.unpack_from('<i', data, off)[0]
            qx, qy, qz, qw = struct.unpack_from('<4f', data, off + 4)
            rot_keyframes.append({'frame': frame, 'quat': (qx, qy, qz, qw)})
            off += 20

        pos_keyframes = []
        if self.version >= (2, 2):
            n_pos = struct.unpack_from('<I', data, off)[0]
            off += 4
            for _ in range(n_pos):
                frame = struct.unpack_from('<i', data, off)[0]
                x, y, z = struct.unpack_from('<3f', data, off + 4)
                _unused = struct.unpack_from('<i', data, off + 16)[0]
                pos_keyframes.append({'frame': frame, 'pos': (x, y, z)})
                off += 20

        texture_keyframe_groups = []
        if self.version >= (2, 3):
            n_groups = struct.unpack_from('<I', data, off)[0]
            off += 4
            for _ in range(n_groups):
                tex_id_anim = struct.unpack_from('<i', data, off)[0]; off += 4
                n_anims = struct.unpack_from('<i', data, off)[0]; off += 4
                anims = []
                for _ in range(n_anims):
                    anim_type = struct.unpack_from('<i', data, off)[0]; off += 4
                    n_frames = struct.unpack_from('<i', data, off)[0]; off += 4
                    frames = []
                    for _ in range(n_frames):
                        fr = struct.unpack_from('<i', data, off)[0]
                        val = struct.unpack_from('<f', data, off + 4)[0]
                        frames.append({'frame': fr, 'offset': val})
                        off += 8
                    anims.append({'type': anim_type, 'frames': frames})
                texture_keyframe_groups.append({'texture_id': tex_id_anim, 'anims': anims})

        mesh = {
            'name': name,
            'parent_name': parent_name,
            'textures': textures,             # solo v2.3 (paths directos)
            'texture_indices': texture_indices,  # v2.2: índices a self.textures
            'matrix': matrix,
            'position_': position_,
            'vertices': vertices,
            'texcoords': texcoords,
            'texcoord_colors': texcoord_colors,
            'faces': faces,
            'scale_keyframes': scale_keyframes,
            'rot_keyframes': rot_keyframes,
            'pos_keyframes': pos_keyframes,
            'texture_keyframe_groups': texture_keyframe_groups,
            'parent': None,
            'children': [],
        }
        return mesh, off

    def resolve_hierarchy(self):
        """Conecta cada malla con su padre/hijos por nombre (in-place)."""
        by_name = {m['name']: m for m in self.meshes}
        for mesh in self.meshes:
            if mesh['parent_name']:
                parent = by_name.get(mesh['parent_name'])
                if parent is not None:
                    mesh['parent'] = parent
                    parent['children'].append(mesh)

    def get_root_meshes(self):
        """Devuelve la lista de mallas raíz (puede haber más de una en RSM2)."""
        return [m for m in self.meshes if m['name'] in self.main_mesh_names]

    def compute_world_matrices(self, frame=0):
        """
        Calcula la matriz 4x4 final (mundo) de cada malla, siguiendo el
        algoritmo documentado para RSM2 v2.2+ (informe técnico basado en
        el parser de referencia de GRF Editor / Tokeiburu, sección 5):

            MeshMatrix = TransformationMatrix
            si hay padre: MeshMatrix @= inversa(TransformationMatrix del padre)

            offset = Position_ (si es raíz)
                   = inversa(matriz del padre) @ (Position_ - Position_ del padre)  (si no)

            MeshMatrixSelf = MeshMatrix con esa traslación

            subir la cadena de padres multiplicando por el MeshMatrix
            (rotación, sin traslación) de cada ancestro, y al final sumar
            el offset final ya calculado del padre inmediato.

        Devuelve {nombre_malla: matriz 4x4 numpy}.

        LIMITACIÓN: solo implementa el caso SIN animación (usa
        TransformationMatrix/Position_ estáticos, ignora scale/rot/pos
        keyframes incluso si existen). Validado contra un archivo real
        de 2 mallas en jerarquía (root sin transformación + hijo con
        rotación y offset reales) — el resultado para la malla hija
        coincidió exactamente con lo esperado a mano. No probado con
        jerarquías de más de 2 niveles ni con animación.
        """
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError(
                "compute_world_matrices requiere numpy (pip install numpy)"
            )

        def mat3_to_mat4(m9):
            a = np.identity(4, dtype=np.float64)
            a[0, 0:3] = m9[0:3]
            a[1, 0:3] = m9[3:6]
            a[2, 0:3] = m9[6:9]
            return a

        local_matrix = {}   # MeshMatrix (rotación relativa al padre, sin traslación)
        world_matrix = {}   # MeshMatrixSelf (final, con traslación de mundo)

        # Procesar en orden padres-antes-que-hijos (BFS desde las raíces)
        by_name = {mm['name']: mm for mm in self.meshes}
        roots = [mm for mm in self.meshes if mm['parent'] is None]
        queue = list(roots)
        order = []
        seen = set()
        while queue:
            mesh = queue.pop(0)
            if mesh['name'] in seen:
                continue
            seen.add(mesh['name'])
            order.append(mesh)
            queue.extend(mesh['children'])
        # por si quedó alguna malla huérfana sin resolver bien la jerarquía
        for mm in self.meshes:
            if mm['name'] not in seen:
                order.append(mm)

        for mesh in order:
            name = mesh['name']
            parent = mesh['parent']
            transform = mat3_to_mat4(mesh['matrix'])

            if parent is None:
                local_matrix[name] = transform
                position = np.array(mesh['position_'], dtype=np.float64)
            else:
                parent_transform = mat3_to_mat4(parent['matrix'])
                parent_transform_inv = np.linalg.inv(parent_transform)
                local_matrix[name] = transform @ parent_transform_inv

                pos_diff = np.array(mesh['position_'], dtype=np.float64) - \
                    np.array(parent['position_'], dtype=np.float64)
                pos_diff_h = np.array([pos_diff[0], pos_diff[1], pos_diff[2], 0.0])
                position = (parent_transform_inv @ pos_diff_h)[:3]

            mesh_self = local_matrix[name].copy()
            mesh_self[0:3, 3] = position

            # Subir la cadena de padres componiendo rotaciones locales
            m = mesh
            result = mesh_self
            while m['parent'] is not None:
                m = m['parent']
                result = result @ local_matrix[m['name']]

            if parent is not None:
                result[0:3, 3] += world_matrix[parent['name']][0:3, 3]

            world_matrix[name] = result

        return world_matrix




def resolve_ro_texture_path(tex_name, roots, basename_index=None):
    """Resuelve rutas de textura del .gnd contra raíces estilo GRF extraído."""
    if not tex_name:
        return None
    if isinstance(roots, (str, bytes)):
        roots = [roots]
    roots = [r for r in roots if r]
    norm = tex_name.replace('\\', '/').lstrip('/')
    base = norm.split('/')[-1]
    base_lower = base.lower()
    candidates = []
    for root in roots:
        if not root:
            continue
        candidates.extend([
            os.path.join(root, norm),
            os.path.join(root, "data", "texture", norm),
            os.path.join(root, "data", "texture", "map", norm),
            os.path.join(root, "texture", norm),
            os.path.join(root, "data", "texture", base),
            os.path.join(root, "data", "texture", "map", base),
            os.path.join(root, base),
        ])
    for p in candidates:
        if os.path.isfile(p):
            return p
    for p in candidates:
        parent = os.path.dirname(p)
        name = os.path.basename(p)
        if not parent or not os.path.isdir(parent):
            continue
        try:
            lower_map = {f.lower(): f for f in os.listdir(parent)}
            if name.lower() in lower_map:
                found = os.path.join(parent, lower_map[name.lower()])
                if os.path.isfile(found):
                    return found
        except OSError:
            pass
    if basename_index is not None and base_lower:
        if base_lower in basename_index:
            return basename_index[base_lower]
        indexed_marker = basename_index.setdefault("__roots__", set())
        for root in roots:
            abs_root = os.path.abspath(root)
            if abs_root in indexed_marker or not os.path.isdir(root):
                continue
            indexed_marker.add(abs_root)
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in filenames:
                    key = fname.lower()
                    if key not in basename_index:
                        basename_index[key] = os.path.join(dirpath, fname)
        return basename_index.get(base_lower)
    return None


class ROMapViewer:
    """Visualizador interactivo de mapas de Ragnarok Online."""
    
    def __init__(self, gat_path, rsw_path=None, gnd_path=None, tile_size=8,
                 gat_strict=False, texture_dir=None):
        pygame.init()
        try:
            self.screen_width = 1280
            self.screen_height = 720
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption("Ragnarok Online Map Viewer — Completo")

            map_dir = os.path.dirname(os.path.abspath(gat_path))
            self.texture_roots = []
            for candidate in (
                texture_dir,
                map_dir,
                os.path.join(map_dir, "data"),
                os.path.join(map_dir, ".."),
                os.path.join(map_dir, "..", "data"),
            ):
                if candidate and os.path.isdir(candidate):
                    abs_c = os.path.abspath(candidate)
                    if abs_c not in self.texture_roots:
                        self.texture_roots.append(abs_c)
            self._tex_basename_index = {}

            # Cargar datos
            self.gat = GATParser(gat_path, strict=gat_strict)

            self.rsw = None
            if rsw_path and os.path.exists(rsw_path):
                self.rsw = RSWParser(rsw_path)

            self.gnd = None
            if gnd_path and os.path.exists(gnd_path):
                if not PIL_AVAILABLE:
                    print(
                        "[VIEWER] Advertencia: se indicó un .gnd pero Pillow "
                        "no está instalado (pip install Pillow). Se ignorará "
                        "el GND y se usarán solo los colores del GAT."
                    )
                else:
                    self.gnd = GNDParser(gnd_path)
                    if self.gnd.error:
                        print(f"[VIEWER] Advertencia: no se pudo usar el GND: {self.gnd.error}")
                        self.gnd = None

            # Determinar nivel de agua (prioridad: GND > RSW)
            self.water_level = None
            if self.gnd and self.gnd.water_level is not None:
                self.water_level = self.gnd.water_level
            elif self.rsw and self.rsw.water_level is not None:
                self.water_level = self.rsw.water_level

            # Configuración de visualización
            self.tile_size = tile_size
            self.zoom = 1.0
            self.min_zoom = 0.1
            self.max_zoom = 10.0

            # Cámara
            self.camera_x = self.gat.width / 2
            self.camera_y = self.gat.height / 2

            # Estado del mouse
            self.dragging = False
            self.last_mouse_pos = (0, 0)
            self.hover_tile = None
            # Capas (teclas G / B / M / H)
            self.show_walkable = False   # G — tinte rojo no-caminable
            self.show_silhouette = False # B — silueta caminable en negro
            self.show_models = False     # M — marcadores de modelos
            self.show_edges = True       # H — borde 1px de tile (estilo GRF)

            # Fuentes
            self.font = pygame.font.SysFont("consolas", 14)
            self.font_large = pygame.font.SysFont("consolas", 16, bold=True)
            self.font_small = pygame.font.SysFont("consolas", 10)

            # Precalcular colores de tiles
            self.tile_colors = None
            self._precompute_tile_colors()

            # Colores de UI
            self.ui_bg = (30, 30, 30, 220)
            self.ui_border = (100, 100, 100)
            self.ui_text = (255, 255, 255)

            # Reloj
            self.clock = pygame.time.Clock()
            self.fps = 60

            # Caché de texturas: guarda directamente pygame.Surface YA
            # escaladas al tamaño de tile actual (tile_px_ceil), para no
            # repetir el resize (PIL) + conversión (pygame) en cada frame.
            # Clave: (tex_name, tile_px_ceil).
            self.texture_cache = {}
            # Caché de la imagen PIL cruda (sin escalar), separada de la
            # anterior: se reusa aunque cambie el zoom.
            self._texture_source_cache = {}

            log_info("[VIEWER] Listo.")
        except Exception:
            # Si algo falla durante la inicialización (archivo corrupto,
            # etc.), liberamos los recursos de Pygame antes de propagar
            # el error para no dejar la ventana/subsistema de video colgado.
            pygame.quit()
            raise
    
    def _precompute_tile_colors(self):
        """Precalcula colores de tiles (para fallback si no hay GND)."""
        log_info("[VIEWER] Precalculando colores de tiles...")
        self.tile_colors = [[None] * self.gat.width for _ in range(self.gat.height)]
        
        for y in range(self.gat.height):
            for x in range(self.gat.width):
                block = self.gat.get_block(x, y)
                level = self.water_level
                if self.gnd is not None and self.gnd.water_zones:
                    level = self.gnd.water_level_at(x // 2, y // 2)   # cada cubo GND = 2x2 tiles GAT
                water_state = self.gat.get_water_state(block, level)
                self.tile_colors[y][x] = self.gat.get_block_color(block, water_state)
        
        log_info(f"[VIEWER] Listo: {self.gat.width}x{self.gat.height} tiles")
    
    def get_visible_tile_range(self):
        """Calcula el rango de tiles visibles en pantalla."""
        # Esquina superior izquierda
        screen_x, screen_y = 0, 0
        map_x1, map_y1 = self.screen_to_map(screen_x, screen_y)
        
        # Esquina inferior derecha
        screen_x, screen_y = self.screen_width, self.screen_height
        map_x2, map_y2 = self.screen_to_map(screen_x, screen_y)
        
        min_x = max(0, int(math.floor(min(map_x1, map_x2))))
        max_x = min(self.gat.width - 1, int(math.ceil(max(map_x1, map_x2))))
        min_y = max(0, int(math.floor(min(map_y1, map_y2))))
        max_y = min(self.gat.height - 1, int(math.ceil(max(map_y1, map_y2))))
        
        # Margen de seguridad
        margin = 2
        return max(0, min_x - margin), max(0, min_y - margin), \
               min(self.gat.width - 1, max_x + margin), min(self.gat.height - 1, max_y + margin)
    
    def screen_to_map(self, screen_x, screen_y):
        """Convierte coordenadas de pantalla a coordenadas del mapa."""
        map_pixel_w = self.gat.width * self.tile_size * self.zoom
        map_pixel_h = self.gat.height * self.tile_size * self.zoom
        
        offset_x = (self.screen_width - map_pixel_w) / 2
        offset_y = (self.screen_height - map_pixel_h) / 2
        
        cam_offset_x = (self.camera_x - self.gat.width / 2) * self.tile_size * self.zoom
        cam_offset_y = (self.camera_y - self.gat.height / 2) * self.tile_size * self.zoom
        
        map_x = (screen_x - offset_x + cam_offset_x) / (self.tile_size * self.zoom)
        map_y = (screen_y - offset_y + cam_offset_y) / (self.tile_size * self.zoom)
        
        return map_x, map_y
    
    def map_to_screen(self, map_x, map_y):
        """Convierte coordenadas del mapa a coordenadas de pantalla."""
        map_pixel_w = self.gat.width * self.tile_size * self.zoom
        map_pixel_h = self.gat.height * self.tile_size * self.zoom
        
        offset_x = (self.screen_width - map_pixel_w) / 2
        offset_y = (self.screen_height - map_pixel_h) / 2
        
        cam_offset_x = (self.camera_x - self.gat.width / 2) * self.tile_size * self.zoom
        cam_offset_y = (self.camera_y - self.gat.height / 2) * self.tile_size * self.zoom
        
        screen_x = map_x * self.tile_size * self.zoom - cam_offset_x + offset_x
        screen_y = map_y * self.tile_size * self.zoom - cam_offset_y + offset_y
        
        return screen_x, screen_y
    
    def load_texture(self, tex_name):
        """Carga (y cachea) la imagen PIL cruda de una textura desde disco."""
        if tex_name in self._texture_source_cache:
            return self._texture_source_cache[tex_name]
        path = resolve_ro_texture_path(
            tex_name,
            getattr(self, "texture_roots", []),
            getattr(self, "_tex_basename_index", None),
        )
        if path:
            try:
                img = Image.open(path)
                img.load()
                self._texture_source_cache[tex_name] = img
                return img
            except Exception:
                pass
        self._texture_source_cache[tex_name] = None
        return None

    def get_tile_texture_surface(self, tex_name, tile_px_ceil):
        """
        Devuelve una pygame.Surface YA escalada y convertida para este
        tex_name + tamaño de tile, usando caché. El resize (PIL) y la
        conversión a pygame.Surface son costosos en CPU, así que se hacen
        UNA sola vez por combinación (textura, tamaño) y no en cada frame.
        """
        cache_key = (tex_name, tile_px_ceil)
        if cache_key in self.texture_cache:
            return self.texture_cache[cache_key]

        surf = None
        tex_img = self.load_texture(tex_name)
        if tex_img is not None:
            try:
                scaled = tex_img.resize((tile_px_ceil, tile_px_ceil), Image.NEAREST)
                mode = 'RGB' if scaled.mode == 'RGB' else 'RGBA'
                if mode == 'RGBA' and scaled.mode != 'RGBA':
                    scaled = scaled.convert('RGBA')
                elif mode == 'RGB' and scaled.mode != 'RGB':
                    scaled = scaled.convert('RGB')
                surf = pygame.image.fromstring(scaled.tobytes(), scaled.size, mode).convert_alpha()
            except Exception:
                surf = None

        self.texture_cache[cache_key] = surf  # cachea también el None (evita reintentos)
        return surf

    def draw_map(self):
        """Texturas GND + capas opcionales (silueta / walkable / modelos)."""
        min_x, min_y, max_x, max_y = self.get_visible_tile_range()
        tile_px = self.tile_size * self.zoom
        tile_px_ceil = max(1, math.ceil(tile_px))
        use_gnd = self.gnd is not None

        # Tinte rojo semi-transparente (estilo GRF "gat types")
        OVERLAY = {
            1: (220, 80, 80, 100),
            2: (150, 35, 35, 120),
            4: (150, 35, 35, 120),
            5: (220, 80, 80, 100),
            6: (220, 80, 80, 100),
        }
        # Tipos caminables para silueta
        WALKABLE_TYPES = (0, 3)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                sx, sy = self.map_to_screen(x, y)
                rect = pygame.Rect(int(sx), int(sy), tile_px_ceil, tile_px_ceil)
                block = self.gat.get_block(x, y)
                btype = block['type'] if block else 0

                if self.show_silhouette:
                    # Miniatura estilo: caminable = negro, no caminable = gris claro
                    if btype in WALKABLE_TYPES:
                        color = (20, 20, 20)
                    else:
                        color = (170, 170, 170)
                    pygame.draw.rect(self.screen, color, rect)
                else:
                    if use_gnd:
                        tex_name = self.gnd.get_texture_name(x, y)
                        surf = self.get_tile_texture_surface(tex_name, tile_px_ceil) if tex_name else None
                        if surf is not None:
                            self.screen.blit(surf, (int(sx), int(sy)))
                        else:
                            pygame.draw.rect(self.screen, self.tile_colors[y][x], rect)
                    else:
                        pygame.draw.rect(self.screen, self.tile_colors[y][x], rect)

                    if self.show_walkable:
                        ov = OVERLAY.get(btype)
                        if ov is not None:
                            tint = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
                            tint.fill(ov)
                            self.screen.blit(tint, rect.topleft)

                # Borde 1px sutil (como GRF): solo con zoom cercano, se pierde al alejar
                if self.show_edges and not self.show_silhouette and tile_px >= 6:
                    # Oscurecer 1px del borde superior e izquierdo
                    edge = (0, 0, 0, 50)
                    edge_s = pygame.Surface((rect.w, 1), pygame.SRCALPHA)
                    edge_s.fill(edge)
                    self.screen.blit(edge_s, (rect.x, rect.y))
                    edge_s2 = pygame.Surface((1, rect.h), pygame.SRCALPHA)
                    edge_s2.fill(edge)
                    self.screen.blit(edge_s2, (rect.x, rect.y))

        if self.rsw and self.rsw.models:
            self.draw_models()

    def draw_models(self):
        """Marcadores RSW: pos.x/pos.z → tiles GAT."""
        if not getattr(self, "show_models", False):
            return
        if self.gnd is not None:
            gw, gh = self.gnd.width, self.gnd.height
        else:
            gw = max(1, self.gat.width // 2)
            gh = max(1, self.gat.height // 2)
        min_x, min_y, max_x, max_y = self.get_visible_tile_range()
        show_labels = self.zoom >= 1.5
        radius = max(2, min(5, int(2.5 * self.zoom)))
        for model in self.rsw.models:
            x_3d, _y_3d, z_3d = model['pos']
            # Origen RSW en el centro; GAT: +X este, +Y sur en pantalla
            # (mismo sentido que capas G/B — antes Z estaba invertida)
            tx = gw + x_3d / 5.0
            ty = gh + z_3d / 5.0
            if tx < min_x or tx > max_x or ty < min_y or ty > max_y:
                continue
            sx, sy = self.map_to_screen(tx, ty)
            pygame.draw.circle(self.screen, (255, 200, 100), (int(sx), int(sy)), radius)
            if radius >= 4:
                pygame.draw.circle(self.screen, (255, 255, 255), (int(sx), int(sy)), radius, 1)
            if show_labels:
                name = os.path.basename(model.get('name') or model.get('file_name') or '')[:18]
                if name:
                    label = self.font_small.render(name, True, (255, 255, 200))
                    self.screen.blit(label, (int(sx + radius + 3), int(sy - 6)))

    def draw_grid(self):
        """Ya no dibujamos malla pesada: los bordes van en draw_map (1px).
        Se mantiene por compatibilidad; no hace nada."""
        return

    def _grid_interval(self):
        if self.zoom < 0.15:
            return 50
        elif self.zoom < 0.4:
            return 20
        elif self.zoom < 1.0:
            return 10
        elif self.zoom < 3.0:
            return 5
        return 1
    
    def draw_hover_highlight(self):
        """Resalta el tile bajo el cursor."""
        if self.hover_tile is None:
            return
        
        tx, ty = self.hover_tile
        if 0 <= tx < self.gat.width and 0 <= ty < self.gat.height:
            sx, sy = self.map_to_screen(tx, ty)
            tile_px = self.tile_size * self.zoom
            rect = pygame.Rect(int(sx), int(sy), math.ceil(tile_px), math.ceil(tile_px))
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)
    
    def draw_ui(self):
        """Dibuja la interfaz de usuario (info, leyenda)."""
        # Fondo semitransparente
        s = pygame.Surface((340, 250), pygame.SRCALPHA)
        s.fill((30, 30, 30, 220))
        self.screen.blit(s, (10, 10))
        
        # Info del mapa
        y = 18
        self._draw_text(self.screen, f"Mapa: {os.path.basename(self.gat.filepath)}", (20, y), (255, 255, 255))
        y += 20
        self._draw_text(self.screen, f"Tiles: {self.gat.width}x{self.gat.height}", (20, y), (200, 200, 200))
        y += 18
        
        if self.gnd:
            gv = self.gnd.version
            gnd_ver = f"{gv[0]}.{gv[1]}" if isinstance(gv, (tuple, list)) else str(gv)
            self._draw_text(self.screen, f"GND: v{gnd_ver}", (20, y), (100, 200, 100))
            y += 18
        
        if self.rsw:
            self._draw_text(self.screen, f"RSW: v{self.rsw.version[0]}.{self.rsw.version[1]}", (20, y), (100, 200, 200))
            y += 18
            self._draw_text(self.screen, f"Modelos: {len(self.rsw.models)}", (20, y), (200, 200, 100))
            y += 18
        
        if self.water_level is not None:
            self._draw_text(self.screen, f"Nivel agua: {self.water_level:.2f}", (20, y), (100, 200, 255))
            y += 18
        
        self._draw_text(self.screen, f"Zoom: {self.zoom:.1f}x", (20, y), (200, 200, 200))
        y += 18
        self._draw_text(
            self.screen,
            f"G tint:{'ON' if self.show_walkable else 'off'}  "
            f"B mask:{'ON' if self.show_silhouette else 'off'}  "
            f"M models:{'ON' if self.show_models else 'off'}  "
            f"H edge:{'ON' if self.show_edges else 'off'}",
            (20, y), (160, 200, 160),
        )
        y += 18
        
        if self.hover_tile:
            tx, ty = self.hover_tile
            block = self.gat.get_block(tx, ty)
            if block:
                block_type = block['type']
                self._draw_text(self.screen, f"Tile: ({tx}, {ty})", (20, y), (255, 255, 200))
                y += 18
                self._draw_text(self.screen, f"Tipo: {BLOCK_NAMES.get(block_type, 'Desconocido')}", (20, y), (200, 200, 200))
                y += 18
                self._draw_text(self.screen, f"Altura: {block['avg_height']:.2f}", (20, y), (200, 200, 200))
        
        # Leyenda (esquina inferior derecha)
        self.draw_legend()
    
    def _draw_text(self, screen, text, pos, color=(255, 255, 255)):
        """Dibuja texto en la pantalla."""
        surf = self.font.render(text, True, color)
        screen.blit(surf, pos)
    
    def draw_legend(self):
        """Dibuja la leyenda de colores."""
        legend_x = self.screen_width - 180
        legend_y = self.screen_height - 200
        
        # Fondo
        s = pygame.Surface((170, 190), pygame.SRCALPHA)
        s.fill((30, 30, 30, 220))
        self.screen.blit(s, (legend_x, legend_y))
        
        self._draw_text(self.screen, "Leyenda", (legend_x + 10, legend_y + 8), (255, 255, 255))
        
        items = [
            ("Caminable", BLOCK_COLORS[0]),
            ("No caminable", BLOCK_COLORS[1]),
            ("Agua no cam.", BLOCK_COLORS[2]),
            ("Agua cam.", BLOCK_COLORS[3]),
            ("Agua (snip)", BLOCK_COLORS[4]),
            ("Acantilado", BLOCK_COLORS[5]),
            ("Acantilado (no snip)", BLOCK_COLORS[6]),
        ]
        
        y_offset = legend_y + 30
        for name, color in items:
            pygame.draw.rect(self.screen, color, (legend_x + 10, y_offset, 14, 14))
            pygame.draw.rect(self.screen, (255, 255, 255), (legend_x + 10, y_offset, 14, 14), 1)
            self._draw_text(self.screen, name, (legend_x + 30, y_offset + 1), (200, 200, 200))
            y_offset += 20
    
    def handle_events(self):
        """Maneja eventos de Pygame."""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    return False
                elif event.key == K_r:
                    self.camera_x = self.gat.width / 2
                    self.camera_y = self.gat.height / 2
                    self.zoom = 1.0
                elif event.key == K_UP or event.key == K_w:
                    self.camera_y -= 20 / self.zoom
                elif event.key == K_DOWN or event.key == K_s:
                    self.camera_y += 20 / self.zoom
                elif event.key == K_LEFT or event.key == K_a:
                    self.camera_x -= 20 / self.zoom
                elif event.key == K_RIGHT or event.key == K_d:
                    self.camera_x += 20 / self.zoom
                elif event.key == K_g:
                    self.show_walkable = not self.show_walkable
                    if self.show_walkable:
                        self.show_silhouette = False
                elif event.key == K_b:
                    self.show_silhouette = not self.show_silhouette
                    if self.show_silhouette:
                        self.show_walkable = False
                elif event.key == K_m:
                    self.show_models = not self.show_models
                elif event.key == K_h:
                    self.show_edges = not self.show_edges
            
            elif event.type == VIDEORESIZE:
                self.screen_width, self.screen_height = event.size
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height),
                    pygame.RESIZABLE
                )
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Click izquierdo
                    self.dragging = True
                    self.last_mouse_pos = event.pos
                elif event.button == 3:  # Click derecho
                    map_x, map_y = self.screen_to_map(event.pos[0], event.pos[1])
                    if 0 <= map_x < self.gat.width and 0 <= map_y < self.gat.height:
                        self.camera_x = map_x
                        self.camera_y = map_y
                        self.hover_tile = (int(map_x), int(map_y))
                elif event.button == 4:  # Scroll up
                    self.zoom = min(self.max_zoom, self.zoom * 1.15)
                elif event.button == 5:  # Scroll down
                    self.zoom = max(self.min_zoom, self.zoom / 1.15)
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            
            elif event.type == MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.camera_x -= dx / (self.tile_size * self.zoom)
                    self.camera_y -= dy / (self.tile_size * self.zoom)
                    self.last_mouse_pos = event.pos
                
                # Actualizar tile bajo el cursor
                map_x, map_y = self.screen_to_map(event.pos[0], event.pos[1])
                if 0 <= map_x < self.gat.width and 0 <= map_y < self.gat.height:
                    self.hover_tile = (int(map_x), int(map_y))
                else:
                    self.hover_tile = None
        
        return True
    
    def run(self):
        """Bucle principal del visualizador."""
        running = True
        
        while running:
            running = self.handle_events()
            
            # Dibujar
            self.screen.fill((20, 20, 25))
            self.draw_grid()
            self.draw_map()
            self.draw_hover_highlight()
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(self.fps)
        
        pygame.quit()
        log_info("[VIEWER] Cerrado.")

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print(" Ragnarok Online Map Viewer — Completo")
        print("=" * 60)
        print()
        print("Uso:")
        print("  python ro_map_viewer.py <mapa.gat> [mapa.rsw] [mapa.gnd]")
        print()
        print("Dependencias (instalar una vez):")
        print("  pip install -r requirements.txt")
        print()
        print("Ejemplos:")
        print("  python ro_map_viewer.py prontera.gat prontera.rsw prontera.gnd")
        print("  python ro_map_viewer.py prontera.gat")
        print()
        print("Controles:")
        print("  Click izquierdo + arrastrar → Mover cámara")
        print("  Rueda del mouse → Zoom in/out")
        print("  Click derecho → Centrar en tile")
        print("  Flechas / WASD → Mover cámara")
        print("  R → Resetear vista")
        print("  ESC / Q → Salir")
        print()
        return
    
    gat_path = sys.argv[1]
    rsw_path = sys.argv[2] if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else None
    gnd_path = sys.argv[3] if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None

    if not os.path.exists(gat_path):
        print(f"Error: No se encontró el archivo: {gat_path}")
        sys.exit(1)

    try:
        viewer = ROMapViewer(gat_path, rsw_path, gnd_path)
    except GATFileError as e:
        print(f"Error: El archivo .gat no se pudo cargar: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        viewer.run()
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main()