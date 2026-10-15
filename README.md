# RO Map Viewer

Visores de mapas de Ragnarok Online (`.gat`, `.gnd`, `.rsw`, `.rsm` / `.rsm2`).
Parsers validados contra archivos reales; colocación de modelos alineada a
BrowEdit3 (`rsm_placement.py`).

## Instalación

```bash
pip install -r requirements.txt
```

Python 3.9+. En escritorio normal (Windows / macOS / Linux con GUI) no hace
falta nada extra.

## Archivos

| Archivo | Rol |
|---------|-----|
| `map_launcher.py` | **Lanzador gráfico** del mapa (2D y terreno 3D). Punto de entrada recomendado. |
| `rsm_launcher.py` | Lanzador gráfico del visor de un solo modelo `.rsm` / `.rsm2`. |
| `ro_map_viewer.py` | Visor 2D + parsers (`GAT`, `RSW`, `GND`, `RSM`, `RSM2`). |
| `terrain_viewer_3d.py` | Visor 3D de mapa: terreno, agua, lightmaps, GAT, modelos RSM/RSM2. |
| `rsm_viewer_3d.py` | Visor 3D de un modelo suelto; **mismas matrices** que el mapa (`rsm_placement`). |
| `rsm_placement.py` | Colocación de instancias RSW idéntica a BrowEdit3 (bbox real + matrices de malla). |
| `mat4.py` | Matrices 4×4 (sin PyGLM). |
| `rolog.py` | Log de consola (`info` solo con `RO_VERBOSE=1`; `warn` siempre). |
| `requirements.txt` | Dependencias. |

## Qué necesitás del GRF

Para un mapa (ej. `aldebaran`):

- `aldebaran.gat`, `aldebaran.gnd`, `aldebaran.rsw` (mismo nombre base)
- Texturas del terreno (`data/texture/...`)
- Modelos (`data/model/...` → `.rsm` / `.rsm2`) y sus texturas

El launcher propone `.rsw` / `.gnd` al elegir el `.gat` y busca carpetas `data/` cercanas.

## Cómo usar

### Lanzador de mapa (recomendado)

```bash
python map_launcher.py
```

1. Elegí el `.gat` (y opcionalmente `.rsw`, `.gnd`, carpeta de texturas).
2. **Abrir visor 2D** o **Abrir terreno 3D**.

### Lanzador de modelo RSM

```bash
python rsm_launcher.py
```

### Línea de comandos

```bash
# 2D
python ro_map_viewer.py mapa.gat [mapa.rsw] [mapa.gnd]

# Terreno 3D + modelos del RSW
python terrain_viewer_3d.py mapa.gnd [mapa.rsw] [carpeta_texturas]

# Un solo modelo
python rsm_viewer_3d.py modelo.rsm [carpeta_texturas]
```

### Consola más verbosa

```bash
# Windows (cmd)
set RO_VERBOSE=1
python map_launcher.py

# PowerShell
$env:RO_VERBOSE=1
python map_launcher.py

# Linux / macOS
RO_VERBOSE=1 python map_launcher.py
```

## Controles (resumen)

**2D** (`ro_map_viewer` / desde el launcher)

| Tecla / ratón | Acción |
|---------------|--------|
| Arrastrar | Mover |
| Rueda | Zoom |
| Click derecho | Centrar en tile |
| G / B / M / H | Overlay caminable / silueta / marcadores / bordes |
| ESC | Salir |

**3D mapa** (`terrain_viewer_3d`)

| Tecla / ratón | Acción |
|---------------|--------|
| Click izq. + arrastrar | Pan |
| Click der. + arrastrar | Orbitar |
| Rueda | Zoom (máx. 1500) |
| G | Overlay caminable |
| M | Marcadores RSW |
| R | Modelos RSM on/off |
| A | Animación RSM |
| S | Lightmaps (sombras de terreno) |
| C | Brújula |
| W | Muros del terreno |
| J L / I K | Nudge global de modelos X/Z (±1; Shift = ±5) |
| Home | Reset nudge |
| { } | Distancia de dibujo RSM |
| ESC | Salir |

**3D modelo suelto** (`rsm_viewer_3d`)

| Tecla / ratón | Acción |
|---------------|--------|
| Click + arrastrar | Orbitar |
| Rueda | Zoom |
| Espacio | Pausa animación |
| [ ] | Velocidad de animación |
| R | Reiniciar animación |
| ESC | Salir |

## Colocación de modelos (RSM)

La posición en el mapa **no** usa el origen del nodo raíz del `.rsm`.
Sigue a BrowEdit3 (`rsm_placement.py`):

1. Matrices de malla (`calcMatrix1` / `calcMatrix2`) y jerarquía.
2. Bounding box **real** del modelo completo (t = 0).
3. Instancia RSW: traslación/rotación/escala del `.rsw` + centrado por bbox
   (XZ) y apoyo en el punto más bajo (Y).

Soporta **RSM1** y **RSM2** (incl. animación de mallas cuando el archivo la trae).

## Limitaciones conocidas

- Agua **multi-zona** del GND: el parser puede leer zonas; el render 3D aún
  usa un plano al nivel global de agua.
- Algunas versiones intermedias de RSW/GND sin muestras reales pueden requerir
  ajustes de offset de header.
- Efectos de partículas del cliente (`.lub` EffectTool) no se renderizan.

## Dependencias

```
pygame>=2.1
Pillow>=9.0
numpy>=1.24
PyOpenGL>=3.1
PyOpenGL_accelerate>=3.1
```
