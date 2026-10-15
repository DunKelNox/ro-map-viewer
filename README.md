# RO Map Viewer

**Visores 2D y 3D de mapas de Ragnarok Online** (`.gat` / `.gnd` / `.rsw` / `.rsm` / `.rsm2`)
escritos en Python + OpenGL, con parsers validados contra archivos reales y
colocación de modelos **idéntica a BrowEdit3**.

> 🧩 **Proyecto libre** — podés usarlo, modificarlo, forkearlo y redistribuirlo
> bajo licencia MIT. Se agradece mención o link al repo original.

## ✨ Características

### Visor 2D (`ro_map_viewer.py`)
- Renderiza la grilla del `.gat` coloreada por tipo de bloque (caminable, agua, acantilado, etc.).
- Superpone las texturas del `.gnd` cuando está disponible.
- Detecta el nivel de agua desde el `.gnd` (v1.8+) o el `.rsw` como fallback.
- Muestra los modelos del `.rsw` como marcadores con nombre.
- Overlays toggleables: caminable (G), silueta (B), modelos (M), bordes (H).

### Visor 3D del mapa (`terrain_viewer_3d.py`)
- Terreno GND con **caras top + north + east** (paredes verticales).
- **Lightmaps** del GND (sombra ambiental horneada) con atlas y padding para evitar costuras.
- **Agua animada** en 4 modos: sin agua / plana / animada con texturas `waterNNN.jpg` (elegible en el launcher). / animada sin texturas
- **Modelos RSM/RSM2** colocados con la misma matemática que BrowEdit3 (bounding box real, jerarquías, animaciones de malla y de texturas).
- Culling por distancia y cono frontal para mapas grandes.
- Brújula con ejes cardinales (C) y nudge global para calibración.

### Visor de modelo suelto (`rsm_viewer_3d.py`)
- Carga un solo `.rsm` / `.rsm2` y lo renderiza con animación de malla y textura.
- Útil para debug de modelos antes de meterlos en un mapa.

### Lanzadores gráficos (`map_launcher.py`, `rsm_launcher.py`)
- Interfaz Tkinter nativa, sin consola.
- Autodetecta `.rsw` / `.gnd` a partir del `.gat`.
- Busca automáticamente carpetas `data/texture` y `data/model` cercanas.
- Autodetecta la carpeta de agua (`water000.jpg … water031.jpg`).

---

## 📦 Instalación

Requiere **Python 3.9+**.

```bash
git clone https://github.com/<tu-usuario>/ro-map-viewer.git
cd ro-map-viewer
pip install -r requirements.txt
