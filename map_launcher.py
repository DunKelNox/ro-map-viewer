#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanzador gráfico simple para los visores de mapa de Ragnarok Online.
Pide .gat / .rsw / .gnd y la carpeta de texturas con diálogos nativos.

  - Visor 2D  → ro_map_viewer.ROMapViewer
  - Terreno 3D → terrain_viewer_3d.TerrainViewer3D
"""
import os
import re
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    print("tkinter no está disponible en este Python.")
    sys.exit(1)


class MapLauncherApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RO Map Viewer — Lanzador")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e24")

        self.gat_path = tk.StringVar()
        self.rsw_path = tk.StringVar()
        self.gnd_path = tk.StringVar()
        self.tex_dir = tk.StringVar()
        self.water_dir = tk.StringVar()
        self.width_var = tk.StringVar(value="1024")
        self.height_var = tk.StringVar(value="600")

        self._build_ui()
        self._center_window(720, 480)

    def _center_window(self, w, h):
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        pad = {"padx": 16, "pady": 5}
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", background="#1e1e24", foreground="#e0e0e0",
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#1e1e24", foreground="#7ec8ff",
                        font=("Segoe UI", 14, "bold"))
        style.configure("Hint.TLabel", background="#1e1e24", foreground="#888888",
                        font=("Segoe UI", 8))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground="#2a2a32", foreground="#ffffff")
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("TFrame", background="#1e1e24")

        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Visor de mapas RO", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", **pad
        )
        ttk.Label(
            frm,
            text="Elegí los archivos del mapa (.gat obligatorio). "
                 "Al elegir el .gat se proponen .rsw y .gnd.",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 8))

        # --- GAT (obligatorio) ---
        ttk.Label(frm, text="Archivo .gat *").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.gat_path, width=40).grid(
            row=2, column=1, sticky="ew", **pad
        )
        ttk.Button(frm, text="Buscar…", command=self._pick_gat, width=10).grid(
            row=2, column=2, **pad
        )

        # --- RSW ---
        ttk.Label(frm, text="Archivo .rsw").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.rsw_path, width=40).grid(
            row=3, column=1, sticky="ew", **pad
        )
        ttk.Button(frm, text="Buscar…", command=self._pick_rsw, width=10).grid(
            row=3, column=2, **pad
        )

        # --- GND ---
        ttk.Label(frm, text="Archivo .gnd").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.gnd_path, width=40).grid(
            row=4, column=1, sticky="ew", **pad
        )
        ttk.Button(frm, text="Buscar…", command=self._pick_gnd, width=10).grid(
            row=4, column=2, **pad
        )

        # --- Texturas (para 3D) ---
        ttk.Label(frm, text="Texturas").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.tex_dir, width=40).grid(
            row=5, column=1, sticky="ew", **pad
        )
        ttk.Button(frm, text="Buscar…", command=self._pick_tex, width=10).grid(
            row=5, column=2, **pad
        )
        ttk.Label(
            frm,
            text="Raíz de extracción GRF o carpeta data/ (texturas en data/texture/...). "
                 "Sirve para 2D y 3D.",
            style="Hint.TLabel",
        ).grid(row=6, column=1, columnspan=2, sticky="w", padx=16, pady=(0, 4))

        # --- Agua (texturas water000..031.jpg, opcional, para el agua animada) ---
        ttk.Label(frm, text="Agua (water)").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.water_dir, width=40).grid(
            row=7, column=1, sticky="ew", **pad
        )
        ttk.Button(frm, text="Buscar…", command=self._pick_water, width=10).grid(
            row=7, column=2, **pad
        )
        ttk.Label(
            frm,
            text="water000.jpg … water031.jpg (opcional) · modo RO = tecla 3",
            style="Hint.TLabel",
        ).grid(row=8, column=1, columnspan=2, sticky="w", padx=16, pady=(0, 4))

        # --- Resolución ---
        ttk.Label(frm, text="Resolución").grid(row=9, column=0, sticky="w", **pad)
        res_frm = ttk.Frame(frm)
        res_frm.grid(row=9, column=1, sticky="w", **pad)
        ttk.Entry(res_frm, textvariable=self.width_var, width=6).pack(side=tk.LEFT)
        ttk.Label(res_frm, text=" × ").pack(side=tk.LEFT)
        ttk.Entry(res_frm, textvariable=self.height_var, width=6).pack(side=tk.LEFT)

        # --- Botones de lanzamiento ---
        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=10, column=0, columnspan=3, pady=16)

        ttk.Button(
            btn_frm,
            text="  Abrir visor 2D  ",
            command=self._launch_2d,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            btn_frm,
            text="  Abrir terreno 3D  ",
            command=self._launch_3d,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(btn_frm, text="Salir", command=self.root.destroy).pack(
            side=tk.LEFT, padx=6
        )

        # Hints de controles
        ttk.Label(
            frm,
            text="2D: arrastrar = mover · rueda = zoom · click derecho = centrar · ESC = salir",
            style="Hint.TLabel",
        ).grid(row=11, column=0, columnspan=3, sticky="w", padx=16, pady=(2, 0))
        ttk.Label(
            frm,
            text="3D: click izq. + arrastrar = orbitar · rueda = zoom · 1-4 = agua · C = brújula · A = animar · ESC = salir",
            style="Hint.TLabel",
        ).grid(row=12, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 4))

        frm.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    # Selectores de archivo
    # ------------------------------------------------------------------

    def _suggest_siblings(self, path, ext):
        """Si existe path con otra extensión en la misma carpeta, lo devuelve."""
        base, _ = os.path.splitext(path)
        candidate = base + ext
        return candidate if os.path.isfile(candidate) else None

    def _pick_gat(self):
        path = filedialog.askopenfilename(
            title="Elegir archivo GAT",
            filetypes=[
                ("GAT (altura / walkable)", "*.gat *.GAT"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return
        self.gat_path.set(path)

        # Auto-proponer RSW y GND del mismo nombre base
        if not self.rsw_path.get().strip():
            rsw = self._suggest_siblings(path, ".rsw") or self._suggest_siblings(path, ".RSW")
            if rsw:
                self.rsw_path.set(rsw)
        if not self.gnd_path.get().strip():
            gnd = self._suggest_siblings(path, ".gnd") or self._suggest_siblings(path, ".GND")
            if gnd:
                self.gnd_path.set(gnd)

        # Proponer raíz de texturas: preferir data/ cercana si existe
        if not self.tex_dir.get().strip():
            map_dir = os.path.dirname(path)
            suggested = map_dir
            for candidate in (
                os.path.join(map_dir, "data"),
                os.path.join(map_dir, "..", "data"),
                os.path.abspath(os.path.join(map_dir, "..")),
                map_dir,
            ):
                tex_probe = os.path.join(candidate, "texture")
                data_tex = os.path.join(candidate, "data", "texture")
                if os.path.isdir(tex_probe) or os.path.isdir(data_tex) or (
                    os.path.basename(os.path.abspath(candidate)).lower() == "data"
                    and os.path.isdir(candidate)
                ):
                    suggested = os.path.abspath(candidate)
                    break
            self.tex_dir.set(suggested)
        self._autodetect_water(self.tex_dir.get().strip())

    def _pick_rsw(self):
        path = filedialog.askopenfilename(
            title="Elegir archivo RSW",
            filetypes=[
                ("RSW (world / objetos)", "*.rsw *.RSW"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.rsw_path.set(path)

    def _pick_gnd(self):
        path = filedialog.askopenfilename(
            title="Elegir archivo GND",
            filetypes=[
                ("GND (terreno / texturas)", "*.gnd *.GND"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.gnd_path.set(path)
            if not self.tex_dir.get().strip():
                self.tex_dir.set(os.path.dirname(path))

    def _pick_tex(self):
        path = filedialog.askdirectory(title="Carpeta de texturas (.bmp)")
        if path:
            self.tex_dir.set(path)
            self._autodetect_water(path)

    def _pick_water(self):
        path = filedialog.askdirectory(
            title="Carpeta con las texturas de agua (water000.jpg … water031.jpg)"
        )
        if path:
            self.water_dir.set(path)

    @staticmethod
    def _find_water_dir(root, max_depth=3):
        """Busca bajo `root` una carpeta que contenga water0NN.jpg (o bmp/png/tga)."""
        if not root or not os.path.isdir(root):
            return None
        pat = re.compile(r"^water\d\d\d\.(jpg|jpeg|bmp|png|tga)$", re.I)
        base_depth = os.path.abspath(root).rstrip(os.sep).count(os.sep)
        for dp, dn, fn in os.walk(root):
            if dp.count(os.sep) - base_depth >= max_depth:
                dn[:] = []
            if any(pat.match(f) for f in fn):
                return dp
        return None

    def _autodetect_water(self, tex_root):
        """Si todavía no hay carpeta de agua, proponer la que esté bajo las texturas."""
        if self.water_dir.get().strip():
            return
        for cand in (tex_root, os.path.join(tex_root, "texture"),
                     os.path.join(tex_root, "data", "texture")):
            found = self._find_water_dir(cand)
            if found:
                self.water_dir.set(found)
                return

    # ------------------------------------------------------------------
    # Validación común
    # ------------------------------------------------------------------

    def _validate_common(self, need_gnd=False, need_tex=False):
        gat = self.gat_path.get().strip()
        if not gat:
            messagebox.showwarning("Falta el GAT", "Elegí un archivo .gat (obligatorio).")
            return None
        if not os.path.isfile(gat):
            messagebox.showerror("Archivo no encontrado", f"No existe:\n{gat}")
            return None

        rsw = self.rsw_path.get().strip() or None
        if rsw and not os.path.isfile(rsw):
            messagebox.showerror("Archivo no encontrado", f"No existe el .rsw:\n{rsw}")
            return None

        gnd = self.gnd_path.get().strip() or None
        if need_gnd and not gnd:
            messagebox.showwarning(
                "Falta el GND",
                "El visor de terreno 3D necesita un archivo .gnd.",
            )
            return None
        if gnd and not os.path.isfile(gnd):
            messagebox.showerror("Archivo no encontrado", f"No existe el .gnd:\n{gnd}")
            return None

        tex = self.tex_dir.get().strip() or None
        if not tex:
            tex = os.path.dirname(gnd or gat)
        if need_tex and not os.path.isdir(tex):
            messagebox.showerror("Carpeta inválida", f"No existe la carpeta:\n{tex}")
            return None
        if tex and not os.path.isdir(tex):
            tex = os.path.dirname(gnd or gat)

        water = self.water_dir.get().strip() or None
        if water and not os.path.isdir(water):
            messagebox.showwarning(
                "Carpeta de agua",
                f"No existe la carpeta de agua:\n{water}\n\nSe ignora (agua animada procedural).",
            )
            water = None

        try:
            w = max(320, int(self.width_var.get()))
            h = max(240, int(self.height_var.get()))
        except ValueError:
            messagebox.showwarning("Resolución", "Usá números enteros (ej. 1280 × 800).")
            return None

        return {
            "gat": gat,
            "rsw": rsw,
            "gnd": gnd,
            "tex": tex,
            "water": water,
            "width": w,
            "height": h,
        }

    def _hide_launcher(self):
        self.root.withdraw()
        self.root.update()

    def _show_launcher(self):
        self.root.deiconify()

    def _show_error(self, title, exc):
        self._show_launcher()
        tb = traceback.format_exc()
        messagebox.showerror(
            title,
            f"{type(exc).__name__}: {exc}\n\nDetalle técnico:\n{tb[-900:]}",
        )

    # ------------------------------------------------------------------
    # Lanzadores
    # ------------------------------------------------------------------

    def _launch_2d(self):
        cfg = self._validate_common(need_gnd=False, need_tex=False)
        if cfg is None:
            return

        self._hide_launcher()
        try:
            from ro_map_viewer import ROMapViewer, GATFileError
        except Exception as e:
            self._show_error("Error al importar el visor 2D", e)
            return

        try:
            try:
                viewer = ROMapViewer(
                    cfg["gat"],
                    rsw_path=cfg["rsw"],
                    gnd_path=cfg["gnd"],
                    texture_dir=cfg["tex"],
                )
            except TypeError:
                # Versión antigua de ROMapViewer sin texture_dir
                viewer = ROMapViewer(
                    cfg["gat"],
                    rsw_path=cfg["rsw"],
                    gnd_path=cfg["gnd"],
                )
            viewer.run()
        except GATFileError as e:
            self._show_error("Error al cargar el GAT", e)
            return
        except Exception as e:
            self._show_error("Error al abrir el visor 2D", e)
            return

        self._show_launcher()

    def _launch_3d(self):
        cfg = self._validate_common(need_gnd=True, need_tex=True)
        if cfg is None:
            return

        self._hide_launcher()
        try:
            from terrain_viewer_3d import TerrainViewer3D
        except Exception as e:
            self._show_error("Error al importar el visor 3D", e)
            messagebox.showinfo(
                "Dependencias",
                "Asegurate de tener en la misma carpeta:\n"
                "  terrain_viewer_3d.py, rsm_placement.py, rolog.py, mat4.py, ro_map_viewer.py, rsm_viewer_3d.py\n\n"
                "Y las libs: pip install pygame numpy Pillow PyOpenGL",
            )
            return

        try:
            viewer = TerrainViewer3D(
                cfg["gnd"],
                rsw_path=cfg["rsw"],
                texture_dir=cfg["tex"],
                width=cfg["width"],
                height=cfg["height"],
                water_dir=cfg["water"],
            )
            viewer.run()
        except Exception as e:
            self._show_error("Error al abrir el terreno 3D", e)
            return

        self._show_launcher()

    def run(self):
        self.root.mainloop()


def main():
    app = MapLauncherApp()
    app.run()


if __name__ == "__main__":
    main()
