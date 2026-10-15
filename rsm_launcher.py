#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanzador gráfico simple para el visor RSM 3D.
Pide el .rsm y la carpeta de texturas con diálogos nativos (sin consola).
"""
import os
import sys
import traceback

# Asegurar que los módulos del paquete estén en el path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    print("tkinter no está disponible en este Python.")
    sys.exit(1)


class LauncherApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RO RSM Viewer — Lanzador")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e24")

        self.rsm_path = tk.StringVar()
        self.tex_dir = tk.StringVar()
        self.width_var = tk.StringVar(value="1024")
        self.height_var = tk.StringVar(value="600")

        self._build_ui()
        self._center_window(440, 280)

    def _center_window(self, w, h):
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        pad = {"padx": 16, "pady": 6}
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", background="#1e1e24", foreground="#e0e0e0",
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#1e1e24", foreground="#7ec8ff",
                        font=("Segoe UI", 14, "bold"))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground="#2a2a32", foreground="#ffffff")
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"))

        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)
        frm.configure(style="TFrame")
        style.configure("TFrame", background="#1e1e24")

        ttk.Label(frm, text="Visor RSM 3D", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", **pad
        )
        ttk.Label(
            frm,
            text="Elegí el modelo .rsm / .rsm2 y la carpeta de texturas (.bmp)",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 10))

        # --- RSM file ---
        ttk.Label(frm, text="Archivo .rsm").grid(row=2, column=0, sticky="w", **pad)
        ent_rsm = ttk.Entry(frm, textvariable=self.rsm_path, width=36)
        ent_rsm.grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Buscar…", command=self._pick_rsm, width=10).grid(
            row=2, column=2, **pad
        )

        # --- Texture folder ---
        ttk.Label(frm, text="Texturas").grid(row=3, column=0, sticky="w", **pad)
        ent_tex = ttk.Entry(frm, textvariable=self.tex_dir, width=36)
        ent_tex.grid(row=3, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Buscar…", command=self._pick_tex, width=10).grid(
            row=3, column=2, **pad
        )

        # --- Resolution ---
        ttk.Label(frm, text="Resolución").grid(row=4, column=0, sticky="w", **pad)
        res_frm = ttk.Frame(frm)
        res_frm.grid(row=4, column=1, sticky="w", **pad)
        ttk.Entry(res_frm, textvariable=self.width_var, width=6).pack(side=tk.LEFT)
        ttk.Label(res_frm, text=" × ").pack(side=tk.LEFT)
        ttk.Entry(res_frm, textvariable=self.height_var, width=6).pack(side=tk.LEFT)

        # --- Launch ---
        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=5, column=0, columnspan=3, pady=18)
        ttk.Button(
            btn_frm, text="  Abrir visor 3D  ", command=self._launch, style="Accent.TButton"
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frm, text="Salir", command=self.root.destroy).pack(
            side=tk.LEFT, padx=8
        )

        # Hint
        ttk.Label(
            frm,
            text="En el visor: click+arrastrar = orbitar · rueda = zoom · "
                 "Espacio = pausa · [ ] = velocidad · R = reiniciar · ESC = salir",
            font=("Segoe UI", 8),
            foreground="#888888",
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=16)

        frm.columnconfigure(1, weight=1)

    def _pick_rsm(self):
        path = filedialog.askopenfilename(
            title="Elegir modelo RSM",
            filetypes=[
                ("Modelos RSM / RSM2", "*.rsm *.RSM *.rsm2 *.RSM2"),
                ("RSM1", "*.rsm *.RSM"),
                ("RSM2", "*.rsm2 *.RSM2"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return
        self.rsm_path.set(path)
        # Si todavía no hay carpeta de texturas, proponer la del .rsm
        if not self.tex_dir.get().strip():
            self.tex_dir.set(os.path.dirname(path))

    def _pick_tex(self):
        path = filedialog.askdirectory(title="Carpeta de texturas (.bmp)")
        if path:
            self.tex_dir.set(path)

    def _launch(self):
        rsm = self.rsm_path.get().strip()
        tex = self.tex_dir.get().strip()
        if not rsm:
            messagebox.showwarning("Falta el modelo", "Elegí un archivo .rsm.")
            return
        if not os.path.isfile(rsm):
            messagebox.showerror("Archivo no encontrado", f"No existe:\n{rsm}")
            return
        if not tex:
            tex = os.path.dirname(rsm)
        if not os.path.isdir(tex):
            messagebox.showerror("Carpeta inválida", f"No existe la carpeta:\n{tex}")
            return

        try:
            w = max(320, int(self.width_var.get()))
            h = max(240, int(self.height_var.get()))
        except ValueError:
            messagebox.showwarning("Resolución", "Usá números enteros (ej. 1024 × 768).")
            return

        # Cerrar el launcher antes de abrir la ventana OpenGL
        self.root.withdraw()
        self.root.update()

        try:
            from rsm_viewer_3d import RSMViewer3D, RSMFileError
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror(
                "Error al importar el visor",
                f"{e}\n\nAsegurate de tener rsm_viewer_3d.py, mat4.py y "
                f"ro_map_viewer.py en la misma carpeta.\n"
                f"Dependencias: pip install pygame numpy Pillow PyOpenGL",
            )
            return

        try:
            viewer = RSMViewer3D(rsm, texture_dir=tex, width=w, height=h)
            viewer.run()
        except Exception as e:
            # Restaurar launcher para poder reintentar
            self.root.deiconify()
            tb = traceback.format_exc()
            messagebox.showerror(
                "Error al abrir el modelo",
                f"{type(e).__name__}: {e}\n\nDetalle técnico:\n{tb[-800:]}",
            )
            return

        # Al salir del visor, volver al launcher
        self.root.deiconify()

    def run(self):
        self.root.mainloop()


def main():
    app = LauncherApp()
    app.run()


if __name__ == "__main__":
    main()
