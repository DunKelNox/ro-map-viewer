# -*- coding: utf-8 -*-
"""Mensajes de consola del visor.

info()  -> silencioso por defecto; se activa con la variable de entorno RO_VERBOSE=1
warn()  -> siempre se muestra (problemas reales: archivos que faltan, errores, etc.)
"""
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")   # sin banner de pygame

VERBOSE = os.environ.get("RO_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def info(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def warn(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)
