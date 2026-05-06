from ._version import __version__


def __getattr__(name):
    """Lazy import to avoid loading IsaacLab when not needed (e.g., for download_assets)."""
    if name == "make_tavis_env":
        from .make_env import make_tavis_env
        return make_tavis_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
