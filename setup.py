"""setuptools configuration for lumina-lob with optional C++ extension."""

import warnings

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext as _build_ext
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pybind11 is required to build the C++ extension. "
        "Install it with `pip install pybind11>=2.12` or use the dev extras."
    ) from exc

from setuptools import setup

sources = [
    "cpp/bindings.cpp",
    "cpp/src/order.cpp",
    "cpp/src/price_level.cpp",
    "cpp/src/event_log.cpp",
    "cpp/src/book.cpp",
    "cpp/src/matching.cpp",
]

include_dirs = ["cpp/include"]


class OptionalBuildExt(_build_ext):
    """Build the C++ extension if a compiler is available, otherwise warn and continue."""

    def run(self):
        try:
            super().run()
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Could not build C++ extension lumina_lob._core: {exc}\n"
                "Falling back to the pure-Python implementation. "
                "Install a C++ compiler and pybind11 to build the accelerated core."
            )
            self.extensions = []


ext_modules = [
    Pybind11Extension(
        "lumina_lob._core",
        sources=sources,
        include_dirs=include_dirs,
        cxx_std=17,
        # Windows: pybind11 handles MSVC runtime via /MD by default.
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
)
