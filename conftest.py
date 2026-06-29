"""Root conftest so the test suite can import top-level packages.

CI runs bare ``pytest tests/`` from the repo root. Unlike ``python -m pytest``,
the bare ``pytest`` console script does not put the invocation directory on
``sys.path``, so ``from utils.color_blend import ...`` (and ``gui`` / ``tools``)
fail at collection with ``ModuleNotFoundError``. A conftest.py at the repo root
makes pytest insert this directory onto ``sys.path``, which fixes collection for
both ``pytest`` and ``python -m pytest``. Keep this file even if it stays empty.
"""
