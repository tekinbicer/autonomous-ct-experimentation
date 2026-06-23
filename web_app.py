"""Chainlit entry point for the autonomous-ct web UI.

Run with::

    uv run chainlit run web_app.py -w

This is a thin launcher. All application logic lives in
:mod:`autonomous_ct.webui`; this file imports it and calls
``register_handlers()`` to wire up the Chainlit chat handlers.

Why a separate launcher instead of pointing ``chainlit run`` straight at
``src/autonomous_ct/webui.py``: Chainlit loads the target file as a
standalone module via ``importlib`` rather than as part of the
``autonomous_ct`` package. That breaks both relative imports and
module-level ``@dataclass`` resolution (the synthetic module name is not in
``sys.modules``). Loading the real module through a normal package import
sidesteps both problems.
"""

from __future__ import annotations

from autonomous_ct.webui import register_handlers

register_handlers()
