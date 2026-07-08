"""
Registers the custom CBAM module with Ultralytics' model parser so that
architecture YAMLs can reference "CBAM" as a layer type, the same way they
reference built-in layers such as C2f or SPPF.

Ultralytics' `parse_model` resolves each YAML layer's module name via
`globals()` inside `ultralytics.nn.tasks`. Importing this module (before
constructing a YOLO(...) model from a *-cbam.yaml file) injects CBAM into
that namespace so the lookup succeeds.
"""
import ultralytics.nn.tasks as tasks

from .attention import CBAM

tasks.CBAM = CBAM


def register():
    """No-op call for readability at call sites; registration already
    happens on import."""
    tasks.CBAM = CBAM
