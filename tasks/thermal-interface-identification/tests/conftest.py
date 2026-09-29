import os
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE = Path(
    os.environ.get(
        "TASK_WORKSPACE",
        str(TASK_ROOT / "workspace"),
    )
).resolve()

workspace_str = str(WORKSPACE)

if workspace_str in sys.path:
    sys.path.remove(workspace_str)

sys.path.insert(0, workspace_str)

for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]