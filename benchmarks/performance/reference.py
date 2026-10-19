"""Load individual reference functions without executing old app startup code."""
import ast
import hashlib
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def original_function(module, name, revision):
    path = module.__name__.replace(".", "/") + ".py"
    directory = os.environ.get("MRS_PERF_REFERENCE_DIR")
    source = ((Path(directory)/path).read_text(encoding="utf-8") if directory else
              subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT).decode("utf-8"))
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    fragment = ast.get_source_segment(source, node)
    namespace = dict(vars(module))
    exec(compile(fragment, f"{revision}:{path}", "exec"), namespace)
    return namespace[name], hashlib.sha256(fragment.encode()).hexdigest()
