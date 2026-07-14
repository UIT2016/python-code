"""示例公共工具：将项目根目录加入 sys.path。"""

from __future__ import annotations

import sys
from pathlib import Path


def setup_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root
