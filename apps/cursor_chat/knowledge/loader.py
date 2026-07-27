"""从本地目录加载知识文件。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

# 当前支持纯文本类格式；PDF/DOCX 可在后续通过可选依赖扩展
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".json", ".yaml", ".yml"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_json_yaml_text(path: Path, raw: str) -> str:
    """JSON/YAML 转为可读文本，便于检索与注入上下文。"""
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(raw)
            return json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return raw
    return raw


def iter_knowledge_files(directories: Iterable[str | Path]) -> List[Path]:
    """递归扫描目录，返回支持格式的文件列表。"""
    files: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in SUPPORTED_EXTENSIONS:
            if root not in seen:
                seen.add(root)
                files.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if path in seen:
                continue
            seen.add(path)
            files.append(path)
    return sorted(files)


def load_file_content(path: Path) -> Tuple[str, str]:
    """读取单个文件，返回 (source_label, content)。"""
    raw = _read_text_file(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        raw = _normalize_json_yaml_text(path, raw)
    return str(path), raw.strip()
