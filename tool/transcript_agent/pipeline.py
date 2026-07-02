from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from transcript_agent.agents.auditor import AuditorAgent
from transcript_agent.agents.extractor import ExtractorAgent
from transcript_agent.agents.rule_curator import load_active_rubric
from transcript_agent.base import PROCESSED_DIR, TRANSCRIPT_DIR, TranscriptContext
from transcript_agent.llm_config import create_llm_client_from_cfg, load_llm_config
from transcript_agent.skills.file_split import FileSplitSkill
from transcript_agent.skills.merge_extract import MergeExtractSkill

ProgressCallback = Callable[[int, str], None]


class TranscriptOrchestrator:
    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        batch_size: int = 7000,
        max_retry: int = 3,
    ):
        self.provider = provider
        self.batch_size = batch_size
        self.max_retry = max_retry

    async def run(
        self,
        transcript_path: Path,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        cfg = load_llm_config(self.provider)
        active = cfg["active_provider"]
        extract_llm = create_llm_client_from_cfg(cfg, provider=active, temperature=0.3)
        audit_llm = create_llm_client_from_cfg(cfg, provider=active, temperature=0.1)

        path = Path(transcript_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"转录文件不存在: {path}")
        root = TRANSCRIPT_DIR.resolve()
        if path.parent != root:
            raise ValueError("仅允许处理 transcripts 目录下的 .txt 文件")

        ctx = TranscriptContext(
            source_path=path,
            source_text=path.read_text(encoding="utf-8"),
            batch_size=self.batch_size,
            max_retry=self.max_retry,
            provider=active,
            rubric=load_active_rubric(),
        )
        ctx.meta["llm_provider"] = active
        ctx.meta["llm_model"] = cfg["model"]
        ctx.meta["rubric_version"] = ctx.rubric.get("version")

        def prog(pct: int, msg: str) -> None:
            if on_progress:
                on_progress(pct, msg)

        prog(5, "切分转写文本...")
        await FileSplitSkill().execute(ctx)

        extractor = ExtractorAgent(extract_llm)
        auditor = AuditorAgent(audit_llm)
        merger = MergeExtractSkill()

        revision_prompt = ""
        previous_draft: Optional[Dict[str, Any]] = None
        passed = False
        final_audit: Optional[Dict[str, Any]] = None

        while ctx.retry_count <= ctx.max_retry:
            if ctx.retry_count == 0 or not previous_draft:
                prog(10, f"提取精华（轮次 {ctx.retry_count + 1}）...")
                await extractor.extract_batches(ctx, on_progress=on_progress)
                result = await merger.merge(ctx)
            else:
                prog(10, f"修订精华（轮次 {ctx.retry_count + 1}）...")
                revised = await extractor.revise_draft(
                    ctx, previous_draft, revision_prompt, on_progress=on_progress
                )
                ctx.batch_extracts = [revised]
                result = await merger.merge(ctx)
            draft = result.to_dict()

            audit = await auditor.audit(ctx, draft, on_progress=on_progress)
            final_audit = audit.to_dict()
            if audit.passed:
                passed = True
                ctx.essence_md = merger.render_essence_md(result, final_audit)
                break

            if ctx.retry_count >= ctx.max_retry:
                ctx.essence_md = merger.render_essence_md(result, final_audit)
                break

            revision_prompt = audit.revision_prompt or _build_revision_from_issues(audit)
            previous_draft = draft
            ctx.retry_count += 1
            prog(55, f"审计未通过，准备第 {ctx.retry_count + 1} 轮修订...")

        ctx.meta["retry_count"] = ctx.retry_count
        ctx.meta["passed"] = passed
        return self._save_outputs(ctx, passed, final_audit)

    def _save_outputs(
        self,
        ctx: TranscriptContext,
        passed: bool,
        final_audit: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        stem = ctx.source_path.stem
        structured_path = PROCESSED_DIR / f"{stem}_structured.json"
        audit_path = PROCESSED_DIR / f"{stem}_audit.json"
        essence_path = PROCESSED_DIR / f"{stem}_essence.md"
        pipeline_path = PROCESSED_DIR / f"{stem}_pipeline.json"

        draft_dict = ctx.draft.to_dict() if ctx.draft else {}
        with structured_path.open("w", encoding="utf-8") as f:
            json.dump(draft_dict, f, ensure_ascii=False, indent=2)
        with audit_path.open("w", encoding="utf-8") as f:
            json.dump(final_audit or {}, f, ensure_ascii=False, indent=2)
        with essence_path.open("w", encoding="utf-8") as f:
            f.write(ctx.essence_md)

        if not passed:
            failed_path = PROCESSED_DIR / f"{stem}_failed.json"
            with failed_path.open("w", encoding="utf-8") as f:
                json.dump({"audit": final_audit, "structured": draft_dict}, f, ensure_ascii=False, indent=2)

        pipeline_payload = {
            "source_file": ctx.source_path.name,
            "llm_provider": ctx.meta.get("llm_provider"),
            "llm_model": ctx.meta.get("llm_model"),
            "rubric_version": ctx.meta.get("rubric_version"),
            "retry_count": ctx.retry_count,
            "passed": passed,
            "meta": ctx.meta,
        }
        with pipeline_path.open("w", encoding="utf-8") as f:
            json.dump(pipeline_payload, f, ensure_ascii=False, indent=2)

        return {
            "passed": passed,
            "source_file": ctx.source_path.name,
            "essence": essence_path.name,
            "structured": structured_path.name,
            "audit": audit_path.name,
            "pipeline": pipeline_path.name,
            "retry_count": ctx.retry_count,
            "score": (final_audit or {}).get("score"),
        }


def _build_revision_from_issues(audit) -> str:
    lines = ["请根据以下问题修订提取结果："]
    for issue in audit.issues:
        lines.append(f"- [{issue.severity}] {issue.field}: {issue.problem} → {issue.fix}")
    return "\n".join(lines)


def list_transcript_files() -> list[Dict[str, str]]:
    if not TRANSCRIPT_DIR.exists():
        return []
    files = []
    for path in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        if path.name.endswith(".timed.txt"):
            continue
        rel = str(path.relative_to(TRANSCRIPT_DIR.parent)).replace("\\", "/")
        files.append({"name": path.name, "path": rel})
    return files


def list_processed_files() -> list[Dict[str, Any]]:
    if not PROCESSED_DIR.exists():
        return []
    items = []
    for path in sorted(PROCESSED_DIR.glob("*_essence.md")):
        stem = path.name.replace("_essence.md", "")
        items.append(
            {
                "stem": stem,
                "essence": path.name,
                "structured": f"{stem}_structured.json",
                "audit": f"{stem}_audit.json",
                "has_failed": (PROCESSED_DIR / f"{stem}_failed.json").exists(),
            }
        )
    return items
