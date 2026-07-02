from __future__ import annotations

from typing import Any, Dict, List

from transcript_agent.base import ExtractResult, InsightItem, TranscriptContext, parse_video_meta


def _norm_key(text: str) -> str:
    return "".join(text.lower().split())[:80]


class MergeExtractSkill:
    name = "MergeExtractSkill"

    async def merge(self, ctx: TranscriptContext) -> ExtractResult:
        filename = ctx.source_path.name
        meta = parse_video_meta(filename)
        core_theses: List[str] = []
        tags: List[str] = []
        removed: List[str] = []
        insight_map: Dict[str, InsightItem] = {}

        for batch in ctx.batch_extracts:
            thesis = (batch.get("core_thesis") or "").strip()
            if thesis:
                core_theses.append(thesis)
            for tag in batch.get("framework_tags") or []:
                t = str(tag).strip()
                if t and t not in tags:
                    tags.append(t)
            rs = (batch.get("removed_summary") or "").strip()
            if rs:
                removed.append(rs)
            for row in batch.get("insights") or []:
                if not isinstance(row, dict):
                    continue
                claim = str(row.get("claim") or "").strip()
                if not claim:
                    continue
                key = _norm_key(claim)
                category = str(row.get("category") or "strategy").strip()
                reasoning = str(row.get("reasoning") or "").strip()
                actionable = str(row.get("actionable") or "").strip()
                quotes = [str(q).strip() for q in (row.get("evidence_quotes") or []) if str(q).strip()]
                if key in insight_map:
                    existing = insight_map[key]
                    for q in quotes:
                        if q not in existing.evidence_quotes:
                            existing.evidence_quotes.append(q)
                    if reasoning and reasoning not in existing.reasoning:
                        existing.reasoning = (existing.reasoning + " " + reasoning).strip()
                    if actionable and not existing.actionable:
                        existing.actionable = actionable
                else:
                    insight_map[key] = InsightItem(
                        category=category,
                        claim=claim,
                        reasoning=reasoning,
                        evidence_quotes=quotes[:3],
                        actionable=actionable,
                    )

        core = core_theses[0] if core_theses else ""

        result = ExtractResult(
            source_file=filename,
            video_meta=meta,
            core_thesis=core,
            framework_tags=tags,
            insights=list(insight_map.values()),
            removed_summary="；".join(dict.fromkeys(removed)),
        )
        ctx.draft = result
        ctx.meta["insight_count"] = len(result.insights)
        return result

    @staticmethod
    def render_essence_md(result: ExtractResult, audit: Dict[str, Any] | None = None) -> str:
        lines = [
            f"# {result.video_meta.get('title') or result.source_file}",
            "",
        ]
        if result.video_meta.get("bvid"):
            lines.append(f"- BVID: `{result.video_meta['bvid']}`")
        lines.append(f"- 核心论点: {result.core_thesis or '（未提取）'}")
        if result.framework_tags:
            lines.append(f"- 框架标签: {', '.join(result.framework_tags)}")
        if audit:
            lines.append(f"- 审计得分: {audit.get('score', 'N/A')} / 通过: {'是' if audit.get('passed') else '否'}")
        lines.extend(["", "---", ""])

        by_cat: Dict[str, List[InsightItem]] = {}
        for item in result.insights:
            by_cat.setdefault(item.category, []).append(item)

        cat_labels = {
            "strategy": "策略框架",
            "sector": "板块/行业",
            "stock": "个股案例",
            "risk": "风险提示",
            "mindset": "交易心态",
            "case": "案例复盘",
        }
        for cat, items in by_cat.items():
            lines.append(f"## {cat_labels.get(cat, cat)}")
            lines.append("")
            for idx, item in enumerate(items, 1):
                lines.append(f"### {idx}. {item.claim}")
                if item.reasoning:
                    lines.append(f"- **逻辑**: {item.reasoning}")
                if item.actionable:
                    lines.append(f"- **可行动**: {item.actionable}")
                if item.evidence_quotes:
                    lines.append("- **原文引用**:")
                    for q in item.evidence_quotes:
                        lines.append(f"  > {q}")
                lines.append("")

        if result.removed_summary:
            lines.extend(["## 已剔除内容", "", result.removed_summary, ""])
        return "\n".join(lines).strip() + "\n"
