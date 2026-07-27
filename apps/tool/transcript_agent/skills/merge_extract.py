from __future__ import annotations

from typing import Any, Dict, List

from transcript_agent.base import ExtractResult, LogicCard, TranscriptContext, merge_logic_cards, parse_logic_card, parse_video_meta


class MergeExtractSkill:
    name = "MergeExtractSkill"

    async def merge(self, ctx: TranscriptContext) -> ExtractResult:
        filename = ctx.source_path.name
        meta = parse_video_meta(filename)
        core_theses: List[str] = []
        removed: List[str] = []
        all_cards: List[LogicCard] = []

        for batch in ctx.batch_extracts:
            thesis = (batch.get("core_thesis") or "").strip()
            if thesis:
                core_theses.append(thesis)
            rs = (batch.get("removed_summary") or "").strip()
            if rs:
                removed.append(rs)
            for row in batch.get("logic_cards") or []:
                card = parse_logic_card(row)
                if card:
                    all_cards.append(card)

        core = core_theses[0] if core_theses else ""
        cards = merge_logic_cards(all_cards)

        result = ExtractResult(
            source_file=filename,
            video_meta=meta,
            core_thesis=core,
            logic_cards=cards,
            removed_summary="；".join(dict.fromkeys(removed)),
        )
        ctx.draft = result
        ctx.meta["logic_card_count"] = len(result.logic_cards)
        return result

    @staticmethod
    def render_essence_md(result: ExtractResult, audit: Dict[str, Any] | None = None) -> str:
        lines = [
            f"# {result.video_meta.get('title') or result.source_file}",
            "",
            "> 本文件由 logic_cards 渲染，RAG 检索请优先使用同目录 `*_logic_cards.json`。",
            "",
        ]
        if result.video_meta.get("bvid"):
            lines.append(f"- BVID: `{result.video_meta['bvid']}`")
        lines.append(f"- 核心论点: {result.core_thesis or '（未提取）'}")
        lines.append(f"- 逻辑卡数量: {len(result.logic_cards)}")
        if audit:
            lines.append(f"- 审计得分: {audit.get('score', 'N/A')} / 通过: {'是' if audit.get('passed') else '否'}")
        lines.extend(["", "---", ""])

        for card in result.logic_cards:
            lines.append(f"## [{card.logic_id}] {card.logic_name}")
            lines.append("")
            if card.logic_family:
                lines.append(f"- **大类**: {card.logic_family}")
            if card.summary:
                lines.append(f"- **定义**: {card.summary}")
            if card.triggers:
                lines.append("- **触发信号**:")
                for t in card.triggers:
                    lines.append(f"  - {t}")
            if card.veto_conditions:
                lines.append("- **否决条件**:")
                for v in card.veto_conditions:
                    lines.append(f"  - {v}")
            if card.preconditions:
                lines.append("- **前置条件**:")
                for p in card.preconditions:
                    lines.append(f"  - {p}")
            if card.analysis_steps:
                lines.append("- **分析步骤**:")
                for i, step in enumerate(card.analysis_steps, 1):
                    lines.append(f"  {i}. {step}")
            if card.answer_sections:
                lines.append("- **回答结构**:")
                for s in card.answer_sections:
                    lines.append(f"  - {s}")
            if card.keywords:
                lines.append(f"- **关键词**: {', '.join(card.keywords)}")
            if card.related_logics:
                lines.append(f"- **相关逻辑**: {', '.join(card.related_logics)}")
            if card.exclude_logics:
                lines.append(f"- **互斥逻辑**: {', '.join(card.exclude_logics)}")
            if card.examples:
                lines.append("- **案例**:")
                for ex in card.examples:
                    subj = ex.get("subject") or ""
                    note = ex.get("note") or ex.get("scenario") or ""
                    lines.append(f"  - {subj}: {note}")
            if card.evidence_quotes:
                lines.append("- **原文引用**:")
                for q in card.evidence_quotes:
                    lines.append(f"  > {q}")
            lines.append("")
            lines.append("### RAG 文本")
            lines.append("")
            lines.append("```")
            lines.append(card.to_dict()["rag_text"])
            lines.append("```")
            lines.append("")

        if result.removed_summary:
            lines.extend(["## 已剔除内容", "", result.removed_summary, ""])
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def build_logic_cards_payload(result: ExtractResult) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for card in result.logic_cards:
            item = card.to_dict()
            item["source_file"] = result.source_file
            item["source_title"] = result.video_meta.get("title") or ""
            item["source_bvid"] = result.video_meta.get("bvid") or ""
            payload.append(item)
        return payload
