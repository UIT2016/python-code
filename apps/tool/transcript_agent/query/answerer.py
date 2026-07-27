from __future__ import annotations

import json
from typing import Optional

from lite_agent.client import OpenAIClient

from transcript_agent.base import LogicCard
from transcript_agent.llm_config import llm_chat
from transcript_agent.query.models import FactBundle, MatchResult, QueryContext

ANSWER_SYSTEM = """你是 A 股定性投资分析助手。必须严格按「命中的 logic_card」框架回答，不得混用其他框架术语。
事实只能来自 FactBundle 与深度调研原文，框架步骤来自 logic_card 的 analysis_steps 与 answer_sections。
输出 Markdown，结构清晰。"""

STOCK_REPORT_SECTIONS = """
个股报告必须包含以下章节（按顺序）：
1. 命中框架说明（logic_id、综合得分）
2. **各业务线评分表**（Markdown 表格：业务线 | logic_id | 得分 | 理由），数据来自 segment_match
3. 按 logic_card.analysis_steps 的主分析
4. 公告/研报要点（来自 research_raw_sections 的 announcements、broker_research，若有）
5. **附录：财务风险要素**（来自 balance_sheet_risks，注明「不影响框架匹配得分」）
6. 风险提示与可行动建议
"""

SECTOR_REPORT_SECTIONS = """
板块报告必须包含：行业景气与催化、logic 框架逐步分析、风险提示与建议。
不要包含各业务线评分表或资产负债表风险附录（板块无个股财务科目）。"""


class AnswerAgent:
    name = "AnswerAgent"

    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def generate(
        self,
        query: QueryContext,
        facts: FactBundle,
        card: Optional[LogicCard],
        match: MatchResult,
    ) -> str:
        if not card or not match.sufficient:
            return self._insufficient_answer(query, facts, match)
        is_sector = query.subject_type == "sector"
        section_guide = SECTOR_REPORT_SECTIONS if is_sector else STOCK_REPORT_SECTIONS
        alice_section = ""
        if facts.research_raw_md:
            excerpt = facts.research_raw_md[:12000]
            alice_section = f"\n\n深度调研原文（节选，Wind Alice）：\n{excerpt}\n"
        user = f"""用户查询：
{json.dumps(query.to_dict(), ensure_ascii=False, indent=2)}

调研事实 FactBundle：
{json.dumps(facts.to_dict(), ensure_ascii=False, indent=2)}
{alice_section}
命中 logic_card：
{json.dumps(card.to_dict(), ensure_ascii=False, indent=2)}

匹配结果：
{json.dumps(match.to_dict(), ensure_ascii=False, indent=2)}

报告结构要求：
{section_guide}

文首注明：命中框架 [{card.logic_id}] {card.logic_name}，匹配得分 {match.selected_score}。
若存在深度调研原文，将其作为事实依据，但仍必须用 logic_card 框架组织输出。
"""
        return await llm_chat(self.llm, ANSWER_SYSTEM, user)

    @staticmethod
    def _research_mode_label(mode: str) -> str:
        labels = {
            "wanxing_search": "万行金融搜索",
            "wanxing_mcp": "万行 MCP",
            "wind_alice": "Wind Alice 深度调研",
            "wind_search": "Wind 结构化检索",
            "wanxing": "万行取数",
        }
        return labels.get(mode, mode or "万行取数")

    @staticmethod
    def _insufficient_answer(query: QueryContext, facts: FactBundle, match: MatchResult) -> str:
        mode_label = AnswerAgent._research_mode_label(facts.research_mode)
        lines = [
            f"# {query.subject} — 逻辑匹配不足",
            "",
            f"- 查询: {query.raw_query}",
            f"- 调研模式: {mode_label}",
            f"- 调研状态: {facts.wind_status}",
            f"- 最高得分: {match.selected_score}（阈值未达或存在 veto）",
            "",
        ]
        if facts.segment_match and facts.segment_match.segment_scores:
            lines.extend(["## 各业务线评分（参考）", ""])
            for seg in facts.segment_match.segment_scores:
                lines.append(
                    f"- **{seg.segment}** → {seg.logic_id or '-'} "
                    f"score={seg.score} {seg.reason[:80]}"
                )
            lines.append("")
        lines.extend(["## 候选排名", ""])
        for row in match.rankings[:5]:
            lines.append(
                f"- **{row.logic_id}** score={row.score} "
                f"trigger={','.join(row.trigger_hits) or '-'} "
                f"veto={','.join(row.veto_hits) or '-'}"
            )
        lines.extend(
            [
                "",
                "## 建议",
                "",
                "请补充更具体的事件描述、确认万行/Wind API Key，或重新精炼 logic_cards 后重建索引。",
            ]
        )
        return "\n".join(lines)
