from __future__ import annotations

import json
from typing import Optional

from lite_agent.client import OpenAIClient

from transcript_agent.base import LogicCard
from transcript_agent.llm_config import llm_chat
from transcript_agent.query.models import FactBundle, MatchResult, QueryContext

ANSWER_SYSTEM = """你是 A 股定性投资分析助手。必须严格按「命中的 logic_card」框架回答，不得混用其他框架术语。
事实只能来自 FactBundle 与深度调研原文，框架步骤来自 logic_card 的 analysis_steps 与 answer_sections。
输出 Markdown，结构清晰，包含：命中框架说明、逐步分析、风险提示、可行动建议。"""


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

请按 logic_card.analysis_steps 逐步分析，章节对齐 answer_sections。
文首注明：命中框架 [{card.logic_id}] {card.logic_name}，匹配得分 {match.selected_score}。
若存在深度调研原文，将其作为事实依据，但仍必须用 logic_card 框架组织输出。
"""
        return await llm_chat(self.llm, ANSWER_SYSTEM, user)

    @staticmethod
    def _insufficient_answer(query: QueryContext, facts: FactBundle, match: MatchResult) -> str:
        mode_label = "Wind Alice 深度调研" if facts.research_mode == "wind_alice" else "万行取数"
        lines = [
            f"# {query.subject} — 逻辑匹配不足",
            "",
            f"- 查询: {query.raw_query}",
            f"- 调研模式: {mode_label}",
            f"- 调研状态: {facts.wind_status}",
            f"- 最高得分: {match.selected_score}（阈值未达或存在 veto）",
            "",
            "## 候选排名",
            "",
        ]
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
