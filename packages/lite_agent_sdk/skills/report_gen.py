from lite_agent.client import OpenAIClient

from lite_agent_sdk.base import BaseSkill, SkillContext, llm_chat

REPORT_SYSTEM = """你是一位专业的 A 股聊天室舆情分析助手。
根据已统计的热点词汇数据，撰写完整的中文 Markdown 分析报告。"""


class ReportGenSkill(BaseSkill):
    name = "ReportGenSkill"

    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def execute(self, ctx: SkillContext) -> None:
        if not ctx.hot_terms:
            ctx.report = "汇总数据中未提取到有效热点词汇。"
            return

        top_lines = []
        for idx, item in enumerate(ctx.hot_terms[:30], 1):
            rooms = "、".join(item.rooms[:5])
            top_lines.append(
                f"{idx}. {item.term} | 类别:{item.category} | 热度:{item.count} | "
                f"情绪:{item.sentiment or '中性'} | 聊天室:{rooms or '-'} | {item.note}"
            )
        stats_block = "\n".join(top_lines)
        meta = ctx.source_data
        user_prompt = f"""请基于以下统计结果撰写完整报告，结构包含：

1. 整体概述（市场情绪与主线）
2. 热点词汇排行榜（表格或列表，引用下方数据）
3. 重点板块/个股解读
4. 跨聊天室共同关注点
5. 风险提示与机会摘要

数据日期: {ctx.date_str}
聊天室数量: {meta.get('room_count', 0)}
消息总数: {meta.get('total_messages', 0)}
去重后热点数: {len(ctx.hot_terms)}

热点统计数据（按热度降序）：
{stats_block}
"""
        ctx.report = await llm_chat(self.llm, REPORT_SYSTEM, user_prompt)
