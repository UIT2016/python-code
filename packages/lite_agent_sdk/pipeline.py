from lite_agent.client import OpenAIClient

from lite_agent_sdk.base import SkillContext
from lite_agent_sdk.skills import (
    DeduplicateSkill,
    FileSplitSkill,
    FinanceExtractSkill,
    ReportGenSkill,
    StatAggSkill,
)


class AnalysisOrchestratorAgent:
    """按固定顺序自主调度 Skill：FileSplit → FinanceExtract(循环) → StatAgg → Deduplicate → ReportGen。"""

    def __init__(self, llm: OpenAIClient, batch_size: int = 8000):
        self.llm = llm
        self.batch_size = batch_size
        self.file_split = FileSplitSkill()
        self.finance_extract = FinanceExtractSkill(llm)
        self.stat_agg = StatAggSkill()
        self.deduplicate = DeduplicateSkill()
        self.report_gen = ReportGenSkill(llm)

    async def run(self, ctx: SkillContext) -> SkillContext:
        ctx.batch_size = self.batch_size

        print(f"[{self.file_split.name}] 拆分消息文件...")
        await self.file_split.execute(ctx)
        print(f"  生成 {len(ctx.batches)} 个批次")

        ctx.extract_results = []
        total = len(ctx.batches)
        for idx, batch in enumerate(ctx.batches, 1):
            print(f"[{self.finance_extract.name}] 解析批次 {idx}/{total} (消息 {batch.message_count} 条)...")
            items = await self.finance_extract.execute_batch(batch, ctx)
            ctx.extract_results.append(items)
            print(f"  提取 {len(items)} 个热点项")

        print(f"[{self.stat_agg.name}] 聚合统计...")
        await self.stat_agg.execute(ctx)
        print(f"  聚合 {ctx.meta.get('aggregated_term_count', 0)} 个词条")

        print(f"[{self.deduplicate.name}] 去重合并...")
        await self.deduplicate.execute(ctx)
        print(f"  去重后 {ctx.meta.get('hot_term_count', 0)} 个热点")

        print(f"[{self.report_gen.name}] 生成报告...")
        await self.report_gen.execute(ctx)
        print("  报告生成完成")

        return ctx
