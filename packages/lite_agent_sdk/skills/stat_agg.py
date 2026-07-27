from lite_agent_sdk.base import BaseSkill, ExtractedItem, SkillContext


def _norm_key(term: str) -> str:
    return term.strip().lower().replace(" ", "")


class StatAggSkill(BaseSkill):
    name = "StatAggSkill"

    async def execute(self, ctx: SkillContext) -> None:
        aggregated: dict[str, ExtractedItem] = {}
        for batch_items in ctx.extract_results:
            for item in batch_items:
                key = _norm_key(item.term)
                if not key:
                    continue
                if key not in aggregated:
                    aggregated[key] = ExtractedItem(
                        term=item.term,
                        category=item.category,
                        count=0,
                        sentiment=item.sentiment,
                        note=item.note,
                        rooms=[],
                    )
                agg = aggregated[key]
                agg.count += 1
                if item.sentiment and not agg.sentiment:
                    agg.sentiment = item.sentiment
                if item.note and len(item.note) > len(agg.note):
                    agg.note = item.note
                for room in item.rooms:
                    if room and room not in agg.rooms:
                        agg.rooms.append(room)
        ctx.aggregated = aggregated
        ctx.meta["aggregated_term_count"] = len(aggregated)
