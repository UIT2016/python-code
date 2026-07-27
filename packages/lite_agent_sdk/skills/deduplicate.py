from lite_agent_sdk.base import BaseSkill, ExtractedItem, SkillContext


def _merge_key(term: str) -> str:
    t = term.strip()
    for ch in ("(", "（", "[", "【"):
        if ch in t:
            t = t.split(ch, 1)[0].strip()
    return t.lower().replace(" ", "")


class DeduplicateSkill(BaseSkill):
    name = "DeduplicateSkill"

    async def execute(self, ctx: SkillContext) -> None:
        merged: dict[str, ExtractedItem] = {}
        for item in sorted(ctx.aggregated.values(), key=lambda x: x.count, reverse=True):
            key = _merge_key(item.term)
            if not key:
                continue
            if key not in merged:
                merged[key] = ExtractedItem(
                    term=item.term,
                    category=item.category,
                    count=item.count,
                    sentiment=item.sentiment,
                    note=item.note,
                    rooms=list(item.rooms),
                )
                continue
            target = merged[key]
            target.count += item.count
            if len(item.term) > len(target.term):
                target.term = item.term
            for room in item.rooms:
                if room and room not in target.rooms:
                    target.rooms.append(room)
            if item.note and len(item.note) > len(target.note):
                target.note = item.note

        ctx.hot_terms = sorted(merged.values(), key=lambda x: x.count, reverse=True)
        ctx.meta["hot_term_count"] = len(ctx.hot_terms)
