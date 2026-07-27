"""任务模式预设（对照原 DeepSeek/Qwen 能力的精简版 Prompt，不依赖旧模块）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModePreset:
    id: str
    label: str
    placeholder: str
    hint: str
    system_prompt: str


_BASE_GUARD = (
    "只做文字分析与说明，不要修改、创建或删除任何本地文件，不要执行终端命令。"
    "输出使用清晰的中文 Markdown。"
)

PRESETS: Dict[str, ModePreset] = {
    "chat": ModePreset(
        id="chat",
        label="普通聊天",
        placeholder="输入消息，Ctrl+Enter 发送",
        hint="自由问答。",
        system_prompt=f"你是简洁、专业的助手，用中文回答。{_BASE_GUARD}",
    ),
    "refine": ModePreset(
        id="refine",
        label="文本精炼",
        placeholder="粘贴 ASR / 长文本，然后发送「请精炼」或直接发送全文",
        hint="对齐「转录精炼」：提取可复用的投资逻辑要点与结构化卡片。",
        system_prompt=(
            "你是一位 A 股投资定性分析助手，负责从转写稿或长文本中提取可 RAG 匹配的逻辑框架。\n"
            f"{_BASE_GUARD}\n"
            "请输出 Markdown，建议结构：\n"
            "1. 一句话主题\n"
            "2. 逻辑要点列表（因果/催化/风险）\n"
            "3. 「逻辑卡」JSON 数组（可写在代码块），每项字段："
            "title, thesis, catalysts[], risks[], tags[], confidence(0-1)\n"
            "4. 原文中不明确之处单独列出。"
        ),
    ),
    "logic": ModePreset(
        id="logic",
        label="逻辑分析",
        placeholder="输入标的/问题，可附带补充事件或上下文",
        hint="对齐「逻辑分析」：按投资框架写结构化 Markdown 报告。",
        system_prompt=(
            "你是一位严谨的 A 股逻辑分析助手。根据用户给出的标的、问题与上下文撰写报告。\n"
            f"{_BASE_GUARD}\n"
            "报告结构必须包含：\n"
            "## 问题理解\n## 分析框架\n## 分步推理\n## 关键结论\n## 风险与证伪条件\n## 信息缺口\n"
            "若上下文不足，明确写出假设，不要编造财报数字。"
        ),
    ),
    "hotspot": ModePreset(
        id="hotspot",
        label="热点分析",
        placeholder="粘贴聊天室消息片段，或说明日期与关注点",
        hint="对齐「聊天室热点报告」：提取个股/板块/情绪并写舆情报告。",
        system_prompt=(
            "你是一位 A 股聊天室舆情分析助手。\n"
            f"{_BASE_GUARD}\n"
            "从用户粘贴的消息中提取热点，输出 Markdown：\n"
            "1. 整体概述（情绪与主线）\n"
            "2. 热点排行（个股/板块/关键词，附热度直觉与情绪）\n"
            "3. 跨聊天室共同关注点\n"
            "4. 风险提示与机会摘要\n"
            "忽略「收到一条语音消息」等无效内容。"
        ),
    ),
    "polish": ModePreset(
        id="polish",
        label="文本润色",
        placeholder="粘贴原文，并说明目标文体（如报告/纪要/口语）",
        hint="对齐「文本转化」：改写、润色、格式整理。",
        system_prompt=(
            "你是专业中文写作与文本转化助手。\n"
            f"{_BASE_GUARD}\n"
            "按用户要求改写：保持事实与数字不变；提升清晰度与结构；"
            "默认输出完整改写稿，必要时用简短列表说明改动要点。"
        ),
    ),
    "fetch": ModePreset(
        id="fetch",
        label="检索下载",
        placeholder="可在此提问如何处理已下载文件；下载请用上方 URL 区域",
        hint="解析并下载视频/音频到本机 download_out，再可切到精炼模式继续分析。",
        system_prompt=(
            "你是媒资助理。用户可能刚下载了音视频文件，请协助说明后续如何精炼/分析。\n"
            f"{_BASE_GUARD}\n"
            "若用户只给了链接而尚未下载，提示其使用页面上的「解析/下载」按钮。"
        ),
    ),
}

DEFAULT_MODE = "chat"


def list_modes() -> List[Dict[str, str]]:
    return [
        {
            "id": p.id,
            "label": p.label,
            "placeholder": p.placeholder,
            "hint": p.hint,
        }
        for p in PRESETS.values()
    ]


def get_preset(mode_id: str) -> ModePreset:
    return PRESETS.get((mode_id or "").strip().lower()) or PRESETS[DEFAULT_MODE]
