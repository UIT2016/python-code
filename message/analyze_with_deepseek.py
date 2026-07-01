import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config_loader import SUPPORTED_LLM_PROVIDERS, load_config, resolve_llm_config
from lite_agent_sdk import AnalysisOrchestratorAgent, SkillContext, create_llm_client_from_cfg

BASE_DIR = Path(__file__).resolve().parent
MESSAGE_DATA_DIR = BASE_DIR / "message_data"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "analysis_results"

cfg = load_config(BASE_DIR)
DEFAULT_BATCH_SIZE = 8000


def load_messages_file(date_str: str) -> Dict[str, Any]:
    path = MESSAGE_DATA_DIR / f"{date_str}.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到消息文件: {path}，请先运行 message.py 获取消息。")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _serialize_hot_terms(ctx: SkillContext) -> list:
    return [asdict(item) for item in ctx.hot_terms]


def save_results(date_str: str, ctx: SkillContext, llm_provider: str, llm_model: str) -> tuple[Path, Path]:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = ANALYSIS_OUTPUT_DIR / f"{date_str}_report.md"
    data_file = ANALYSIS_OUTPUT_DIR / f"{date_str}_pipeline.json"

    header = (
        f"# 聊天室热点分析报告\n\n"
        f"- 分析日期: {date_str}\n"
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- LLM: {llm_provider} / {llm_model}\n"
        f"- 聊天室: {ctx.source_data.get('room_count', 0)} 个\n"
        f"- 消息总数: {ctx.source_data.get('total_messages', 0)} 条\n"
        f"- 热点词条: {len(ctx.hot_terms)} 个\n\n"
        f"---\n\n"
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(header + ctx.report)

    pipeline_payload = {
        "date": date_str,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "meta": ctx.meta,
        "hot_terms": _serialize_hot_terms(ctx),
        "batch_count": len(ctx.batches),
    }
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_payload, f, ensure_ascii=False, indent=2)

    return report_file, data_file


async def analyze_date(
    date_str: str,
    provider: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[Path, Path]:
    data = load_messages_file(date_str)
    llm_cfg = resolve_llm_config(cfg, provider=provider)
    active_provider = llm_cfg["provider"]
    active_model = llm_cfg["model"]

    print(
        f"已加载 {date_str}.json："
        f"{data.get('room_count', 0)} 个聊天室，"
        f"共 {data.get('total_messages', 0)} 条消息"
    )
    print(f"使用 LLM: {active_provider} / {active_model}")

    llm = create_llm_client_from_cfg(cfg, provider=active_provider, temperature=0.3)
    ctx = SkillContext(date_str=date_str, source_data=data)
    agent = AnalysisOrchestratorAgent(llm, batch_size=batch_size)
    ctx = await agent.run(ctx)
    return save_results(date_str, ctx, active_provider, active_model)


def main():
    parser = argparse.ArgumentParser(description="使用 Lite-Agent Skill 流水线分析聊天室热点")
    parser.add_argument(
        "date",
        nargs="?",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="消息文件日期，格式 YYYY-MM-DD（默认今天）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="FileSplitSkill 单批最大字符数（默认 8000）",
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_LLM_PROVIDERS,
        default=None,
        help=f"LLM 提供商（默认读取 config.local.json 的 llm_provider，当前: {cfg['llm_provider']}）",
    )
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"日期格式无效: {args.date}，请使用 YYYY-MM-DD")
        sys.exit(1)

    try:
        report_file, data_file = asyncio.run(
            analyze_date(args.date, provider=args.provider, batch_size=max(1000, args.batch_size))
        )
        print("\n分析完成！")
        print(f"  报告: {report_file}")
        print(f"  结构化数据: {data_file}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
