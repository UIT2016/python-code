"""Fetch A-share index/sector volume and fund flow via East Money."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_TTL_SEC = 60
_TOP_N = 10
_cache: Dict[str, Any] = {"ts": 0.0, "data": None}

# Major A-share indices to prefer when ranking spot list.
_INDEX_KEYWORDS = (
    "上证指数",
    "深证成指",
    "创业板指",
    "科创50",
    "沪深300",
    "中证500",
    "中证1000",
    "上证50",
    "深证100",
    "中证2000",
    "北证50",
    "中小100",
    "创业板50",
    "红利指数",
    "上证180",
)

# Index board filters (重要指数优先，其余系列作补齐)
_INDEX_FS = ("b:MK0010", "m:1+t:1", "m:0+t:5", "m:2")

# 本机对 push2.*.eastmoney.com 常被 RemoteDisconnected；delay 节点更稳定
_PUSH2_HOSTS = (
    "https://push2delay.eastmoney.com",
    "https://push2delayh.eastmoney.com",
    "https://push2.eastmoney.com",
    "https://82.push2.eastmoney.com",
    "https://33.push2.eastmoney.com",
    "https://48.push2.eastmoney.com",
    "https://17.push2.eastmoney.com",
)

_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.proxies = {"http": None, "https": None}
_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json, text/plain, */*",
        "Connection": "close",
    }
)


def _yi_yuan(value: Any) -> float:
    """Normalize amount in yuan to 亿元."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(v / 1e8, 2)


def _top_bars(
    df: pd.DataFrame,
    name_col: str,
    value_col: str,
    n: int = _TOP_N,
) -> List[Dict[str, Any]]:
    work = df[[name_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work = work.sort_values(value_col, ascending=False).head(n)
    return [
        {"name": str(row[name_col]), "value": _yi_yuan(row[value_col])}
        for _, row in work.iterrows()
    ]


def _em_get_json(url: str, params: Dict[str, str], timeout: int = 20) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.45 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _em_clist(url: str, params: Dict[str, str], timeout: int = 20) -> pd.DataFrame:
    first = dict(params)
    first["pn"] = "1"
    payload = _em_get_json(url, first, timeout=timeout)
    data = payload.get("data") or {}
    total = int(data.get("total") or 0)
    page_size = int(params.get("pz") or 100)
    pages = max(1, math.ceil(total / page_size)) if total else 1
    frames = [pd.DataFrame(data.get("diff") or [])]
    for page in range(2, min(pages, 2) + 1):
        p = dict(params)
        p["pn"] = str(page)
        frames.append(
            pd.DataFrame(
                (_em_get_json(url, p, timeout=timeout).get("data") or {}).get("diff") or []
            )
        )
    if not frames or all(f.empty for f in frames):
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _em_clist_hosts(
    path: str,
    params: Dict[str, str],
    hosts: Tuple[str, ...] = _PUSH2_HOSTS,
) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for host in hosts:
        try:
            df = _em_clist(f"{host}{path}", params)
            if not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.debug("EM host failed %s: %s", host, exc)
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


def _fetch_indices() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        frames: List[pd.DataFrame] = []
        last_err: Optional[str] = None
        for fs in _INDEX_FS:
            try:
                raw = _em_clist_hosts(
                    "/api/qt/clist/get",
                    {
                        "pn": "1",
                        "pz": "100",
                        "po": "1",
                        "np": "1",
                        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                        "fltt": "2",
                        "invt": "2",
                        "wbp2u": "|0|0|0|web",
                        "fid": "f6",
                        "fs": fs,
                        "fields": "f12,f14,f5,f6",
                    },
                )
                if raw.empty:
                    continue
                part = pd.DataFrame(
                    {
                        "名称": raw.get("f14"),
                        "成交额": pd.to_numeric(raw.get("f6"), errors="coerce"),
                    }
                ).dropna(subset=["名称"])
                frames.append(part)
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)

        if not frames:
            return [], last_err or "指数现货拉取失败"

        df = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["名称"], keep="first"
        )
        preferred = df[df["名称"].astype(str).isin(_INDEX_KEYWORDS)]
        source = preferred if len(preferred) >= 5 else df
        return _top_bars(source, "名称", "成交额"), None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def _fetch_sector_volume() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        raw = _em_clist_hosts(
            "/api/qt/clist/get",
            {
                "pn": "1",
                "pz": "100",
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f6",
                "fs": "m:90 t:2 f:!50",
                "fields": "f12,f14,f5,f6",
            },
        )
        if raw.empty:
            return [], "行业板块现货为空"
        df = pd.DataFrame(
            {
                "板块名称": raw.get("f14"),
                "成交额": pd.to_numeric(raw.get("f6"), errors="coerce"),
            }
        ).dropna(subset=["板块名称"])
        return _top_bars(df, "板块名称", "成交额"), None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def _fetch_sector_fund_flow() -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Optional[str],
]:
    try:
        raw = _em_clist_hosts(
            "/api/qt/clist/get",
            {
                "pn": "1",
                "pz": "100",
                "po": "1",
                "np": "1",
                "ut": "b2884a393a59ad64002292a3e90d46a5",
                "fltt": "2",
                "invt": "2",
                "fid0": "f62",
                "fs": "m:90 t:2",
                "stat": "1",
                "fields": "f12,f14,f62,f66,f69,f72,f75,f78,f81,f84,f87",
                "rt": "52975239",
                "_": str(int(time.time() * 1000)),
            },
        )
        if raw.empty:
            return [], [], "板块资金流为空"
        df = pd.DataFrame(
            {
                "名称": raw.get("f14"),
                "今日主力净流入-净额": pd.to_numeric(raw.get("f62"), errors="coerce"),
            }
        ).dropna(subset=["名称"])
        name_col = "名称"
        net_col = "今日主力净流入-净额"
        work = df[[name_col, net_col]].copy()
        work[net_col] = pd.to_numeric(work[net_col], errors="coerce").fillna(0.0)
        in_df = work[work[net_col] > 0].copy()
        out_df = work[work[net_col] < 0].copy()
        out_df[net_col] = out_df[net_col].abs()
        return _top_bars(in_df, name_col, net_col), _top_bars(out_df, name_col, net_col), None
    except Exception as exc:  # noqa: BLE001
        return [], [], str(exc)


def get_overview(*, force: bool = False) -> Dict[str, Any]:
    now = time.time()
    if (
        not force
        and _cache["data"] is not None
        and now - float(_cache["ts"]) < CACHE_TTL_SEC
    ):
        return _cache["data"]

    indices, idx_err = _fetch_indices()
    sectors, sec_err = _fetch_sector_volume()
    inflow, outflow, flow_err = _fetch_sector_fund_flow()

    errors: Dict[str, str] = {}
    if idx_err:
        errors["indices"] = idx_err
    if sec_err:
        errors["sectors_volume"] = sec_err
    if flow_err:
        errors["fund_flow"] = flow_err

    data: Dict[str, Any] = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "unit": "亿元",
        "indices": indices,
        "sectors_volume": sectors,
        "inflow": inflow,
        "outflow": outflow,
        "errors": errors,
    }
    # 有任一成功结果才写入缓存，避免把全失败结果缓存 60s
    if indices or sectors or inflow or outflow:
        _cache["ts"] = now
        _cache["data"] = data
    return data
