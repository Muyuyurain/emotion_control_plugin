"""情绪模型 —— 纯函数 + 一个无 IO 的状态容器（时钟一律由外部注入）。

三层时间轴
----------
    temper   常驻底座，不随对话改变（只有 `hermes emotion set` 能手动改）
    mood     = temper + Σ(事件 × e^(-(now-ts)/TAU_EVENT))      # 账本，τ = 2h
    emotion  = mood + 本轮尚未结转的 impulse                    # impulse 见下

不变量：同一笔事件在任一时刻只存在于一个容器里
---------------------------------------------
    IMPULSE_LEDGER_EXCLUSIVITY

一笔自报事件要么在 `impulse`（本轮的快速分量），要么在 `ledger`（已结转的慢账本），
绝不两边同时存在：

    写入一笔 clean：
        impulse = impulse × e^(-Δt / TAU_IMPULSE) + clean      # 衰减累加，不覆盖
        （此时不碰 ledger）
    下一轮开始（rollover）：
        把 impulse 的**现值**并进 ledger，然后清零 impulse
    读数：
        emotion(now) = clamp(mood(now) + impulse × e^(-(now - imp_ts) / TAU_IMPULSE))

于是"本轮"看到的增量只来自 impulse，`mood()` 只统计已结转的事件，两个集合不相交。
demo 里那版 `apply()` 同时把 delta 写进 ledger 又赋给 impulse，同一笔被算两次
（实测注入 +0.05、自报 +0.02、结算 +0.09，瞬时 2.0 倍），这个结构下不可能发生。

结转刻意用"已衰减值 + ts=now"而不是"原值 + ts=旧时刻"：轮次切换那一刻的读数连续，
不会因为事件从快轨（20min）换到慢轨（2h）而出现一小段向上反弹。两者都不违反上面的
不变量，取连续的那个。

边界
----
负向地板比正向天花板收得更紧（不对称），单笔再额外限幅。理由见 demo 的实测：
对称 ±1.0 时，一小时内 30 笔 -0.08 会把 v 钉死在 -1.000，情绪层从此只会说丧气话。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

# --------------------------------------------------------------------------- 常量

AXES: tuple[str, str, str] = ("v", "a", "d")

# 情绪底座。b 站上常见的那种"默认人格"：略微偏正、略偏唤醒、判断中性。
DEFAULT_TEMPER: dict[str, float] = {"v": 0.05, "a": 0.15, "d": 0.0}

# 不对称边界：负向收得更紧（-0.55 而不是 -1.0），正向留足（+0.85）
LO: dict[str, float] = {"v": -0.55, "a": 0.0, "d": -0.60}
HI: dict[str, float] = {"v": 0.85, "a": 1.0, "d": 0.60}

# 单笔上限：一次自报最多推动这么多，防止一笔 +0.22 就把 v 推到满量程 22%
PER_EVENT: dict[str, float] = {"v": 0.10, "a": 0.10, "d": 0.08}

TAU_EVENT: float = 2 * 3600.0      # 慢账本：2 小时
TAU_IMPULSE: float = 20 * 60.0     # 快分量：20 分钟

PRUNE_WEIGHT: float = 0.01         # 权重低于它的事件直接删掉
MAX_EVENTS: int = 2000             # 账本硬上限

# 语气词表（极小）：仅在模型没自报时兜底推动 trust。
_PRAISE = ("赞", "谢", "感谢", "厉害", "牛", "棒", "辛苦", "太好了", "搞定")
_BLAME = ("骂", "滚", "废物", "垃圾", "闭嘴", "蠢", "傻", "没用", "烦死")


# --------------------------------------------------------------------- 纯函数


def clamp_state(axis: str, value: float) -> float:
    """把三层合成后的值压进不对称边界。"""
    lo = LO[axis]
    hi = HI[axis]
    return max(lo, min(hi, value))


def clamp_event(axis: str, value: float) -> float:
    """单笔限幅：正负都不超过 PER_EVENT[axis]。"""
    cap = PER_EVENT[axis]
    return max(-cap, min(cap, value))


def clean_delta(delta: Mapping[str, float] | None) -> dict[str, float]:
    """把任意输入归一成三轴 dict，并做单笔限幅。缺失轴按 0。"""
    src: Mapping[str, float] = delta or {}
    out: dict[str, float] = {}
    for axis in AXES:
        try:
            raw = float(src.get(axis, 0.0) or 0.0)
        except (TypeError, ValueError):
            raw = 0.0
        if math.isnan(raw) or math.isinf(raw):
            raw = 0.0
        out[axis] = clamp_event(axis, raw)
    return out


def is_zero(delta: Mapping[str, float] | None) -> bool:
    """三轴全 0 —— 视为"本轮没有报到"。"""
    return all(abs(float((delta or {}).get(axis, 0.0) or 0.0)) < 1e-9 for axis in AXES)


def decay(dt: float, tau: float) -> float:
    """e^(-dt/tau)。dt<0（时钟回拨）按 0 处理，永不放大。"""
    if tau <= 0:
        return 1.0
    if dt <= 0:
        return 1.0
    return math.exp(-dt / tau)


def decay_at(ts: float, now: float, tau: float) -> float:
    return decay(now - ts, tau)


def mood_from(
    temper: Mapping[str, float],
    events: Iterable["Event"],
    now: float,
    tau_event: float = TAU_EVENT,
) -> dict[str, float]:
    """temper + Σ(事件 × 衰减)，再压进不对称边界。"""
    acc = {axis: float(temper.get(axis, DEFAULT_TEMPER[axis])) for axis in AXES}
    for ev in events:
        weight = decay_at(ev.ts, now, tau_event)
        if weight < PRUNE_WEIGHT:
            continue
        for axis in AXES:
            acc[axis] += ev.delta_of(axis) * weight
    return {axis: clamp_state(axis, acc[axis]) for axis in AXES}


def impulse_view(impulse: Mapping[str, float], imp_ts: float, now: float,
                 tau_impulse: float = TAU_IMPULSE) -> dict[str, float]:
    """impulse 在 `now` 时刻的剩余量（未结转部分）。imp_ts=0 表示没有待结转的冲动。"""
    if not imp_ts:
        return {axis: 0.0 for axis in AXES}
    weight = decay_at(imp_ts, now, tau_impulse)
    return {axis: float(impulse.get(axis, 0.0)) * weight for axis in AXES}


def emotion_from(
    mood: Mapping[str, float],
    impulse: Mapping[str, float],
    imp_ts: float,
    now: float,
    tau_impulse: float = TAU_IMPULSE,
) -> dict[str, float]:
    """emotion = clamp(mood + 本轮尚未结转的 impulse)。"""
    live = impulse_view(impulse, imp_ts, now, tau_impulse)
    return {axis: clamp_state(axis, float(mood.get(axis, 0.0)) + live[axis]) for axis in AXES}


def accumulate_impulse(
    impulse: Mapping[str, float],
    imp_ts: float,
    clean: Mapping[str, float],
    now: float,
    tau_impulse: float = TAU_IMPULSE,
) -> tuple[dict[str, float], float]:
    """衰减累加（不是覆盖）：上一笔按快时间常数衰减后与新一笔相加。"""
    weight = decay_at(imp_ts, now, tau_impulse) if imp_ts else 0.0
    out = {axis: float(impulse.get(axis, 0.0)) * weight + float(clean.get(axis, 0.0))
           for axis in AXES}
    return out, float(now)


def should_prune(ts: float, now: float, tau_event: float = TAU_EVENT) -> bool:
    """权重 < PRUNE_WEIGHT 的事件可以删了。"""
    return decay_at(ts, now, tau_event) < PRUNE_WEIGHT


def appraise(text: str) -> float:
    """极小的语气词表：只在模型没自报时兜底推动 trust。

    返回本次应给的 trust 步长（±0.01 ~ -0.02），无信号返回 0。
    """
    body = (text or "").strip()
    if not body:
        return 0.0
    if any(word in body for word in _BLAME):
        return -0.02
    if any(word in body for word in _PRAISE):
        return 0.01
    return 0.0


# ------------------------------------------------------------------------- 事件


@dataclass(slots=True)
class Event:
    """账本里的一条事件。时间戳是"结转时刻"，不是"自报时刻"。"""

    ts: float
    dv: float = 0.0
    da: float = 0.0
    dd: float = 0.0
    note: str = ""

    def delta_of(self, axis: str) -> float:
        if axis == "v":
            return self.dv
        if axis == "a":
            return self.da
        return self.dd


def event_from_dict(ts: float, delta: Mapping[str, float], note: str = "") -> Event:
    return Event(
        ts=float(ts),
        dv=float(delta.get("v", 0.0) or 0.0),
        da=float(delta.get("a", 0.0) or 0.0),
        dd=float(delta.get("d", 0.0) or 0.0),
        note=note,
    )


# --------------------------------------------------------------- 状态容器（无 IO）


@dataclass
class Emotion:
    """三层状态的容器。所有 IO 在 `store` 里；这里只有算术。

    `clock` 外部注入，演示脚本可以把 4 小时的间隔瞬间跑完。
    """

    temper: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TEMPER))
    ledger: list[Event] = field(default_factory=list)
    impulse: dict[str, float] = field(default_factory=lambda: {axis: 0.0 for axis in AXES})
    imp_ts: float = 0.0
    tau_event: float = TAU_EVENT
    tau_impulse: float = TAU_IMPULSE
    clock: Callable[[], float] | None = None

    def __post_init__(self) -> None:
        # 修法 6：少给轴不再 KeyError，而是用默认 temper 补齐
        self.temper = {**DEFAULT_TEMPER, **{k: float(v) for k, v in (self.temper or {}).items()}}
        self.impulse = {**{axis: 0.0 for axis in AXES},
                        **{k: float(v) for k, v in (self.impulse or {}).items()}}

    # ---- 时钟 ----

    def now(self) -> float:
        return float(self.clock()) if self.clock else time.time()

    # ---- 三层 ----

    def mood(self, now: float | None = None) -> dict[str, float]:
        return mood_from(self.temper, self.ledger, self.now() if now is None else now,
                         self.tau_event)

    def emotion(self, now: float | None = None) -> dict[str, float]:
        ts = self.now() if now is None else now
        return emotion_from(self.mood(ts), self.impulse, self.imp_ts, ts, self.tau_impulse)

    # ---- 写 ----

    def apply(self, delta: Mapping[str, float] | None, now: float | None = None,
              note: str = "") -> dict[str, float]:
        """记一笔自报。只写 impulse，不碰 ledger（不变量）。返回限幅后的实际值。"""
        ts = self.now() if now is None else now
        clean = clean_delta(delta)
        if is_zero(clean):
            # 全 0 = 未报到：不写 impulse、不重置 imp_ts
            return clean
        self.impulse, self.imp_ts = accumulate_impulse(
            self.impulse, self.imp_ts, clean, ts, self.tau_impulse)
        return clean

    def rollover(self, now: float | None = None) -> Event | None:
        """把上一轮的冲动结转进账本（已衰减值 + ts=now），然后清零。

        同一笔事件此刻从 impulse 搬到 ledger，仍然是"只在一个地方"。
        """
        ts = self.now() if now is None else now
        if not self.imp_ts:
            return None
        live = impulse_view(self.impulse, self.imp_ts, ts, self.tau_impulse)
        self.impulse = {axis: 0.0 for axis in AXES}
        self.imp_ts = 0.0
        if is_zero(live):
            return None
        ev = event_from_dict(ts, live, note="rollover")
        self.ledger.append(ev)
        return ev

    def prune(self, now: float | None = None, max_events: int = MAX_EVENTS) -> int:
        """剪枝：权重 < PRUNE_WEIGHT 的删掉，再按 ts 保住最近 max_events 条。"""
        ts = self.now() if now is None else now
        before = len(self.ledger)
        self.ledger = [e for e in self.ledger
                       if not should_prune(e.ts, ts, self.tau_event)]
        if max_events > 0 and len(self.ledger) > max_events:
            self.ledger.sort(key=lambda e: e.ts)
            self.ledger = self.ledger[-max_events:]
        return before - len(self.ledger)

    def halve(self) -> None:
        """朝 temper 收一半（on_session_end 的"睡一觉"）。"""
        for ev in self.ledger:
            ev.dv *= 0.5
            ev.da *= 0.5
            ev.dd *= 0.5
        self.impulse = {axis: self.impulse[axis] * 0.5 for axis in AXES}

    # ---- 审计 ----

    def explain(self, now: float | None = None) -> list[dict[str, object]]:
        """哪几笔在推 mood：按 |dv 贡献| 排序。"""
        ts = self.now() if now is None else now
        rows: list[dict[str, object]] = []
        for ev in self.ledger:
            weight = decay_at(ev.ts, ts, self.tau_event)
            if weight < PRUNE_WEIGHT:
                continue
            rows.append({
                "ts": ev.ts,
                "note": ev.note,
                "w": round(weight, 3),
                "dv": round(ev.dv * weight, 3),
                "da": round(ev.da * weight, 3),
                "dd": round(ev.dd * weight, 3),
            })
        rows.sort(key=lambda r: abs(float(r["dv"])), reverse=True)
        return rows
