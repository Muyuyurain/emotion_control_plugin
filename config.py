"""配置 —— 全部走 ``ctx.get_config()``。

键位在 ``plugins.entries.emotion_control.settings`` 下（Hermes 的插件相对路径约定）：

    plugins:
      entries:
        emotion_control:
          settings:
            enabled: true
            injection_budget: 400
            dedupe_ttl_seconds: 300

所有读取都 fail-open：拿不到配置就用默认值，插件不该因为读配置失败而罢工。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import core


@dataclass(slots=True)
class Settings:
    """插件参数。默认值就是规格里定好的那套。"""

    # 总开关：关掉后四个钩子都直接返回 None（仍注册，但不做事）
    enabled: bool = True
    # 注入文本的字符预算。两条 [边界] 永不裁剪，可选行先丢。
    injection_budget: int = 400
    # 慢账本 / 快分量的时间常数（秒）
    tau_event: float = core.TAU_EVENT
    tau_impulse: float = core.TAU_IMPULSE
    # 账本硬上限与剪枝阈值（权重低于阈值即删）
    max_events: int = core.MAX_EVENTS
    prune_weight: float = core.PRUNE_WEIGHT
    # 幂等去重窗口：同一 (session_id, sha1(回复全文)) 在这个窗口内只记一次
    # 默认 300s，覆盖 平台被动回复超时重发的常见间隔
    dedupe_ttl_seconds: float = 300.0
    # trust 的可动区间与基线
    trust_base: float = 0.35
    trust_lo: float = 0.12
    trust_hi: float = 0.95
    # 单轮 trust 最大步长
    trust_step: float = 0.02
    # 每轮 familiarity 增量（只增，上限 1）
    familiarity_step: float = 0.01
    # "睡一觉"的最小间隔（秒）。Hermes 的 on_session_end 每轮都触发，靠这个闸门限频。
    # 3h 而不是 8h：τ_event=2h，隔 8 小时账本只剩 1.8%，折半等于没做。
    halve_min_gap: float = 3 * 3600.0
    # 高风险任务（删库/改防火墙/提权…）跳过本轮注入，情绪层闭嘴
    skip_on_risk: bool = True


def defaults() -> Settings:
    """离线场景（demo / 单测 / CLI）用的默认配置。"""
    return Settings()


def from_ctx(ctx) -> Settings:
    """从 ``register(ctx)`` 拿到的上下文里读配置；任何异常都退默认值。"""

    def get(key: str, default):
        try:
            value = ctx.get_config(key, default=default)
        except Exception:
            return default
        return default if value is None else value

    base = Settings()
    try:
        budget = int(get("injection_budget", base.injection_budget))
    except (TypeError, ValueError):
        budget = base.injection_budget
    try:
        tau_event = float(get("tau_event", base.tau_event))
    except (TypeError, ValueError):
        tau_event = base.tau_event
    try:
        tau_impulse = float(get("tau_impulse", base.tau_impulse))
    except (TypeError, ValueError):
        tau_impulse = base.tau_impulse
    try:
        max_events = int(get("max_events", base.max_events))
    except (TypeError, ValueError):
        max_events = base.max_events
    try:
        dedupe_ttl = float(get("dedupe_ttl_seconds", base.dedupe_ttl_seconds))
    except (TypeError, ValueError):
        dedupe_ttl = base.dedupe_ttl_seconds

    return Settings(
        enabled=bool(get("enabled", base.enabled)),
        injection_budget=max(120, budget),
        tau_event=tau_event if tau_event > 0 else base.tau_event,
        tau_impulse=tau_impulse if tau_impulse > 0 else base.tau_impulse,
        max_events=max(1, max_events),
        prune_weight=float(get("prune_weight", base.prune_weight) or base.prune_weight),
        dedupe_ttl_seconds=max(0.0, dedupe_ttl),
        trust_base=float(get("trust_base", base.trust_base) or base.trust_base),
        trust_lo=float(get("trust_lo", base.trust_lo) or base.trust_lo),
        trust_hi=float(get("trust_hi", base.trust_hi) or base.trust_hi),
        trust_step=float(get("trust_step", base.trust_step) or base.trust_step),
        familiarity_step=float(get("familiarity_step", base.familiarity_step)
                               or base.familiarity_step),
        skip_on_risk=bool(get("skip_on_risk", base.skip_on_risk)),
    )
