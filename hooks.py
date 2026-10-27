"""钩子适配器 —— 全部 fail-open。

这里没有一行业务算术：取参数、走 ``begin_turn`` / ``apply_output`` 两个共享函数、
异常吞掉。用户的一次对话不该因为一个情绪插件打不开而失败。

四个钩子
--------
pre_llm_call          新一轮开始：把上一轮的冲动结转进账本，注入"此刻"的词（不给数字）。
                      子 agent 直接跳过 —— 它们有自己的 session，让它们参与会把父对话的
                      情绪搅乱。
transform_llm_output  剥壳 + 解析 + 记账，返回剥壳后的正文（返回 None 表示不替换）。
post_tool_call        只累加到内存 dict，给 event 的 note 打标签，不落库。
on_session_end        把情绪朝 temper 收一半（"睡一觉"）。

幂等
----
``transform_llm_output`` 的运行时签名里**是有 turn_id 的**（`agent/turn_finalizer.py`
显式传 `turn_id=turn_id`，官方文档的钩子表格漏写了这一项），所以去重键优先用
``(session_id, turn_id)``：同一轮重发会被挡掉，而模型连着两轮说同一句话
（都回"好的"）仍然各记一笔。拿不到 turn_id 的老版本才退回 ``sha1(全文)``。

临时状态
--------
``pre_llm_call`` 与 ``transform_llm_output`` 之间要传两样东西：本轮用户消息（给
trust 的语气兜底）和用户作用域键（transform 的签名里没有 sender_id）。
两者都按 ``(session_id, turn_id)`` 存 —— 只按 session 存的话，两个群同时说话会把
A 的自报记到 B 头上（实测 ``pre(A) → pre(B) → transform(A)`` 记账落到 B）。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from . import config, protocol, render, risk, store as store_mod

logger = logging.getLogger(__name__)

_settings = config.defaults()

_PendingKey = tuple[str, str]          # (session_id, turn_id)

# (session, turn) -> 已 scrub 的本轮用户消息，给 trust 的语气兜底用
_PENDING_USER: dict[_PendingKey, str] = {}
# (session, turn) -> 用户作用域键。transform_llm_output 的签名里没有 sender_id，
# 所以 pre_llm_call 先把它记下来，记账时才不会写到"anonymous"头上。
_PENDING_UKEY: dict[_PendingKey, str] = {}
# session -> 上一轮用户消息。begin_turn 结转时要给 event 打标签，只按 session 存就够
# （它永远只关心"最近一句话"）。
_LAST_USER: dict[str, str] = {}
# session -> 本轮用过的工具名
_PENDING_TOOLS: dict[str, list[str]] = {}
_MAX_PENDING = 256
_LOCK = threading.Lock()


def _trim(mapping: dict, limit: int = _MAX_PENDING) -> None:
    """超限时丢最早的一条（dict 保序）。调用方必须已持锁。"""
    while len(mapping) > limit:
        mapping.pop(next(iter(mapping)), None)


# --------------------------------------------------------------------- 装配


def configure(ctx) -> None:
    """注册时调用一次：把 ctx 上的配置读进来。"""
    global _settings
    try:
        _settings = config.from_ctx(ctx)
    except Exception as exc:                       # fail-open
        logger.warning("emotion_control: 读取配置失败，改回默认值：%s", exc)
        _settings = config.defaults()


def settings() -> config.Settings:
    return _settings


def reset_memory() -> None:
    """清掉内存里的临时状态（测试 / 会话收尾用）。"""
    with _LOCK:
        _PENDING_USER.clear()
        _PENDING_UKEY.clear()
        _LAST_USER.clear()
        _PENDING_TOOLS.clear()


def forget_session(session_id: str) -> None:
    """只清这个 session 的临时状态（收尾用；不能误清别人的）。"""
    if not session_id:
        return
    with _LOCK:
        _LAST_USER.pop(session_id, None)
        _PENDING_TOOLS.pop(session_id, None)
        for bucket in (_PENDING_USER, _PENDING_UKEY):
            for key in [k for k in bucket if k[0] == session_id]:
                bucket.pop(key, None)


def pending_user_key(session_id: str, turn_id: str = "") -> str:
    """本轮的用户作用域键（pre_llm_call 记下来的）。读走即删。"""
    with _LOCK:
        if turn_id:
            found = _PENDING_UKEY.pop((session_id, turn_id), "")
            if found:
                return found
        # 兜底：这一轮没带 turn_id 时退回 session 级槽位
        return _PENDING_UKEY.get((session_id, ""), "")


# ----------------------------------------------------------- 共享管线（demo 也用）


def begin_turn(
    st: store_mod.Store,
    now: float,
    *,
    session_id: str = "",
    turn_id: str = "",
    user_message: str = "",
    ukey: str = "",
    budget: int | None = None,
) -> str:
    """一轮开始时该做的事：结转冲动 → 记住用户消息 → 渲染注入文本。

    demo 和 pre_llm_call 走的是同一个函数，所以"演示里的语义"就是"线上的语义"。
    """
    with _LOCK:
        tools = _PENDING_TOOLS.pop(session_id, [])
        prev_user = _LAST_USER.get(session_id, "")

    # 结转时给 event 打上"上一轮发生了什么"的标签，方便 explain 追溯
    st.begin_turn(now, note=_label(prev_user, tools))

    scrubbed = protocol.scrub_user_text(user_message)
    with _LOCK:
        if session_id:
            _LAST_USER[session_id] = scrubbed
        _PENDING_USER[(session_id, turn_id)] = scrubbed
        if ukey:
            _PENDING_UKEY[(session_id, turn_id)] = ukey
            _PENDING_UKEY[(session_id, "")] = ukey     # turn_id 缺失时的兜底槽
        _trim(_PENDING_USER)
        _trim(_PENDING_UKEY)
        _trim(_LAST_USER)

    snap = st.snapshot(now)
    user = st.user(ukey) if ukey else {
        "trust": settings().trust_base, "familiarity": 0.0}
    limit = settings().injection_budget if budget is None else budget
    return render.context(
        snap.emotion,
        trust=float(user["trust"]),
        familiarity=float(user["familiarity"]),
        extra=(),
        budget=limit,
    )


def apply_output(
    st: store_mod.Store,
    raw: str,
    now: float,
    *,
    session_id: str = "",
    turn_id: str = "",
    ukey: str = "",
    user_text: str = "",
    key: str = "",
) -> str | None:
    """剥壳 + 记账，返回要替换成的正文（None 表示保持原样）。

    返回值的规则（都不许返回空串 —— 空串在 Hermes 里等于"不替换"，但更危险的是
    语义上会被人当成"清空回复"）：
      * 空正文（只有标签）→ 占位符（不能是原文，那样会把 @E 漏进群里）
      * 没标签 → None，一字不动
      * 有标签且剥出了正文 → 返回正文
    """
    clean, delta, reason = protocol.parse(raw)
    if user_text:
        pending = protocol.scrub_user_text(user_text)
    else:
        with _LOCK:
            pending = _PENDING_USER.pop((session_id, turn_id), "") \
                or _PENDING_USER.get((session_id, ""), "")
    # transform 的签名里没有 sender_id：优先用 pre_llm_call 记下的作用域键
    target = ukey or pending_user_key(session_id, turn_id)

    try:
        st.apply_report(
            delta,
            now=now,
            session_id=session_id,
            note=_label(pending, []),
            user_text=pending,
            verdict=reason,
            head=protocol.first_line(raw),
            ukey=target,
            key=key or store_mod.dedupe_key(session_id, raw, turn_id),
        )
    except Exception as exc:                       # 记账失败不该影响回复
        logger.warning("emotion_control: 记账失败（回复照常投递）：%s", exc)

    if reason == protocol.REASON_EMPTY_BODY:
        return protocol.EMPTY_BODY_FALLBACK
    if clean == raw:
        return None
    if not clean.strip():
        return None
    return clean


def note_tool(session_id: str, tool_name: str) -> None:
    """post_tool_call 的落地：只在内存里记名字。"""
    name = (tool_name or "").strip()
    if not name or not session_id:
        return
    with _LOCK:
        bucket = _PENDING_TOOLS.setdefault(session_id, [])
        if len(bucket) < 32:
            bucket.append(name)
        if len(_PENDING_TOOLS) > _MAX_PENDING:
            _PENDING_TOOLS.pop(next(iter(_PENDING_TOOLS)), None)


def _label(user_text: str, tools: list[str]) -> str:
    """event 的 note：上一轮用户说了什么 + 用了什么工具。都没有就留空。"""
    body = (user_text or "").strip().replace("\n", " ")
    head = body[:12]
    if not tools:
        return head
    uniq: list[str] = []
    for name in tools:
        if name not in uniq:
            uniq.append(name)
    tag = f"tool:{','.join(uniq[:3])}"
    return f"{head}｜{tag}" if head else tag


# --------------------------------------------------------------------- 四个钩子


def pre_llm_call(**kw: Any):
    """注入"此刻"。返回 ``{"context": text}``，或 None（不注入）。"""
    try:
        if not settings().enabled:
            return None
        # 子 agent 直接跳过：它们有自己的 session，父对话的情绪不该被它们推动
        if kw.get("parent_session_id"):
            return None
        session_id = str(kw.get("session_id") or "")
        sender_id = str(kw.get("sender_id") or "")
        platform = str(kw.get("platform") or "")
        user_message = str(kw.get("user_message") or "")
        now = _now()
        with store_mod.Store.open(settings=settings()) as st:
            text = begin_turn(
                st, now,
                session_id=session_id,
                turn_id=str(kw.get("turn_id") or ""),
                user_message=user_message,
                ukey=store_mod.user_key(sender_id, platform) if sender_id else "",
            )
        # 高风险任务：情绪层闭嘴。结转照做（账本要连续），但一个字都不注入。
        if settings().skip_on_risk:
            hit = risk.match(user_message)
            if hit:
                logger.info("emotion_control: 高风险任务（%s），本轮跳过注入", hit)
                return None
        return {"context": text} if text else None
    except Exception as exc:                       # fail-open
        logger.warning("emotion_control: pre_llm_call 跳过：%s", exc)
        return None


def transform_llm_output(response_text: str = "", session_id: str = "", model: str = "",
                         platform: str = "", **kw: Any):
    """剥掉 @E 行并记账。返回替换后的正文，None 表示保持原样。

    顺序刻意是"先剥壳、再落库"：剥壳不依赖数据库，没有任何理由让它在 DB 打不开时
    一起失败、把 `@E v=-0.06` 原样投进群里（QA 实测过这条泄漏路径）。
    """
    try:
        if not settings().enabled:
            return None
        raw = response_text if isinstance(response_text, str) else str(response_text or "")
        if not raw.strip():
            return None
        sid = str(session_id or kw.get("session_id") or "")
        tid = str(kw.get("turn_id") or "")     # 运行时确实会传，见模块 docstring

        clean, _delta, reason = protocol.parse(raw)

        try:
            now = _now()
            with store_mod.Store.open(settings=settings()) as st:
                apply_output(st, raw, now, session_id=sid, turn_id=tid)
        except Exception as exc:                   # 记账失败不该影响投递
            logger.warning("emotion_control: 记账失败（正文照常投递）：%s", exc)

        if reason == protocol.REASON_EMPTY_BODY:
            return protocol.EMPTY_BODY_FALLBACK
        if clean == raw or not clean.strip():
            return None
        return clean
    except Exception as exc:                       # fail-open
        logger.warning("emotion_control: transform_llm_output 跳过：%s", exc)
        return None


def post_tool_call(**kw: Any) -> None:
    """只记内存，用于 event note 的标签。"""
    try:
        if not settings().enabled:
            return
        note_tool(str(kw.get("session_id") or ""), str(kw.get("tool_name") or ""))
    except Exception as exc:                       # fail-open
        logger.warning("emotion_control: post_tool_call 跳过：%s", exc)


def on_session_end(**kw: Any) -> None:
    """收尾的"睡一觉"：情绪朝 temper 收一半，坏日子不带进下一次。

    ⚠️ 这个钩子的名字是误导：Hermes 里 ``on_session_end`` **每轮对话结束都会触发一次**
    （``agent/turn_finalizer.py`` 的 ``finalize_turn()``，而它由 ``conversation_loop``
    每条消息调一次），不是"会话结束"。所以必须走 ``halve_if_due`` —— 闸门内的收束才会
    真正执行，否则每说一句话账本就被砍半，情绪永远攒不起来。

    同理，**不能在这里 ``forget_session()``**：它每轮都会触发，把 ``_LAST_USER`` 和
    ``_PENDING_TOOLS`` 每轮清空，结果是账本里所有 event 的标签都变成 "rollover"，
    ``hermes emotion explain`` 再也说不出"这笔是跟哪句话走的"。临时表已经按
    ``(session_id, turn_id)`` 存、用完即删，靠 ``_trim`` 兜住上限就够了。
    """
    try:
        if not settings().enabled:
            return
        now = _now()
        with store_mod.Store.open(settings=settings()) as st:
            if st.halve_if_due(now, min_gap=settings().halve_min_gap):
                logger.info("emotion_control: 情绪收束（距上次 ≥ %ss）", settings().halve_min_gap)
    except Exception as exc:                       # fail-open
        logger.warning("emotion_control: on_session_end 跳过：%s", exc)


def _now() -> float:
    return time.time()
