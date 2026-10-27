"""CLI —— ``hermes emotion show|tail|explain|set``。

四个命令和钩子用的是同一个 ``Store``，所以手动调整与自动演化不可能出现两套语义。
``show`` 才给数字：那是给人看的；注入给模型的文本永远只有词（见 render.py）。
"""

from __future__ import annotations

import time

from . import config, core, render, store as store_mod

_settings: config.Settings = config.defaults()


def register(ctx) -> None:
    """注册 ``hermes emotion`` 子命令。"""
    global _settings
    try:
        _settings = config.from_ctx(ctx)
    except Exception:
        _settings = config.defaults()
    ctx.register_cli_command(
        "emotion",
        "查看情绪层状态",
        _setup,
        _handle,
        description="emotion_control：情绪三层（temper/mood/emotion）、自报审计与账本追溯",
    )


def _setup(sub) -> None:
    parser = sub.add_subparsers(dest="cmd", required=True)

    p = parser.add_parser("show", help="三轴当前值、temper、trust、事件数")
    p.add_argument("--user", default="cli:u_42", help="用户作用域，如 cli:u_42 或 demo:12345")

    p = parser.add_parser("tail", help="最近的自报审计（原始首行 + 解析结论）")
    p.add_argument("-n", type=int, default=20)

    p = parser.add_parser("explain", help="哪些事件在推 mood（按 |贡献| 排序）")
    p.add_argument("-n", type=int, default=20)

    p = parser.add_parser("set", help="手动改 temper（v/a/d 之一）")
    p.add_argument("axis", choices=list(core.AXES))
    p.add_argument("value", type=float)


def _handle(args) -> int:
    cmd = getattr(args, "cmd", "")
    try:
        with store_mod.Store.open(settings=_settings) as st:
            if cmd == "show":
                return _show(st, args)
            if cmd == "tail":
                return _tail(st, args)
            if cmd == "explain":
                return _explain(st, args)
            if cmd == "set":
                return _set(st, args)
    except KeyError as exc:
        print(f"✗ {exc.args[0] if exc.args else exc}")
        return 2
    except Exception as exc:                       # CLI 也不该因为库坏了抛栈
        print(f"✗ 读取情绪库失败：{exc}")
        return 2
    print(f"✗ 未知子命令：{cmd}")
    return 2


# --------------------------------------------------------------------- 命令


def _show(st: store_mod.Store, args) -> int:
    now = st.now()
    snap = st.snapshot(now)
    ukey = store_mod.parse_user_key(getattr(args, "user", "") or "")
    user = st.user(ukey) if ukey else {"trust": _settings.trust_base, "familiarity": 0.0,
                                       "turns": 0}

    print(f"情绪层 emotion_control   db: {_db_label(st)}")
    print("  " + "-" * 62)
    print(f"  {_pad('emotion', 9)}" + _axes(snap.emotion)
          + f"    {render.state_line(snap.emotion).replace('[此刻] ', '')}")
    print(f"  {_pad('mood', 9)}" + _axes(snap.mood))
    print(f"  {_pad('temper', 9)}" + _axes(snap.temper))
    live = core.impulse_view(snap.impulse, snap.imp_ts, now, _settings.tau_impulse)
    age = _dur(now - snap.imp_ts) if snap.imp_ts else "无"
    print(f"  {_pad('impulse', 9)}" + _axes(live) + f"   （待结转，{age}）")
    print(f"  {_pad('关系', 9)}信任 {user['trust']:.2f}"
          f"（{render.word_trust(float(user['trust']))}）"
          f"｜熟悉 {float(user['familiarity']):.2f}｜{int(user['turns'])} 轮"
          f"   {ukey or '（未指定用户）'}")
    total_reports = _count(st, "reports")
    print(f"  {_pad('账本', 9)}{snap.events} 条事件｜{total_reports} 条自报审计"
          f"｜τ_event={_settings.tau_event / 3600:.0f}h"
          f" τ_impulse={_settings.tau_impulse / 60:.0f}min")
    print(f"  {_pad('边界', 9)}v ∈ [{core.LO['v']:+.2f}, {core.HI['v']:+.2f}]"
          f"｜单笔 ≤ {core.PER_EVENT['v']:.2f}")
    return 0


def _tail(st: store_mod.Store, args) -> int:
    from . import protocol

    rows = st.reports(getattr(args, "n", 20))
    if not rows:
        print("(还没有自报记录)")
        return 0
    print(f"{'时间':<17}{'解析':<14}{'Δv':>7}{'Δa':>7}{'Δd':>7}  原始首行")
    print("-" * 84)
    for row in reversed(rows):
        stamp = time.strftime("%m-%d %H:%M:%S", time.localtime(row["ts"] or 0))
        verdict = str(row["verdict"] or "")
        print(f"{stamp:<17}{protocol.explain_reason(verdict):<14}"
              f"{float(row['dv'] or 0):>+7.2f}{float(row['da'] or 0):>+7.2f}"
              f"{float(row['dd'] or 0):>+7.2f}  {row['head'] or ''}")
    return 0


def _explain(st: store_mod.Store, args) -> int:
    now = st.now()
    rows = st.explain(now, limit=getattr(args, "n", 20))
    if not rows:
        print("(账本里没有还在起作用的这笔时间窗内的事件)")
        return 0
    snap = st.snapshot(now)
    print(f"temper {_axes(snap.temper)}   （底座，不随时间衰减）")
    print(f"mood   {_axes(snap.mood)}   （= temper + Σ 事件×权重）")
    print()
    print(f"{'时间':<17}{'权重':>6}{'dv':>8}{'da':>8}{'dd':>8}  来源")
    print("-" * 80)
    for row in rows:
        stamp = time.strftime("%m-%d %H:%M", time.localtime(float(row["ts"])))
        print(f"{stamp:<17}{float(row['w']):>6.2f}{float(row['dv']):>+8.3f}"
              f"{float(row['da']):>+8.3f}{float(row['dd']):>+8.3f}  {row['note'] or ''}")
    return 0


def _set(st: store_mod.Store, args) -> int:
    axis = str(args.axis)
    wanted = float(args.value)
    value = st.set_temper(axis, wanted, st.now())
    note = "" if abs(value - wanted) < 1e-9 else f"（超出 [{core.LO[axis]}, {core.HI[axis]}]，已压到边界）"
    print(f"temper.{axis} = {value:+.3f}{note}")
    snap = st.snapshot()
    print(f"此刻 emotion {_axes(snap.emotion)}")
    return 0


# --------------------------------------------------------------------- 小工具


def _axes(values: dict[str, float]) -> str:
    return (f"v {float(values.get('v', 0.0)):+.2f}   "
            f"a {float(values.get('a', 0.0)):+.2f}   "
            f"d {float(values.get('d', 0.0)):+.2f}")


def _pad(text: str, width: int) -> str:
    """按终端显示宽度左对齐（中日韩字符占两列）。"""
    import unicodedata

    shown = sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)
    return text + " " * max(0, width - shown)


def _count(st: store_mod.Store, table: str) -> int:
    try:
        row = st.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"] if row else 0)
    except Exception:
        return 0


def _db_label(st: store_mod.Store) -> str:
    try:
        for row in st.conn.execute("PRAGMA database_list"):
            if str(row["name"]) == "main":
                return str(row["file"])
    except Exception:
        pass
    return str(store_mod.db_path())


def _dur(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{int(seconds)} 秒前"
    if seconds < 5400:
        return f"{int(seconds // 60)} 分钟前"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} 小时前"
    return f"{seconds / 86400:.1f} 天前"
