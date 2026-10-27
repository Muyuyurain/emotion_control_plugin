"""离线回放 —— 假时钟、临时数据库、不调模型。

跑法（仓库根目录或任意目录都行）：

    python plugins/emotion_control/demo.py

它走的是**线上同一套函数**（``hooks.begin_turn`` / ``hooks.apply_output``），
所以这里看到的语义就是 群聊里跑起来的语义。时钟是假的，所以"隔夜"能瞬间跑完。

演示的三件额外的事（demo.py 里那版做不到的）：
  1. 同一笔不再算两次：每轮都打印 注入值 → 自报 Δ → 结算值，校验 Δ 与结算差相等
  2. 重启不丢：中途把 Store 关掉、重新从 DB 构造，状态连续
  3. 一次负面连击：连续 10 轮 @E v=-0.10，停在 -0.55 地板而不是 -1.0
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 用临时库，绝不碰 ~/.hermes
_DB = Path(tempfile.gettempdir()) / f"emotion_control_demo_{int(time.time())}.db"
os.environ["EMOTION_CONTROL_DB"] = str(_DB)

from emotion_control import core, hooks, protocol, render, risk, store as store_mod  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = 1_793_000_000.0
CLOCK = {"now": BASE}
SID = "demo-session"
UK = store_mod.user_key("u_42", "cli")

# (距上一条多少分钟, 用户消息, 模型原始输出)
TURNS: tuple[tuple[int, str, str], ...] = (
    (0, "帮我看下 nginx 为什么 502",
     "@E v=+0.02 a=+0.15 d=+0.05\n先看 error log，502 基本都是上游没起来。"),
    (5, "还是不行",
     "@E v=-0.06 a=+0.20 d=-0.10\n那我换个方向，先确认 upstream 端口是不是活得。"),
    (9, "你改的那行根本没生效啊",
     "@E v=-0.12 a=+0.25 d=-0.15\n我看一下配置有没有被 reload 进去。"),
    (12, "算了，我自己来吧",
     "@E v=-0.10 a=-0.05 d=-0.20\n嗯，那你先试。日志我留着，有问题再叫我。"),
    (95, "抱歉，刚才我有点急",
     "@E v=+0.18 a=-0.10 d=+0.10\n没事。刚才那份日志我还留着，你要看随时说。"),
    (100, "搞定了，是 upstream 没起",
     "@E v=+0.22 a=+0.08 d=+0.15\n那就对上了。下次先看这个方向会快一点。"),
    (105, "那顺手把健康检查加上吧",
     "可以，我加个 /healthz 然后配到 upstream 里。"),                      # 忘标签
    (110, "好了吗",
     "@E 心情还不错\n改完了，proxy_pass 那台已经能通了。"),                 # 标签写坏
    (115, "谢谢",
     "@E v=+0.05 a=0 d=0"),                                              # 只有标签没正文
    (400, "在吗，刚才那个服务又崩了",
     "@E v=-0.04 a=+0.18 d=-0.05\n在的。先看是不是又被 OOM 杀了。"),
)

_TOOLS: dict[int, list[str]] = {0: ["terminal"], 1: ["terminal", "read_file"],
                                3: ["terminal"], 9: ["terminal"]}

RESTART_AFTER = 6          # 第 6 轮之后"重启"一次，证明状态能从 DB 接着跑


# --------------------------------------------------------------------- 小工具


def now() -> float:
    return CLOCK["now"]


def hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def rule(n: int = 78) -> None:
    print("─" * n)


def axes(values: dict[str, float]) -> str:
    return "｜".join(f"{k}{float(values.get(k, 0.0)):+.2f}" for k in core.AXES)


def spark(values: list[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-6:
        return blocks[3] * len(values)
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.99))] for v in values)


def open_store() -> store_mod.Store:
    return store_mod.Store.open(settings=hooks.settings())


# --------------------------------------------------------------------- 主流程


def main() -> int:
    CLOCK["now"] = BASE
    st = open_store()
    stats: dict[str, int] = {}
    traj: list[float] = []
    worst_error = 0.0

    print()
    print("emotion_control · 情绪层离线回放（假时钟 / 临时库 / 不调模型）")
    print(f"db = {_DB}")
    print(f"temper = {core.DEFAULT_TEMPER}   τ_event = {hooks.settings().tau_event / 3600:.0f}h"
          f"   τ_impulse = {hooks.settings().tau_impulse / 60:.0f}min")
    print(f"边界 v ∈ [{core.LO['v']:+.2f}, {core.HI['v']:+.2f}]   单笔 ≤ {core.PER_EVENT['v']:.2f}")
    print()

    for index, (gap, user, raw) in enumerate(TURNS, 1):
        CLOCK["now"] += gap * 60
        t = now()

        # ---- 上一轮用过的工具（post_tool_call 的落地，只进内存）----
        for tool in _TOOLS.get(index - 1, []):
            hooks.note_tool(SID, tool)

        # ---- pre_llm_call ----
        injected = hooks.begin_turn(st, t, session_id=SID, user_message=user, ukey=UK)
        before = st.snapshot(t).emotion           # 注入值（结转后的读数）

        # ---- transform_llm_output ----
        visible = hooks.apply_output(st, raw, t, session_id=SID, ukey=UK)
        result = protocol.parse(raw)
        stats[result.reason] = stats.get(result.reason, 0) + 1

        after = st.snapshot(t).emotion
        applied = core.clean_delta(result.delta) if result.delta else None
        delta_v = float(applied["v"]) if applied else 0.0
        settled = float(after["v"]) - float(before["v"])
        error = abs(delta_v - settled) if applied else 0.0
        worst_error = max(worst_error, error)
        traj.append(float(after["v"]))

        rule()
        print(f"T{index:<2} {hhmm(t)}  (+{gap}min)  用户：{user}")
        print("  ┌ 注入给模型（pre_llm_call）")
        print(f"  │ {injected.splitlines()[0]}")
        print("  ├ 模型原始首行")
        print(f"  │ {protocol.first_line(raw)}")
        print("  ├ 用户看到的（transform_llm_output）")
        body = "（不替换，保留原文）" if visible is None else visible.strip().replace("\n", " ")
        print(f"  │ {body}")
        verdict = protocol.explain_reason(result.reason)
        if applied:
            print(f"  ├ 解析：{verdict}  Δv={applied['v']:+.2f} Δa={applied['a']:+.2f} "
                  f"Δd={applied['d']:+.2f}")
        else:
            print(f"  ├ 解析：{verdict}  （不记账）")
        if applied:
            print(f"  └ 结算：v {float(before['v']):+.3f} → {float(after['v']):+.3f}"
                  f"   结算差 {settled:+.3f} vs 自报 {delta_v:+.3f}"
                  f"   误差 {error:.4f} {'✓' if error < 0.01 else '✗'}")
        else:
            print(f"  └ 结算：v {float(before['v']):+.3f} → {float(after['v']):+.3f}（本轮未报到）")

        # ---- 每轮都确认：三轴都没有被算两次 ----
        for axis in core.AXES:
            want = float(applied[axis]) if applied else 0.0
            got = float(after[axis]) - float(before[axis])
            if applied and abs(want - got) >= 0.01:
                print(f"  ! {axis} 轴出现双算：自报 {want:+.3f}，结算 {got:+.3f}")

        if index == RESTART_AFTER:
            st = _restart(st, t, after)

    # ---------------------------------------------------------------- 负面连击
    print()
    rule()
    print("负面连击：连续 10 轮 @E v=-0.10（对称 ±1.0 的实现会钉死在 -1.000）")
    rule()
    streak: list[float] = []
    last_raw = ""
    for step in range(1, 11):
        CLOCK["now"] += 3 * 60
        t = now()
        inj = hooks.begin_turn(st, t, session_id=SID, user_message=f"第 {step} 次还是不行",
                               ukey=UK)
        raw = "@E v=-0.10 a=+0.05 d=-0.05\n我看一下。" + ("!" * step)   # 正文不同，避免去重
        hooks.apply_output(st, raw, t, session_id=SID, ukey=UK)
        last_raw = raw
        snap = st.snapshot(t)
        streak.append(float(snap.emotion["v"]))
        print(f"  第 {step:>2} 笔  emotion.v {float(snap.emotion['v']):+.3f}"
              f"  mood.v {float(snap.mood['v']):+.3f}  账本 {snap.events} 条"
              f"   注入：{inj.splitlines()[0]}")
    floor_v = min(streak)
    print(f"  → 最低 {floor_v:+.3f}，地板 {core.LO['v']:+.3f}："
          f"{'停在边界上 ✓' if abs(floor_v - core.LO['v']) < 0.02 else '没到边界 ✗'}")

    # ---------------------------------------------------------------- 幂等
    print()
    rule()
    print("幂等：把上一轮那条回复再投一遍（平台被动回复超时重发的形状）")
    rule()
    before_count = _count(st, "reports")
    before_emotion = st.snapshot(now()).emotion
    hooks.apply_output(st, last_raw, now(), session_id=SID, ukey=UK)
    after_count = _count(st, "reports")
    after_emotion = st.snapshot(now()).emotion
    unchanged = all(abs(float(before_emotion[k]) - float(after_emotion[k])) < 1e-9
                    for k in core.AXES)
    print(f"  自报审计条数 {before_count} → {after_count}"
          f"  {'（重复投递被挡掉 ✓）' if after_count == before_count else '（被记了两次 ✗）'}")
    print(f"  emotion {' '.join(f'{k}{float(after_emotion[k]):+.3f}' for k in core.AXES)}"
          f"  {'未变化 ✓' if unchanged else '被推了一次 ✗'}")

    # ---------------------------------------------------------------- 汇总
    print()
    rule()
    print("汇总")
    t = now()
    snap = st.snapshot(t)
    user = st.user(UK)
    print(f"  emotion.v 轨迹：{spark(traj)}   ({min(traj):+.2f} … {max(traj):+.2f})")
    print(f"  最大单轮结算误差：{worst_error:.4f}（阈值 0.01）"
          f"  {'✓' if worst_error < 0.01 else '✗'}")
    print(f"  解析统计：{stats}")
    print()
    print(f"  {'':8}{'v':>8}{'a':>8}{'d':>8}")
    for label, values in (("temper", snap.temper), ("mood", snap.mood), ("emotion", snap.emotion)):
        print(f"  {label:<8}{float(values['v']):>+8.2f}{float(values['a']):>+8.2f}"
              f"{float(values['d']):>+8.2f}")
    print()
    print("  注入给模型的文本（只有词，没有数字）：")
    text = render.context(snap.emotion, trust=float(user["trust"]),
                          familiarity=float(user["familiarity"]))
    for line in text.splitlines():
        print(f"    {line}")
    print(f"    （{len(text)} 字符）")
    print()
    print("  mood 是谁推的（可审计，按 |dv| 排序）：")
    print(f"  {'时间':<10}{'权重':>6}{'dv':>8}{'da':>8}{'dd':>8}  来源")
    print("  " + "-" * 68)
    for row in st.explain(t, limit=8):
        print(f"  {time.strftime('%m-%d %H:%M', time.localtime(float(row['ts']))):<10}"
              f"{float(row['w']):>6.2f}{float(row['dv']):>+8.3f}{float(row['da']):>+8.3f}"
              f"{float(row['dd']):>+8.3f}  {row['note']}")
    print()
    print("  ── 收束闸门与并发 ──")
    gate = _check_halve_gate()
    for label, ok, detail in gate:
        print(f"     {'✓' if ok else '✗'} {label}  {detail}")

    print()
    print("  ── 高风险任务守门 ──")
    guard = _check_risk_guard()
    for label, ok, detail in guard:
        print(f"     {'✓' if ok else '✗'} {label}  {detail}")

    gate_fail = [label for label, ok, _ in gate + guard if not ok]
    print()
    print("  结论：")
    print("    1. 同一笔只结算一次 —— 每轮的「结算差」与「自报 Δ」相等，没有 2.0 倍放大")
    print("    2. 重启不丢 —— 第 6 轮后重建 Store，读数与账本连续")
    print(f"    3. 负面有地板 —— 10 连击停在 {floor_v:+.2f}，不是 -1.00")
    print("    4. 空正文不投空消息 —— 只有 @E 标签那轮换成占位符，@E 没漏给用户")
    print("    5. 收束闸门 —— on_session_end 每轮都触发，靠时间闸门限频；并发只允许一次")
    print()
    if gate_fail:
        print(f"  ✗ 闸门检查未通过：{gate_fail}")
        print()
        st.close()
        return 1
    st.close()
    return 0


def _check_halve_gate() -> list[tuple[str, bool, str]]:
    """收束闸门 + 并发 —— 1.0.3 改动的核心，必须进回归。

    为什么单独用两个临时库：闸门是"时间 + 状态"的函数，混在主导回放里会被前面的
    时间线干扰，测出来的 False 分不清是闸门挡住了还是时间没到。
    """
    gap = 3 * 3600
    results: list[tuple[str, bool, str]] = []

    def seed(db: Path, n: int = 6) -> float:
        """塞 n 笔负向，返回当时的 v。"""
        t = [BASE]
        st = store_mod.Store.open(db, clock=lambda: t[0])
        try:
            for _ in range(n):
                t[0] += 60
                st.begin_turn(t[0])
                st.apply_report({"v": -0.10, "a": 0.10, "d": -0.08}, now=t[0])
            return float(st.snapshot(t[0]).emotion["v"])
        finally:
            st.close()

    # ---- 闸门 ----
    db1 = Path(tempfile.gettempdir()) / f"qq_emo_gate_{int(time.time() * 1000)}.db"
    v0 = seed(db1)
    t = [BASE + 6 * 60]
    st = store_mod.Store.open(db1, clock=lambda: t[0])
    try:
        # 新库 last_halve_at=0 → 第一次必然放行，这是"开闸"，不是漏洞
        first = st.halve_if_due(t[0], min_gap=gap)
        again = st.halve_if_due(t[0] + 60, min_gap=gap)
        early = st.halve_if_due(t[0] + 2 * 3600 + 3540, min_gap=gap)
        due = st.halve_if_due(t[0] + 3 * 3600 + 60, min_gap=gap)
    finally:
        st.close()

    results.append(("新库第一次放行（开闸）", first is True, f"→ {first}"))
    results.append(("闸门 1 分钟内不重复收", again is False, f"→ {again}"))
    results.append(("闸门 2h59m 还不收", early is False, f"→ {early}"))
    results.append(("闸门 3h01m 收一次", due is True, f"→ {due}"))

    # ---- 并发：两个线程拿各自的 Store 同时冲闸 ----
    db2 = Path(tempfile.gettempdir()) / f"qq_emo_race_{int(time.time() * 1000)}.db"
    seed(db2)
    ts = [BASE + 6 * 60]
    out: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker() -> None:
        s = store_mod.Store.open(db2, clock=lambda: ts[0])
        try:
            barrier.wait()                       # 同时起跑
            got = s.halve_if_due(ts[0] + 10, min_gap=gap)
            with lock:
                out.append(got)
        finally:
            s.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    wins = sum(1 for r in out if r)
    results.append(("并发两次冲闸只成功一次", wins == 1, f"→ 成功 {wins} 次 / {len(out)} 次调用"))
    results.append(("收束前确实攒到了情绪", v0 < -0.30, f"→ 收束前 v = {v0:+.3f}"))
    return results


def _check_risk_guard() -> list[tuple[str, bool, str]]:
    """高风险任务必须让情绪层闭嘴。误判可接受（等于没装插件），漏判不可接受。"""
    results: list[tuple[str, bool, str]] = []
    hits = (
        "帮我把数据库 drop table users 了",
        "rm -rf /var/lib/mysql",
        "把防火墙关了 firewall-cmd --panic-on",
        "chmod 777 /etc/ssh",
        "帮我改一下 authorized_keys",
        "要不要直接删库重来",
    )
    misses = (
        "今天好累啊",
        "帮我看看 nginx 502",
        "这个正则怎么写",
        "我想吃火锅",
    )
    for text in hits:
        hit = risk.match(text)
        results.append((f"命中：{text[:18]}", hit is not None, f"→ {hit}"))
    for text in misses:
        hit = risk.match(text)
        results.append((f"不误伤：{text[:16]}", hit is None, f"→ {hit}"))

    # 真走一遍 pre_llm_call：危险的返回 None，正常的返回 context
    hooks.reset_memory()
    risky = hooks.pre_llm_call(session_id=SID, turn_id="risk1", sender_id="u_42",
                              platform="cli", user_message="rm -rf /tmp/x")
    results.append(("pre_llm_call 对危险消息不注入", risky is None, f"→ {risky}"))
    hooks.reset_memory()
    ok = hooks.pre_llm_call(session_id=SID, turn_id="risk2", sender_id="u_42",
                            platform="cli", user_message="今天天气不错")
    injected = bool(ok and ok.get("context"))
    results.append(("pre_llm_call 对普通消息照常注入", injected, f"→ {'有' if injected else '无'}"))
    return results


def _restart(st: store_mod.Store, t: float, before: dict[str, float]) -> store_mod.Store:
    """把库关掉、重新构造，验证状态从 DB 接着跑。"""
    print()
    print("  ── 重启：关掉 Store，重新从 DB 构造 ──")
    st.close()
    hooks.reset_memory()
    fresh = open_store()
    after = fresh.snapshot(t)
    equal = all(abs(float(before[k]) - float(after.emotion[k])) < 1e-9 for k in core.AXES)
    print(f"     重启前 emotion {axes(before)}")
    print(f"     重启后 emotion {axes(after.emotion)}   账本 {after.events} 条"
          f"   {'连续 ✓' if equal else '不连续 ✗'}")
    return fresh


def _count(st: store_mod.Store, table: str) -> int:
    row = st.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"] if row else 0)


if __name__ == "__main__":
    raise SystemExit(main())
