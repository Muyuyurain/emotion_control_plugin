"""SQLite 持久化 —— 所有状态变更的唯一写入口。

作用域
------
* **agent 级（全局一份）**：temper / mood / emotion。一个 bot 只有一个人格，
  不按用户分裂；否则"它今天心情不好"这种事得几十份各算一遍。
* **per-user**：只放 trust / familiarity，key 是 ``user:{platform}:{sender_id}``。
  同一个人的关系账本跨会话共享。

写入口
------
``apply_report()`` 是唯一入口：一次事务里写完 impulse / ledger / reports / users。
``set_temper()``（CLI 手动改底座）、``begin_turn()``（impulse 结转）、``halve()``
（会话收尾）是三个维护动作，都不改"某笔自报"的语义，因此不会出现两套记账。

数据目录走官方约定 ``plugin_storage.plugin_data_dir("emotion_control")``；拿不到 Hermes
运行时时退到 ``$HERMES_HOME/plugin-data/emotion_control/``，绝不写进插件安装目录
（那个目录会被 `hermes plugins update` 覆盖）。
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from . import core

logger = logging.getLogger(__name__)

# 进程内写串行。钩子每次调用都开一条新连接，所以"同一进程里两个群同时自报"
# 必须靠它挡住；跨进程（gateway + CLI 同时用）再靠 BEGIN IMMEDIATE 兜底。
# 没有它，`apply_report` 的"读 impulse → 算 → 写"会丢更新（10 并发只记 1 笔），
# `begin_turn` 与 `apply_report` 交错会让同一笔既进 ledger 又留在 impulse（2 倍放大）。
_WRITE_LOCK = threading.RLock()

PLUGIN_NAME = "emotion_control"
DB_FILENAME = "emotion.db"
SCHEMA_VERSION = 1
AGENT_ID = 1

# 幂等去重键在 meta 表里的前缀
_SEEN_PREFIX = "seen:"
_SEEN_KEEP = 512

_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS agent_state (
    id        INTEGER PRIMARY KEY CHECK(id=1),
    temper_v  REAL,
    temper_a  REAL,
    temper_d  REAL,
    imp_v     REAL DEFAULT 0,
    imp_a     REAL DEFAULT 0,
    imp_d     REAL DEFAULT 0,
    imp_ts    REAL DEFAULT 0,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    dv REAL,
    da REAL,
    dd REAL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS users (
    user_key    TEXT PRIMARY KEY,
    trust       REAL,
    familiarity REAL,
    turns       INTEGER DEFAULT 0,
    last_seen   REAL,
    last_tone   TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL,
    session_id TEXT,
    head       TEXT,
    dv         REAL,
    da         REAL,
    dd         REAL,
    verdict    TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_ts  ON events(ts);
CREATE INDEX IF NOT EXISTS idx_reports_ts ON reports(ts);
"""


# --------------------------------------------------------------------- 小工具


def db_path() -> Path:
    """数据库路径：环境变量覆盖 → 官方 plugin_data_dir → HERMES_HOME 兜底。"""
    override = os.environ.get("EMOTION_CONTROL_DB", "").strip()
    if override:
        return Path(override)
    try:
        from plugins.plugin_storage import plugin_data_dir   # 官方约定位置

        return plugin_data_dir(PLUGIN_NAME) / DB_FILENAME
    except Exception:
        home = os.environ.get("HERMES_HOME", "").strip() or str(Path.home() / ".hermes")
        return Path(home) / "plugin-data" / PLUGIN_NAME / DB_FILENAME


def user_key(sender_id: str = "", platform: str = "") -> str:
    """per-user 作用域键。"""
    ident = (sender_id or "").strip() or "anonymous"
    plat = (platform or "").strip() or "unknown"
    return f"user:{plat}:{ident}"


def parse_user_key(raw: str) -> str:
    """把 ``cli:u_42`` / ``u_42`` / ``user:cli:u_42`` 归一成作用域键。"""
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("user:"):
        return text
    if ":" in text:
        platform, _, ident = text.partition(":")
        return user_key(ident, platform)
    return user_key(text)


def dedupe_key(session_id: str, response_text: str, turn_id: str = "") -> str:
    """幂等键。优先按 turn_id（同一轮重发才该被挡），拿不到才退回内容哈希。

    turn_id 版是"按轮次"去重：模型连着两轮说一模一样的话（都回"好的"）也能各记一笔；
    内容哈希版会把第二轮误挡掉——那不是去重，是丢账。
    """
    if turn_id:
        return f"{session_id or '-'}:t:{turn_id}"
    digest = hashlib.sha1((response_text or "").encode("utf-8")).hexdigest()
    return f"{session_id or '-'}:{digest}"


def _retrying(fn: Callable[[], Any]) -> None:
    """SQLite 偶发忙（另一个连接正在初始化同一个库）时重试一次，仍失败就放过。

    DDL 是 IF NOT EXISTS、schema_version 是 UPSERT，跳过这一次下次打开会补上；
    但抛出去的话整个钩子就废了这一轮。
    """
    try:
        fn()
    except sqlite3.OperationalError:
        time.sleep(0.05)
        try:
            fn()
        except sqlite3.OperationalError as exc:
            logger.warning("emotion_control: 初始化语句被跳过（下次打开会补）：%s", exc)


def connect(path: Path | None = None) -> sqlite3.Connection:
    """打开连接：WAL + busy_timeout，DDL 幂等。

    `isolation_level=None`（真·自动提交）：事务由 `_tx()` 用显式 BEGIN IMMEDIATE
    控制。默认的隐式事务会把 SELECT 关在事务外、把 UPDATE 关在事务内，
    "读 → 算 → 写"正好横跨两次事务——并发丢更新的根因就在这。
    """
    target = Path(path) if path else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None
    _set_wal(conn)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _retrying(lambda: conn.executescript(_DDL))
    _retrying(lambda: conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    ))
    return conn


def _set_wal(conn: sqlite3.Connection) -> None:
    """已经是 wal 就不要再设：并发打开时 `PRAGMA journal_mode=WAL` 这句写操作
    会撞上别人的锁，抛 `attempt to write a readonly database`（实测 6 线程挂 5 个）。
    """
    def set_it() -> None:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        if row and str(row[0]).lower() == "wal":
            return
        conn.execute("PRAGMA journal_mode=WAL")

    _retrying(set_it)


# ------------------------------------------------------------------- 返回结构


@dataclass(slots=True)
class ApplyResult:
    """一次 ``apply_report`` 的结果，给钩子和 demo 读。"""

    status: str = "noop"            # written | skipped | zero | error
    delta: dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in core.AXES})
    trust: float = 0.35
    familiarity: float = 0.0
    verdict: str = ""
    events: int = 0

    @property
    def wrote(self) -> bool:
        return self.status == "written"


@dataclass(slots=True)
class Snapshot:
    """``load()`` 的返回值集合（也给 CLI 用）。"""

    temper: dict[str, float]
    mood: dict[str, float]
    emotion: dict[str, float]
    impulse: dict[str, float]
    imp_ts: float
    events: int


# ----------------------------------------------------------------- 存储本体


class Store:
    """一次回合用的连接包装。用 ``with`` 打开，退出即关。时钟外部注入。"""

    def __init__(self, conn: sqlite3.Connection, clock: Callable[[], float] | None = None,
                 settings: Any = None) -> None:
        self.conn = conn
        self._clock = clock or time.time
        self.settings = settings or _default_settings()
        self._ensure_agent_state(self.now())

    # ---- 生命周期 ----

    @classmethod
    def open(cls, path: Path | str | None = None, clock: Callable[[], float] | None = None,
             settings: Any = None) -> "Store":
        return cls(connect(Path(path) if path else None), clock=clock, settings=settings)

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ---- 事务 ----

    @contextmanager
    def _tx(self):
        """显式 immediate 事务：写锁从 BEGIN 就持有，跨进程也不交错。

        配合模块级 `_WRITE_LOCK`：进程内靠锁（快），进程间靠 SQLite（对）。
        """
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            try:
                self.conn.rollback()
            except Exception:
                pass
            raise
        else:
            self.conn.commit()

    def now(self) -> float:
        return float(self._clock())

    # ---- 初始化 ----

    def _ensure_agent_state(self, now: float) -> None:
        """首次写入时插 id=1 默认行（temper 建到默认值）。

        注意：不在这里 `commit()`。连接是 `isolation_level=None`（自动提交），
        单条语句本身就会落盘；而这方法可能在 `_tx()` 内部被 `_state_row()` 调起来，
        里面再 commit 会把外层事务提前提交掉。

        两条语句要能容忍并发建库时的瞬时 readonly（另一个连接正在切 WAL）：
        失败就跳过，下次打开会补 —— 这一轮宁可少一次状态自愈，也不能让钩子抛出。
        """
        def seed() -> None:
            self.conn.execute(
                "INSERT OR IGNORE INTO agent_state"
                "(id, temper_v, temper_a, temper_d, imp_v, imp_a, imp_d, imp_ts, updated_at)"
                " VALUES(?,?,?,?,0,0,0,0,?)",
                (AGENT_ID, core.DEFAULT_TEMPER["v"], core.DEFAULT_TEMPER["a"],
                 core.DEFAULT_TEMPER["d"], now),
            )
            # 旧库里 temper 可能因为手工删列变成 NULL：补成默认值，状态必须能自愈
            self.conn.execute(
                "UPDATE agent_state SET temper_v=COALESCE(temper_v,?),"
                " temper_a=COALESCE(temper_a,?), temper_d=COALESCE(temper_d,?) WHERE id=?",
                (core.DEFAULT_TEMPER["v"], core.DEFAULT_TEMPER["a"],
                 core.DEFAULT_TEMPER["d"], AGENT_ID),
            )

        _retrying(seed)

    # ---- 读 ----

    def _state_row(self) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM agent_state WHERE id=?", (AGENT_ID,)).fetchone()
        if row is None:                      # 极端情况：行被外部删了
            self._ensure_agent_state(self.now())
            row = self.conn.execute(
                "SELECT * FROM agent_state WHERE id=?", (AGENT_ID,)).fetchone()
        if row is None:
            # 建不起来（并发建库撞锁、库只读）：给一份静息态，让这一轮按默认值跑完，
            # 而不是把整个钩子抛掉。缺的那一行下次打开会补。
            return {
                "temper_v": core.DEFAULT_TEMPER["v"], "temper_a": core.DEFAULT_TEMPER["a"],
                "temper_d": core.DEFAULT_TEMPER["d"], "imp_v": 0.0, "imp_a": 0.0,
                "imp_d": 0.0, "imp_ts": 0.0, "updated_at": 0.0,
            }
        return row

    def _events(self) -> list[core.Event]:
        rows = self.conn.execute("SELECT ts, dv, da, dd, note FROM events ORDER BY id")
        return [core.Event(ts=float(r["ts"] or 0.0), dv=float(r["dv"] or 0.0),
                           da=float(r["da"] or 0.0), dd=float(r["dd"] or 0.0),
                           note=str(r["note"] or "")) for r in rows]

    def _temper(self) -> dict[str, float]:
        row = self._state_row()
        return {
            "v": float(row["temper_v"]),
            "a": float(row["temper_a"]),
            "d": float(row["temper_d"]),
        }

    def _impulse(self) -> tuple[dict[str, float], float]:
        row = self._state_row()
        return (
            {"v": float(row["imp_v"] or 0.0), "a": float(row["imp_a"] or 0.0),
             "d": float(row["imp_d"] or 0.0)},
            float(row["imp_ts"] or 0.0),
        )

    def snapshot(self, now: float | None = None) -> Snapshot:
        """当前三层。不写库，纯读。"""
        ts = self.now() if now is None else float(now)
        temper = self._temper()
        events = self._events()
        impulse, imp_ts = self._impulse()
        mood = core.mood_from(temper, events, ts, self.settings.tau_event)
        emotion = core.emotion_from(mood, impulse, imp_ts, ts, self.settings.tau_impulse)
        return Snapshot(temper=temper, mood=mood, emotion=emotion, impulse=impulse,
                        imp_ts=imp_ts, events=len(events))

    def load(self, now: float | None = None) -> tuple[dict[str, float], dict[str, float],
                                                      dict[str, float]]:
        """返回 ``(mood, emotion, temper)``。顺带剪枝（维护动作，不改语义）。"""
        ts = self.now() if now is None else float(now)
        snap = self.snapshot(ts)
        self._prune(ts)
        return snap.mood, snap.emotion, snap.temper

    def emotion_object(self, now: float | None = None) -> core.Emotion:
        """把库里的状态装成一个 ``core.Emotion``（demo / explain 用）。"""
        ts = self.now() if now is None else float(now)
        temper = self._temper()
        impulse, imp_ts = self._impulse()
        return core.Emotion(
            temper=temper,
            ledger=self._events(),
            impulse=impulse,
            imp_ts=imp_ts,
            tau_event=self.settings.tau_event,
            tau_impulse=self.settings.tau_impulse,
            clock=lambda: ts,
        )

    # ---- 用户 ----

    def user(self, key: str) -> dict[str, Any]:
        """读用户行；没有就按基线建一行（不改动已有值）。"""
        if not key:
            return {"user_key": "", "trust": self.settings.trust_base, "familiarity": 0.0,
                    "turns": 0, "last_seen": 0.0, "last_tone": ""}
        row = self.conn.execute("SELECT * FROM users WHERE user_key=?", (key,)).fetchone()
        if row is None:
            # 不 commit：可能正跑在 apply_report 的事务里（自动提交模式下单条语句已落盘）
            self.conn.execute(
                "INSERT OR IGNORE INTO users(user_key, trust, familiarity, turns, last_seen,"
                " last_tone) VALUES(?,?,?,0,?, '')",
                (key, self.settings.trust_base, 0.0, self.now()),
            )
            row = self.conn.execute("SELECT * FROM users WHERE user_key=?", (key,)).fetchone()
        return {
            "user_key": str(row["user_key"]),
            "trust": float(row["trust"] if row["trust"] is not None
                           else self.settings.trust_base),
            "familiarity": float(row["familiarity"] or 0.0),
            "turns": int(row["turns"] or 0),
            "last_seen": float(row["last_seen"] or 0.0),
            "last_tone": str(row["last_tone"] or ""),
        }

    # ---------------------------------------------------------- 唯一写入口

    def apply_report(
        self,
        delta: Mapping[str, float] | None,
        *,
        now: float,
        session_id: str = "",
        note: str = "",
        user_text: str = "",
        verdict: str = "ok",
        head: str = "",
        ukey: str = "",
        key: str = "",
    ) -> ApplyResult:
        """记一笔自报。一个事务里写完 impulse / reports / users。

        注意 ledger：**这一笔不进 ledger**，它先待在 `impulse` 里，下一轮
        ``begin_turn()`` 再整体搬进 `events`。这不是遗漏，是 core 里那条不变量
        （同一笔事件要么在 impulse、要么在 ledger，绝不两边同时存在）的要求；
        两边都写就会回到 demo 的 2.0 倍放大。

        参数说明（比规格多出的几个都是可选的）：
            ``user_text``  已 scrub 过的用户消息，仅用于"自报缺失时的语气兜底"
            ``verdict``    protocol 的解析结论，落进 reports 供审计
            ``head``       原始首行（截断），tail 里能看到模型到底写了什么
            ``ukey``       用户作用域键
            ``key``        幂等去重键（(session_id, sha1(全文))）
        """
        ts = float(now)
        clean = core.clean_delta(delta)
        result = ApplyResult(verdict=verdict)

        try:
            # 锁 + immediate 事务包住整段"读 → 算 → 写"。
            # 幂等检查也必须在锁内：两个并发带着同一个 key 进来，锁外检查会双双通过。
            with _WRITE_LOCK, self._tx():
                if key and self._seen(key, ts):
                    result.status = "skipped"
                    return result

                user = self.user(ukey) if ukey else {
                    "trust": self.settings.trust_base, "familiarity": 0.0, "turns": 0}
                trust = float(user["trust"])
                familiarity = float(user["familiarity"])
                turns = int(user["turns"])

                if not core.is_zero(clean):
                    impulse, imp_ts = self._impulse()
                    impulse, imp_ts = core.accumulate_impulse(
                        impulse, imp_ts, clean, ts, self.settings.tau_impulse)
                    self.conn.execute(
                        "UPDATE agent_state SET imp_v=?, imp_a=?, imp_d=?, imp_ts=?, updated_at=?"
                        " WHERE id=?",
                        (impulse["v"], impulse["a"], impulse["d"], imp_ts, ts, AGENT_ID),
                    )
                    # 事件此刻只存在于 impulse；它会在下一轮 begin_turn 时搬进 ledger。
                    result.status = "written"

                    # trust 由自报推动（小幅）
                    step = max(-self.settings.trust_step,
                               min(self.settings.trust_step, 0.3 * clean["v"]))
                    trust = _clampf(trust + step, self.settings.trust_lo, self.settings.trust_hi)
                else:
                    # 本轮没报到（没标签 / 全 0 / 写坏）：用极小的语气词表兜底
                    step = core.appraise(user_text)
                    if step:
                        trust = _clampf(trust + step, self.settings.trust_lo,
                                        self.settings.trust_hi)
                    result.status = "zero"

                familiarity = _clampf(familiarity + self.settings.familiarity_step, 0.0, 1.0)
                turns += 1

                if ukey:
                    self.conn.execute(
                        "UPDATE users SET trust=?, familiarity=?, turns=?, last_seen=?,"
                        " last_tone=? WHERE user_key=?",
                        (trust, familiarity, turns, ts, _verdict_tone(verdict), ukey),
                    )

                self.conn.execute(
                    "INSERT INTO reports(ts, session_id, head, dv, da, dd, verdict)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (ts, session_id or "", (head or "")[:120], clean["v"], clean["a"],
                     clean["d"], verdict),
                )
                if note:
                    # 留给下一轮 begin_turn 当 event 的标签：重启后内存里没有上一轮的
                    # 上下文，这一个键让 explain 仍然能说清"这笔是跟哪句话走的"。
                    self.conn.execute(
                        "INSERT INTO meta(key, value) VALUES('pending_note', ?)"
                        " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (note[:60],))
                if key:
                    self._mark_seen(key, ts)
                # 提交由 _tx() 的出口完成
        except sqlite3.Error as exc:            # 落库失败不该把回复吞掉
            logger.warning("emotion_control: apply_report 写入失败：%s", exc)
            result.status = "error"
            return result

        self._prune(ts)
        result.delta = clean
        result.trust = trust
        result.familiarity = familiarity
        result.events = self._count_events()
        return result

    # ---------------------------------------------------------------- 维护动作

    def begin_turn(self, now: float | None = None, note: str = "") -> dict[str, float]:
        """新一轮开始：把上一轮尚未结转的 impulse 搬进 ledger，然后清零。

        这是不变量（同一笔只在 impulse 或 ledger 之一）的执行点。搬运用"已衰减值 +
        ts=now"，保证切换瞬间的读数连续。返回被结转的量（三轴）。

        ``note`` 是这笔事件的人类可读标签（上一轮用户说了什么、用了哪些工具），
        只用于 ``hermes emotion explain`` 的可追溯性，不参与算术。
        """
        ts = self.now() if now is None else float(now)
        with _WRITE_LOCK, self._tx():
            impulse, imp_ts = self._impulse()
            if not imp_ts:
                return {axis: 0.0 for axis in core.AXES}
            live = core.impulse_view(impulse, imp_ts, ts, self.settings.tau_impulse)
            self.conn.execute(
                "UPDATE agent_state SET imp_v=0, imp_a=0, imp_d=0, imp_ts=0, updated_at=?"
                " WHERE id=?", (ts, AGENT_ID))
            if not core.is_zero(live):
                label = (note or self._pending_note() or "rollover")[:60]
                self.conn.execute(
                    "INSERT INTO events(ts, dv, da, dd, note) VALUES(?,?,?,?,?)",
                    (ts, live["v"], live["a"], live["d"], label))
        self._prune(ts)
        return live

    def halve(self, now: float | None = None) -> int:
        """把情绪朝 temper 收一半。**裸操作**，不做频率限制（CLI/测试用）。"""
        ts = self.now() if now is None else float(now)
        with _WRITE_LOCK, self._tx():
            self._halve_rows(ts)
        self._prune(ts)
        return self._count_events()

    def _halve_rows(self, ts: float) -> None:
        """实际的折半语句。调用方必须已持锁并开了事务。"""
        self.conn.execute("UPDATE events SET dv=dv*0.5, da=da*0.5, dd=dd*0.5")
        self.conn.execute(
            "UPDATE agent_state SET imp_v=imp_v*0.5, imp_a=imp_a*0.5, imp_d=imp_d*0.5,"
            " updated_at=? WHERE id=?", (ts, AGENT_ID))

    def halve_if_due(self, now: float | None = None, *, min_gap: float = 3 * 3600.0) -> bool:
        """按时间闸门的「睡一觉」。返回是否真的收了。

        为什么需要闸门：Hermes 的 ``on_session_end`` **不是会话结束，而是每轮结束**
        （``agent/turn_finalizer.py`` 在 ``finalize_turn()`` 里触发，而 ``finalize_turn``
        由 ``conversation_loop`` 每条消息调一次）。裸调 ``halve()`` 等于每说一句话就把
        整个账本砍半 —— 实测 12 轮连续 −0.10 会一直停在 +0.002，地板 −0.55 永远碰不到。

        闸门本身必须是**原子的**："读 last_halve_at → 判 → 折半 → 写回" 要在同一个
        写锁与事务里完成。分开做的话，两个群同时收尾就会双双通过闸门、折半两次
        （0.25 倍）—— 和并发记账丢更新是同一类错误。

        默认 min_gap 取 3 小时而不是 8 小时：τ_event = 2h，隔 8 小时账本已衰减到
        e^(-4) ≈ 1.8%，这时候再折半等于什么都没做。3 小时留 22%，折半后 11%，
        "午休回来情绪淡了一半"才是这个机制唯一有意义的口径。
        """
        ts = self.now() if now is None else float(now)
        with _WRITE_LOCK, self._tx():
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='last_halve_at'").fetchone()
            try:
                last = float(row["value"]) if row is not None else 0.0
            except (TypeError, ValueError):
                last = 0.0
            if ts - last < min_gap:
                return False
            self._halve_rows(ts)
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('last_halve_at', ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(ts),))
        self._prune(ts)
        return True

    def set_temper(self, axis: str, value: float, now: float | None = None) -> float:
        """手动改 temper（CLI）。越界会被压到不对称边界上。"""
        if axis not in core.AXES:
            raise KeyError(f"未知的轴：{axis}（只能是 v/a/d）")
        ts = self.now() if now is None else float(now)
        clamped = core.clamp_state(axis, float(value))
        self._ensure_agent_state(ts)
        self.conn.execute(
            f"UPDATE agent_state SET temper_{axis}=?, updated_at=? WHERE id=?",
            (clamped, ts, AGENT_ID),
        )
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('temper_set_at', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(ts),))
        self.conn.commit()
        return clamped

    # ---------------------------------------------------------------- 审计

    def explain(self, now: float | None = None, limit: int = 20) -> list[dict[str, Any]]:
        ts = self.now() if now is None else float(now)
        return self.emotion_object(ts).explain(ts)[:limit]

    def reports(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM reports ORDER BY id DESC LIMIT ?", (max(1, int(limit)),)))

    def _count_events(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
        return int(row["n"] if row else 0)

    # ---------------------------------------------------------------- 剪枝

    def _prune(self, now: float) -> None:
        """权重 < prune_weight 的删掉；再按 ts 保住最近 max_events 条。"""
        weight = max(1e-6, min(0.999999, float(self.settings.prune_weight)))
        # 权重 e^(-age/tau) == weight  →  age = tau * ln(1/weight)（0.01 → 约 9.2 小时）
        max_age = self.settings.tau_event * _log(1.0 / weight)
        threshold_ts = now - max_age
        try:
            self.conn.execute("DELETE FROM events WHERE ts < ?", (threshold_ts,))
            over = self._count_events() - self.settings.max_events
            if over > 0:
                self.conn.execute(
                    "DELETE FROM events WHERE id IN"
                    " (SELECT id FROM events ORDER BY ts ASC LIMIT ?)", (over,))
            # 去重键不能无限长：只留最近 _SEEN_KEEP 条
            seen = self.conn.execute(
                "SELECT COUNT(*) AS n FROM meta WHERE key LIKE ?", (f"{_SEEN_PREFIX}%",)
            ).fetchone()
            extra = int(seen["n"] if seen else 0) - _SEEN_KEEP
            if extra > 0:
                self.conn.execute(
                    "DELETE FROM meta WHERE key IN (SELECT key FROM meta WHERE key LIKE ?"
                    " ORDER BY CAST(value AS REAL) ASC LIMIT ?)", (f"{_SEEN_PREFIX}%", extra))
            # 自报审计表每轮一行、永久累积；留最近 5000 行足够回溯，再多就是占地方
            self.conn.execute(
                "DELETE FROM reports WHERE id <= (SELECT MAX(id) - 5000 FROM reports)")
        except sqlite3.Error as exc:
            logger.warning("emotion_control: 剪枝失败（不影响读取）：%s", exc)

    # ---- 幂等记录 ----

    def _pending_note(self) -> str:
        """上一笔自报留下的标签（重启后仍然可用）。"""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='pending_note'").fetchone()
        return str(row["value"] or "") if row else ""

    def _seen(self, key: str, now: float) -> bool:
        ttl = float(self.settings.dedupe_ttl_seconds)
        if ttl <= 0:
            return False
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (f"{_SEEN_PREFIX}{key}",)).fetchone()
        if row is None:
            return False
        try:
            last = float(row["value"] or 0.0)
        except (TypeError, ValueError):
            return False
        if ttl and now - last > ttl:
            return False
        return True

    def _mark_seen(self, key: str, now: float) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (f"{_SEEN_PREFIX}{key}", str(now)))


# ----------------------------------------------------------------- 内部小函数


def _verdict_tone(verdict: str) -> str:
    """把 verdict 折成一个短标签存进 users.last_tone（CLI 里看趋势）。"""
    return {
        "ok": "self-report",
        "zero": "zero",
        "tag_unparsable": "broken",
        "empty_body": "empty-body",
        "no_tag": "no-tag",
        "empty": "empty",
    }.get(verdict or "", verdict or "")


def _clampf(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _log(value: float) -> float:
    return math.log(value) if value > 0 else 0.0


def _default_settings():
    from . import config

    return config.Settings()
