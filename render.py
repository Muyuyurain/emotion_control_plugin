"""渲染层 —— 状态到注入文本的唯一出口。

为什么不给数字
--------------
模型看到 `心情 -0.12` 会把它当 setpoint：这一轮自报 +0.02，下一轮看着还是 -0.12 就再报
+0.02，形成闭环，情绪永远回不到中间。所以注入只给**词**，数字只出现在 `hermes emotion show`
这种给人看的界面上。

为什么边界两行永不裁剪
----------------------
`[边界]` 是这套东西的安全阀：情绪只改语气与详略，不改事实判断。预算不够时先丢可选行，
再丢关系行，边界一行都不许丢 —— 宁可少说一句"心情很好"，也不能让模型在心情差的时候
少给你一条命令或一条风险提示。
"""

from __future__ import annotations

from typing import Final, Iterable, Mapping

# 两行硬边界。永不裁剪、永不改写。
BOUNDARIES: Final[tuple[str, str]] = (
    "[边界] 情绪只改语气与详略，不改事实判断、安全边界与确认习惯。",
    "[边界] 可以话少、可以直接、可以说不看好；不可冷淡、说教、阴阳怪气；"
    "该给的命令与风险提示一条不少。",
)

DEFAULT_BUDGET: Final[int] = 400


def word_v(value: float) -> str:
    if value >= 0.45:
        return "很好"
    if value >= 0.12:
        return "偏好"
    if value > -0.12:
        return "平稳"
    if value > -0.45:
        return "低落"
    return "沉重"


def word_a(value: float) -> str:
    if value >= 0.70:
        return "高"
    if value >= 0.40:
        return "中"
    return "低"


def word_d(value: float) -> str:
    if value >= 0.30:
        return "稳"
    if value > -0.30:
        return "一般"
    return "偏低"


def word_trust(value: float) -> str:
    if value >= 0.70:
        return "信任"
    if value >= 0.50:
        return "较信任"
    if value >= 0.30:
        return "初步"
    return "生疏"


def state_line(emotion: Mapping[str, float]) -> str:
    """第一行：三轴各一个词。"""
    return (f"[此刻] 心情 {word_v(float(emotion.get('v', 0.0)))}"
            f"｜唤醒 {word_a(float(emotion.get('a', 0.0)))}"
            f"｜把握 {word_d(float(emotion.get('d', 0.0)))}")


def relation_line(trust: float, familiarity: float) -> str:
    """第二行：关系。熟悉度给一位小数（它不是情绪轴，不会被当 setpoint）。"""
    return f"[关系] 信任 {word_trust(trust)}｜熟悉 {max(0.0, min(1.0, familiarity)):.1f}"


def context(
    emotion: Mapping[str, float],
    *,
    trust: float = 0.35,
    familiarity: float = 0.0,
    extra: Iterable[str] = (),
    budget: int = DEFAULT_BUDGET,
) -> str:
    """拼出注入文本。

    丢弃顺序：可选行（extra，从前往后）→ 关系行。两行 [边界] 不参与裁剪。
    """
    head = state_line(emotion)
    relation = relation_line(trust, familiarity)
    optional = [line for line in extra if line]
    tail = list(BOUNDARIES)

    def size(parts: list[str]) -> int:
        return sum(len(part) + 1 for part in parts)

    while optional and size([head, relation] + optional + tail) > budget:
        optional.pop(0)
    middle: list[str] = [relation] + optional
    # 关系行都放不下时（预算被人为调到很低）：只留 head + 边界
    if size([head] + middle + tail) > budget:
        middle = []
    return "\n".join([head] + middle + tail)
