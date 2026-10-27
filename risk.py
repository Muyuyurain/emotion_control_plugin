"""高风险任务识别 —— 命中就跳过本轮情绪注入。

为什么要有这个
--------------
情绪层的边界（`[边界]` 那两行）是 prompt 级约束，靠模型自觉。真到删库、改防火墙、
提权这种活上，多一条"心情不太好"的提示就是多一份风险 —— 哪怕它写了"不改事实判断"。

所以策略很干脆：**这一轮看起来在干危险的事，情绪层直接闭嘴。** 不注入、不解释、不提示。

代价与取舍
----------
宁可误判。误判的代价只是这轮没有情绪提示（等于退回没有这个插件），
漏判的代价是模型在动防火墙的时候还在"心情沉重"。

因此匹配规则刻意宽松：中英文关键词 + 常见危险命令，命中任意一条即跳。
"""

from __future__ import annotations

import re

# 危险命令（英文）
_CMD = re.compile(
    r"|".join((
        r"\brm\s+-[a-z]*[rf]",              # rm -rf / -fr / -Rf
        r"\bmkfs(\.\w+)?\b",                # 格式化
        r"\bdd\s+if=",                      # dd 写盘
        r">\s*/dev/(sd|hd|nvme|vd)\w*",     # 重定向到块设备
        r"\bdrop\s+(table|database|schema)\b",
        r"\btruncate\s+table\b",
        r"\bdelete\s+from\b",
        r"\bupdate\s+\w+\s+set\b",
        r"\bchmod\s+(-[a-zA-Z]+\s+)?0?777\b",
        r"\bchown\s+-R\s+\S+\s+/",
        r"\b(shutdown|poweroff|halt|reboot)\b",
        r"\biptables\s+-F\b",
        r"\bnft\s+flush\b",
        r"\bfirewall-cmd\b[^\n]*--(panic|remove-(port|service|source|rich-rule))",
        r"\bsystemctl\s+(stop|disable|mask)\b",
        r"\bpasswd\s+root\b",
        r"\bvisudo\b",
        r"\bauthorized_keys\b",
        r"\bssh-keygen\s+-f\s+/root",
        r"\bdocker\s+(rm|rmi|prune)\b",
        r"\bgit\s+push\s+(-f|--force)\b",
        r"\bpodman\s+(rm|rmi)\b",
    )),
    re.IGNORECASE,
)

# 危险意图（中文，无词边界，直接子串匹配）
_ZH = (
    "删库", "删表", "删掉", "清空数据", "清空日志", "格式化", "重装系统",
    "关机", "重启服务", "重启服务器", "关闭防火墙", "停掉防火墙", "放开端口",
    "放行所有", "提权", "免密登录", "root密码", "root 密码", "私钥", "授权密钥",
    "覆盖原文件", "批量删除", "卸载", "直接改线上", "不要备份",
)


def match(text: str | None) -> str | None:
    """命中返回命中的那条规则（给日志用），没命中返回 None。"""
    body = (text or "").strip()
    if not body:
        return None

    found = _CMD.search(body)
    if found:
        return f"cmd:{found.group(0)[:40]}"

    for word in _ZH:
        if word in body:
            return f"zh:{word}"
    return None


def is_risky(text: str | None) -> bool:
    return match(text) is not None
