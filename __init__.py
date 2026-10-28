"""emotion_control —— 模型自报情绪 + @E 协议的情绪层（聊天网关友好）。

注册入口只做三件事：读配置、注册四个钩子、注册 CLI 子命令。所有逻辑都在别处，
所以这里的任何一行出错都能被 plugin loader 的 try/except 兜住而只损失这个插件。
"""

from __future__ import annotations

__all__ = ["register"]


def register(ctx) -> None:
    """插件加载时唯一被调用的地方。"""
    from . import cli, hooks

    hooks.configure(ctx)

    # pre_llm_call：注入"此刻"（子 agent 会被跳过）
    ctx.register_hook("pre_llm_call", hooks.pre_llm_call)
    # transform_llm_output：剥掉 @E 行并记账，返回正文
    ctx.register_hook("transform_llm_output", hooks.transform_llm_output)
    # 中间消息（工具调用前说的话）不走上面的钩子，发送层再兜一道
    hooks.install_send_relay()
    # post_tool_call：只记内存，给 event note 打标签
    ctx.register_hook("post_tool_call", hooks.post_tool_call)
    # on_session_end：情绪朝 temper 收一半
    ctx.register_hook("on_session_end", hooks.on_session_end)

    cli.register(ctx)
