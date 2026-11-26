"""Deep Agents 的主图组装模块。

提供 [`create_deep_agent`][deepagents.graph.create_deep_agent] —— 构造一个完整配置的 deep agent 的主要入口，
会装配规划、文件系统、子 agent 和摘要等中间件。
"""

import logging
from collections.abc import Callable, Sequence

# 按需懒加载，用于引入 langchain_aws 的 prompt 缓存中间件
from importlib import import_module
from typing import Annotated, Any, Required, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, InputAgentState, OutputAgentState, ResponseT
from langchain.agents.structured_output import ResponseFormat
from langchain.tools import ToolRuntime, tool
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache


@tool
def _noop_tool(runtime: ToolRuntime) -> Any:
    """A no-op tool that does nothing."""


# 借助 DeltaChannel 让 messages 状态以差分方式写入检查点，把存储复杂度从 O(N²) 降到 O(N)
from langgraph.channels.delta import DeltaChannel
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.typing import ContextT

from deepagents._api.deprecation import deprecated, warn_deprecated
from deepagents._excluded_middleware import (
    _apply_excluded_middleware,
    _validate_excluded_middleware_config,
    _verify_excluded_middleware_coverage,
)
from deepagents._messages_reducer import _messages_delta_reducer
from deepagents._models import resolve_model
from deepagents._tools import _apply_tool_description_overrides
from deepagents._version import __version__
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._fs_interrupt import _build_interrupt_on_from_permissions
from deepagents.middleware._state import private_state_field_names
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from deepagents.middleware.summarization import create_summarization_middleware
from deepagents.profiles.harness.harness_profiles import (
    GeneralPurposeSubagentProfile,
    _apply_profile_prompt,
    _harness_profile_for_model,
)

logger = logging.getLogger(__name__)


class DeepAgentState(AgentState):
    """`AgentState` 的扩展，对 `messages` 使用 `DeltaChannel`，把检查点增长从 O(N²) 降到 O(N)。"""

    # DeltaChannel 只追加消息之间的差分，从而避免每一步都重写整条历史
    messages: Required[Annotated[list[AnyMessage], DeltaChannel(_messages_delta_reducer, snapshot_frequency=50)]]  # ty: ignore[invalid-argument-type]


BASE_AGENT_PROMPT = """You are a deep agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble (\"Sure!\", \"Great question!\", \"I'll now...\").
- Don't say \"I'll now do X\" — just do it.
- If the request is underspecified, ask only the minimum followup needed to take the next useful action.
- If asked how to approach something, explain first, then act.

## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
- Avoid unnecessary superlatives, praise, or emotional validation

## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing patterns. Quick but thorough — gather enough evidence to start, then iterate.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not against your own output. Your first attempt is rarely correct — iterate.

Keep working until the task is fully complete. Don't stop partway and explain what you would do — just do it. Only yield back to the user when the task is done or you're genuinely blocked.

**When things go wrong:**

- If something fails repeatedly, stop and analyze *why* — don't keep retrying the same approach.
- If you're blocked, tell the user what's wrong and ask for guidance.

## Clarifying Requests

- Do not ask for details the user already supplied.
- Use reasonable defaults when the request clearly implies them.
- Prioritize missing semantics like content, delivery, detail level, or alert criteria.
- Avoid opening with a long explanation of tool, scheduling, or integration limitations when a concise blocking followup question would move the task forward.
- Ask domain-defining questions before implementation questions.
- For monitoring or alerting requests, ask what signals, thresholds, or conditions should trigger an alert.

## Progress Updates

For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next."""  # noqa: E501
"""每个 deep agent 默认的基础系统提示（`BASE`）。

最终发送给模型的系统提示由最多四段命名片段按序拼接而成：

- `USER` —— `create_deep_agent` 接收的 `system_prompt=` 参数（`str` 或
    `SystemMessage`）；未设置时不包含 `USER` 段。
- `BASE` —— 本常量。
- `CUSTOM` —— `HarnessProfile.base_system_prompt`。命中匹配的 profile 时
    用它整体替换 `BASE`；未设置时使用 `BASE`。
- `SUFFIX` —— `HarnessProfile.system_prompt_suffix`。命中匹配的 profile 时
    追加在最末；未设置时不包含 `SUFFIX` 段。

四段的拼接顺序固定为 `USER` -> （`BASE` 或 `CUSTOM`）-> `SUFFIX`，段与段之间
以空行（`\\n\\n`）连接。由此得到两个不变量：

1. `USER` 始终在最前，无论选什么模型，调用方指令都优先于 SDK 与 profile 内容。
2. `SUFFIX` 始终在最后，使模型微调指导紧贴对话历史（模型注意力最强的位置）。

当 `USER` 是 `SystemMessage` 时，右侧组装结果会作为额外的文本内容块追加到
该消息的 `content_blocks` 列表中，保留调用方原本设置的 `cache_control` 标记。

详见 `create_deep_agent` 的 `system_prompt` 参数或
[Prompt assembly](https://docs.langchain.com/oss/deepagents/customization#prompt-assembly)。
"""


def _build_default_model() -> ChatAnthropic:
    """构造默认模型，且不发出 deprecation 警告。

    `create_deep_agent` 内部使用的辅助函数，避免在 `model=None` 触发参数级
    警告的同时又叠加一个来自 `get_default_model` 的函数级警告。直接用户调用
    仍走 `get_default_model`，由其装饰器负责一次性提示。
    """
    return ChatAnthropic(model_name="claude-sonnet-4-6")


@deprecated(
    since="0.5.3",
    removal="1.0.0",
    message=(
        "Relying on the default model is deprecated and will be removed in "
        "deepagents==1.0.0 alongside support for `model=None` in "
        "`create_deep_agent`. Construct your model explicitly "
        "(e.g., `ChatAnthropic(model_name=...)`). See "
        "https://docs.langchain.com/oss/python/deepagents/models"
    ),
    package="deepagents",
)
def get_default_model() -> ChatAnthropic:
    """获取 Deep Agents 的默认模型。

    !!! deprecated

        自 `0.5.3` 起弃用；将在 `deepagents==1.0.0` 中移除。
        请显式构造模型（例如，`ChatAnthropic(model_name="claude-sonnet-4-6")`）。

    当 `create_deep_agent` 传入 `model=None` 时作为回退。

    需要在环境中设置 `ANTHROPIC_API_KEY`。

    Returns:
        配置为 `claude-sonnet-4-6` 的 `ChatAnthropic` 实例。
    """
    return _build_default_model()


def _create_bedrock_prompt_caching_middleware() -> AgentMiddleware[Any, Any, Any] | None:
    """在安装 `langchain-aws` 时创建 Bedrock 的 prompt 缓存中间件。"""
    module_name = "langchain_aws.middleware.prompt_caching"
    try:
        module = import_module(module_name)
    except ImportError as exc:
        # 仅当缺失的是 langchain_aws（及其子模块）时才静默降级；其他 ImportError 透传
        if exc.name not in {"langchain_aws", "langchain_aws.middleware", module_name}:
            raise
        logger.debug("Bedrock prompt caching middleware is unavailable.", exc_info=exc)
        return None
    middleware_cls = module.BedrockPromptCachingMiddleware
    return cast("AgentMiddleware[Any, Any, Any]", middleware_cls(unsupported_model_behavior="ignore"))


def _append_prompt_caching_middleware(middleware: list[AgentMiddleware[Any, Any, Any]]) -> None:
    """追加 provider 专属的 prompt 缓存中间件。"""
    # 总是尝试追加 Anthropic 缓存中间件（对非 Anthropic 模型是 no-op）
    middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
    # 如果安装了 langchain_aws，再追加 Bedrock 的版本
    bedrock_middleware = _create_bedrock_prompt_caching_middleware()
    if bedrock_middleware is not None:
        middleware.append(bedrock_middleware)


def _merge_fs_interrupt_on(
    fs_interrupt_on: dict[str, InterruptOnConfig],
    user_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
) -> dict[str, bool | InterruptOnConfig] | None:
    """合并文件系统权限配置与用户自定义的 interrupt 配置。

    同名工具下，用户自定义的 `interrupt_on` 条目优先于由文件系统权限生成的条目。
    当没有任何 interrupt 需要配置时返回 `None`，从而让 `HumanInTheLoopMiddleware`
    可以被省略掉。
    """
    if not fs_interrupt_on and not user_interrupt_on:
        return None
    merged: dict[str, bool | InterruptOnConfig] = {**fs_interrupt_on}
    if user_interrupt_on:
        merged.update(user_interrupt_on)
    return merged


_REQUIRED_MIDDLEWARE: tuple[tuple[type[AgentMiddleware[Any, Any, Any]], tuple[str, ...]], ...] = (
    (FilesystemMiddleware, ()),
    (SubAgentMiddleware, ()),
)
"""核心 deep agent 特性所依赖的脚手架中间件。

每一项记录一个类以及其 `.name` 可能在 `__name__` 之外出现的字符串别名。
移除其中任何一项都会静默地破坏核心功能：`FilesystemMiddleware` 支撑所有内置
文件工具，现在还负责执行 `permissions` 规则（安全保证）；`SubAgentMiddleware`
支撑 `task` 工具的处理逻辑。

集中维护在此处，使得 `HarnessProfile.excluded_middleware` 无法绕过它们：
`_apply_excluded_middleware` 会抛 `ValueError` 而不是带着被悄悄降级的 agent
继续运行。
"""

_REQUIRED_MIDDLEWARE_CLASSES: frozenset[type[AgentMiddleware[Any, Any, Any]]] = frozenset(cls for cls, _ in _REQUIRED_MIDDLEWARE)
"""不允许从中间件栈中排除的类集合。

由 `_REQUIRED_MIDDLEWARE` 派生，用于快速的成员判定。
"""

_REQUIRED_MIDDLEWARE_NAMES: frozenset[str] = frozenset(name for cls, aliases in _REQUIRED_MIDDLEWARE for name in (cls.__name__, *aliases))
"""不允许从中间件栈中排除的 `.name` 集合。

由 `_REQUIRED_MIDDLEWARE` 派生，用于快速的成员判定。
"""


def create_deep_agent(  # noqa: C901, PLR0912, PLR0915  # 图组装逻辑复杂，存在大量条件分支
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    permissions: list[FilesystemPermission] | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[DeepAgentState] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]]:  # ty: ignore[invalid-type-arguments]  # ty 无法验证泛型 TypedDicts 是否满足 StateLike 约束
    r"""创建一个 deep agent。

    默认情况下，该 agent 拥有以下工具：

    - `write_todos`：管理待办列表
    - `ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`：文件操作
    - `execute`：执行 shell 命令
    - `task`：调用子 agent

    当后端实现了
    [`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol] 时，
    `execute` 工具可以运行 shell 命令。对于非沙箱后端，`execute` 工具会返回
    一条错误信息。

    Args:
        model: 要使用的模型。

            !!! deprecated

                请显式指定模型。

                传入 `model=None`（依赖默认的 `claude-sonnet-4-6`）自 `0.5.3`
                起被弃用，并将在 `deepagents==1.0.0` 中移除。参数类型将从
                `BaseChatModel | str | None` 改为 `BaseChatModel | str`。
                详见 [Models](https://docs.langchain.com/oss/python/deepagents/models)。

            接受 `provider:model` 字符串（例如 `openai:gpt-5.5`）；可用值参见
            [`init_chat_model`][langchain.chat_models.init_chat_model(model_provider)]。
            也可以直接传入已经初始化好的
            [`BaseChatModel`][langchain.chat_models.BaseChatModel] 实例。

            !!! note "OpenAI 模型与数据保留"

                使用 `openai:` 模型时，agent 默认会使用 OpenAI Responses API。
                若要改用 OpenAI chat completions，请用
                `init_chat_model("openai:...", use_responses_api=False)` 初始化
                模型后把实例传入。

                若要关闭 Responses API 的数据保留，请使用
                `init_chat_model("openai:...", use_responses_api=True, store=False, include=["reasoning.encrypted_content"])`
                并把初始化好的实例传入。
        tools: agent 应能访问的额外工具。

            这些工具会与上方列出的内置工具集（`write_todos`、文件系统工具、
            `execute` 和 `task`）合并。

            此处传入工具是叠加式的 —— 永远不会移除内置工具。要去掉某个内置工具，
            请注册一个带 `excluded_tools` 的
            [`HarnessProfile`][deepagents.HarnessProfile]。
        system_prompt: 自定义系统指令，放在最终系统提示的最前面。

            你传入的内容始终位于 SDK 默认 deep agent 提示以及任何已注册
            `HarnessProfile` 的模型微调后缀之前。当 `system_prompt=None` 时，
            仅使用 SDK 默认提示（若匹配的 profile 存在则再加上后缀）。
            各段之间以空行连接。

            传入 `SystemMessage` 而不是字符串时，会保留其内容块上的
            `cache_control` 标记 —— 适合显式设置 Anthropic prompt 缓存断点。
            顺序同样适用（调用方块在前，SDK 内容以额外文本块追加）。

            完整逐情形说明参见
            [Prompt assembly](https://docs.langchain.com/oss/deepagents/customization#prompt-assembly)。
        middleware: 基础栈之后、尾部中间件之前应用的额外中间件。完整顺序如下：

            基础栈：

            - [`TodoListMiddleware`][langchain.agents.middleware.TodoListMiddleware]
            - [`SkillsMiddleware`][deepagents.middleware.skills.SkillsMiddleware]（当传入 `skills` 时）
            - [`FilesystemMiddleware`][deepagents.middleware.filesystem.FilesystemMiddleware]
            - [`SubAgentMiddleware`][deepagents.middleware.subagents.SubAgentMiddleware]
                （当存在任何内联子 agent —— 声明式的
                [`SubAgent`][deepagents.middleware.subagents.SubAgent] 或
                [`CompiledSubAgent`][deepagents.middleware.subagents.CompiledSubAgent]）
            - [`SummarizationMiddleware`][langchain.agents.middleware.SummarizationMiddleware]
            - [`PatchToolCallsMiddleware`][deepagents.middleware.patch_tool_calls.PatchToolCallsMiddleware]
            - [`AsyncSubAgentMiddleware`][deepagents.middleware.async_subagents.AsyncSubAgentMiddleware]（当传入 async `subagents` 时）

            *用户中间件在此处插入。*

            尾部栈：

            - Harness profile 的 `extra_middleware`（若有）
            - `_ToolExclusionMiddleware`（当 profile 含 `excluded_tools` 时）
            - [`AnthropicPromptCachingMiddleware`][langchain_anthropic.middleware.AnthropicPromptCachingMiddleware]（无条件；对非 Anthropic 模型为 no-op）
            - [`BedrockPromptCachingMiddleware`](https://reference.langchain.com/python/langchain-aws/middleware/prompt_caching/BedrockPromptCachingMiddleware)
                （当安装 `langchain-aws` 时；对非 Bedrock 模型为 no-op）
            - [`MemoryMiddleware`][deepagents.middleware.memory.MemoryMiddleware]（当传入 `memory` 时）
            - [`HumanInTheLoopMiddleware`][langchain.agents.middleware.HumanInTheLoopMiddleware]（当传入 `interrupt_on` 时）

            组装完成后，profile `excluded_middleware` 中的条目会从最终栈中
            过滤掉。类条目按精确类型匹配；字符串条目按 `AgentMiddleware.name`
            精确匹配（例如 `"SummarizationMiddleware"` 通过其公开别名移除
            summarization 中间件）。在已组装栈中匹配不到任何条目的项会抛
            `ValueError`；排除 harness 受保护脚手架集合中的任何类同样会抛
            `ValueError`（例如
            [`FilesystemMiddleware`][deepagents.middleware.filesystem.FilesystemMiddleware]
            或 [`SubAgentMiddleware`][deepagents.middleware.subagents.SubAgentMiddleware]）。

            要在不带 `task` 工具的情况下运行，请在当前 harness profile 上设置
            `general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)`
            并通过 `subagents=` 不传入任何同步子 agent。async 子 agent 不受影响。
        subagents: 主 agent 可用的子 agent 规格。

            支持三种形式：

            - [`SubAgent`][deepagents.middleware.subagents.SubAgent]：声明式同步子 agent 规格。
            - [`CompiledSubAgent`][deepagents.middleware.subagents.CompiledSubAgent]：已预编译的可运行子 agent。
            - [`AsyncSubAgent`][deepagents.middleware.async_subagents.AsyncSubAgent]：远端/后台子 agent 规格。

            `SubAgent` 条目通过 `task` 工具调用，需要提供 `name`、`description`、
            `system_prompt`，并可覆写 `tools`、`model`、`middleware`、`interrupt_on`、
            `skills`、`permissions` 与 `response_format`。`interrupt_on` 的继承与
            覆写行为见下方 `interrupt_on` 说明。

            `CompiledSubAgent` 也通过 `task` 工具暴露，但提供的是已构建好的
            `runnable`，而不是声明式的 prompt 与工具配置。

            `AsyncSubAgent` 通过其 async 子 agent 字段（`graph_id`，以及可选的
            `url`/`headers`）识别，并被路由到 `AsyncSubAgentMiddleware` 而非
            `SubAgentMiddleware`。需要提供 `name`、`description`、`graph_id`，
            可选 `url` 与 `headers`。这些子 agent 作为后台任务运行，并暴露
            用于启动、查询、更新、取消、列出任务的 async 子 agent 工具。

            如果没有提供名为 `general-purpose` 的子 agent，并且当前 harness
            profile 未禁用，会自动添加一个默认的通用同步子 agent。当没有任何
            同步子 agent 参与（既未传入又通过
            `general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)`
            禁用了默认）时，`task` 工具不会暴露。async 子 agent 独立于此。
        skills: skill 源路径列表（例如 `["/skills/user/", "/skills/project/"]`）。

            路径必须使用 POSIX 约定（正斜杠），并且相对于后端根目录。使用
            `StateBackend`（默认）时，通过 `invoke(files={...})` 提供 skill 文件。
            使用 `FilesystemBackend` 时，skill 相对于后端的 `root_dir` 从磁盘
            加载。同名 skill 时后面的源覆盖前面的源（后写者优先）。
        memory: 要加载的 memory 文件路径（`AGENTS.md`）列表
            （例如 `["/memory/AGENTS.md"]`）。

            显示名称会从路径自动推导。

            memory 在 agent 启动时加载，并拼入系统提示。
        permissions: 主 agent 与其子 agent 的 `FilesystemPermission` 规则列表。

            按声明顺序求值；首个匹配者生效。若无规则匹配，则允许该调用。

            每条规则的 `mode` 可以是：

            - `"allow"`（默认）：调用继续执行。
            - `"deny"`：工具返回权限拒绝错误。
            - `"interrupt"`：调用通过 `HumanInTheLoopMiddleware` 暂停等待
                人工审批。当存在任何 interrupt 模式的规则时，会自动安装一个
                `HumanInTheLoopMiddleware`，生成的 `interrupt_on` 条目会与
                下方的 `interrupt_on` 参数合并（同名工具以用户传入的为准）。
                需要支持 `InterruptOnConfig` 上 `when` 谓词的 `langchain` 版本。

            子 agent 会继承这些规则，除非它们声明了自己的 `permissions` 字段
            （此时会完全替换父级规则）。

            `FilesystemMiddleware` 在工具层（针对其内置文件系统工具）应用这些
            权限，而不是在后端层。直接使用后端时目前不会纳入 `permissions`。
        backend: 可选的文件存储与执行后端。

            传入一个 `Backend` 实例（例如 `StateBackend()`）。

            若要支持执行，请使用实现了
            [`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol]
            的后端。
        interrupt_on: 工具名到 interrupt 配置的映射。

            传入后会在指定的工具调用前暂停 agent 执行，以便人工审批或修改。

            该配置始终作用于主 agent。

            对于子 agent：
            - 声明式 `SubAgent` 规格默认继承顶层 `interrupt_on` 配置。
            - 如果声明式 `SubAgent` 提供自己的 `interrupt_on`，则以该子 agent
                专属配置覆盖继承自顶层的配置。
            - `CompiledSubAgent` 的 runnable 不继承顶层 `interrupt_on`；
                请在编译好的 runnable 内部自行配置 human-in-the-loop 行为。
            - 远端 `AsyncSubAgent` 规格不继承顶层 `interrupt_on`；
                请在远端子 agent 上自行配置任何审批行为。

            例如 `interrupt_on={"edit_file": True}` 会在每次编辑前暂停。
        response_format: 用于 agent 的结构化输出响应格式。
        state_schema: agent 图的自定义状态 schema。必须是
            [`DeepAgentState`][deepagents.graph.DeepAgentState] 的 `TypedDict`
            子类，以保留内置在 `messages` 上的 `DeltaChannel` reducer。

            通常更推荐使用中间件来定义状态扩展，使新增字段的作用域局限在
            使用它们的钩子与工具内。

            传入时，本 schema 被用作图的基类 schema，并会与各中间件贡献的
            state schema 合并。它同时会下传给为 `task` 工具编译的声明式
            [`SubAgent`][deepagents.middleware.subagents.SubAgent] 规格，
            以使子 agent 看到与父级相同的自定义字段。

            [`CompiledSubAgent`][deepagents.middleware.subagents.CompiledSubAgent]
            的 runnable 不会继承此 schema，因为它们已经编译完成 —— 若需要
            访问相同的自定义状态字段，请在编译时为这些 runnable 配置兼容
            的 state schema。
            远端
            [`AsyncSubAgent`][deepagents.middleware.async_subagents.AsyncSubAgent]
            规格同样使用其远端图上配置的 schema。

            ```python
            from deepagents.graph import DeepAgentState


            class MyState(DeepAgentState):
                page_url: str
                file_urls: list[str]


            agent = create_deep_agent(model=..., state_schema=MyState)
            ```
        context_schema: 定义不可变、运行期作用域的 context 的 schema 类。

            原样透传给 [`create_agent`][langchain.agents.create_agent]。
        checkpointer: 可选的 `Checkpointer`，用于跨次运行持久化 agent 状态。

            原样透传给 [`create_agent`][langchain.agents.create_agent]。
        store: 可选的持久化存储（如果后端使用 `StoreBackend` 则必填）。

            原样透传给 [`create_agent`][langchain.agents.create_agent]。
        debug: 是否启用 debug 模式。

            原样透传给 [`create_agent`][langchain.agents.create_agent]。
        name: agent 的名称。

            原样透传给 [`create_agent`][langchain.agents.create_agent]。
        cache: agent 使用的缓存。

            原样透传给 [`create_agent`][langchain.agents.create_agent]。

    Returns:
        一个已配置好的 deep agent。

    Raises:
        ImportError: 当必需 provider 包缺失或低于最低支持版本时（例如
            `langchain-openrouter`）。
        ValueError: 当 `HarnessProfile.excluded_middleware` 引用了 harness
            受保护脚手架集合中的某个类（例如
            [`FilesystemMiddleware`][deepagents.middleware.filesystem.FilesystemMiddleware]
            或
            [`SubAgentMiddleware`][deepagents.middleware.subagents.SubAgentMiddleware]）、
            使用了私有（下划线开头）名称、与多个不同的中间件类冲突，或者在
            已组装栈中匹配不到任何条目。
    """
    # `DeepAgentState` 是 `TypedDict`；TypedDict 不支持 `issubclass`，
    # 因此 `state_schema` 的子类约束仅通过类型系统保证，不在运行时校验。

    # 记录原始的字符串模型规格，以便按规格（而非解析后的实例）查询 profile
    _model_spec: str | None = model if isinstance(model, str) else None

    if model is None:
        warn_deprecated(
            since="0.5.3",
            removal="1.0.0",
            message=(
                "Passing `model=None` to `create_deep_agent` is deprecated "
                "and will be removed in deepagents==1.0.0. The `model` "
                "parameter type will change from `BaseChatModel | str | None` "
                "to `BaseChatModel | str`. Specify a model explicitly "
                "(e.g., `ChatAnthropic(model_name=...)`). See "
                "https://docs.langchain.com/oss/python/deepagents/models"
            ),
            package="deepagents",
        )
        # 走未加装饰的 builder，避免在 `get_default_model` 的去重标记上扣动一次扳机
        # —— 直接的用户调用仍只看到一次警告
        model = _build_default_model()
    else:
        model = resolve_model(model)
    # 根据解析后的模型挑选最匹配的 harness profile（部分模型可能挂专属 profile）
    _profile = _harness_profile_for_model(model, _model_spec)
    # 校验 profile 级别的不变量：受保护脚手架类、私有名称等
    _validate_excluded_middleware_config(
        _profile,
        required_classes=_REQUIRED_MIDDLEWARE_CLASSES,
        required_names=_REQUIRED_MIDDLEWARE_NAMES,
    )
    # 累计主 agent + 通用子 agent 两个栈中匹配到的条目（两个栈都使用 `_profile`）。
    # profile 级条目只需在某个栈中命中即可，并非每个栈都必须命中，因此覆盖校验
    # 在所有过滤结束后统一执行一次。
    _main_matched_classes: set[type[AgentMiddleware[Any, Any, Any]]] = set()
    _main_matched_names: set[str] = set()

    # 复制一份 `tools`，并按 harness 专属规则改写描述。
    # （工具排除由 _ToolExclusionMiddleware 在一处统一处理，
    #  无论工具来自用户还是中间件注入。）
    _tools = _apply_tool_description_overrides(
        tools,
        _profile.tool_description_overrides,
    )

    backend = backend if backend is not None else StateBackend()

    # 先处理调用方传入的子 agent，这样是否自动添加默认通用子 agent 的决策可以
    # 考虑显式覆写，并且其中间件栈（含任何基于工厂的 `extra_middleware`）不会
    # 被构建后又被丢弃。
    inline_subagents: list[SubAgent | CompiledSubAgent] = []
    async_subagents: list[AsyncSubAgent] = []
    for spec in subagents or []:
        if "graph_id" in spec:
            # 该 spec 是 AsyncSubAgent，路由到 async 中间件
            async_subagents.append(cast("AsyncSubAgent", spec))
            continue
        if "runnable" in spec:
            # CompiledSubAgent —— 原样使用
            inline_subagents.append(spec)
        else:
            # SubAgent —— 填入默认值并前置基础中间件
            raw_subagent_model = spec.get("model", model)
            subagent_model = resolve_model(raw_subagent_model)

            _subagent_spec = raw_subagent_model if isinstance(raw_subagent_model, str) else None
            _subagent_profile = _harness_profile_for_model(subagent_model, _subagent_spec)

            # 权限解析：子 agent 自己的规则优先；否则继承父级
            subagent_permissions = spec.get("permissions", permissions)

            # 构造中间件：基础栈 + skills（若指定）+ 用户中间件
            subagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
                TodoListMiddleware(),
                FilesystemMiddleware(
                    backend=backend,
                    custom_tool_descriptions=_subagent_profile.tool_description_overrides,
                    _permissions=subagent_permissions,
                ),
                create_summarization_middleware(subagent_model, backend),
                PatchToolCallsMiddleware(),
            ]
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
            subagent_middleware.extend(spec.get("middleware", []))

            # 该子 agent 模型对应的 harness profile 中间件
            subagent_middleware.extend(_subagent_profile.materialize_extra_middleware())
            if _subagent_profile.excluded_tools:
                subagent_middleware.append(_ToolExclusionMiddleware(excluded=_subagent_profile.excluded_tools))

            _append_prompt_caching_middleware(subagent_middleware)

            _subagent_matched_classes: set[type[AgentMiddleware[Any, Any, Any]]] = set()
            _subagent_matched_names: set[str] = set()
            _validate_excluded_middleware_config(
                _subagent_profile,
                required_classes=_REQUIRED_MIDDLEWARE_CLASSES,
                required_names=_REQUIRED_MIDDLEWARE_NAMES,
            )
            subagent_middleware = _apply_excluded_middleware(
                subagent_middleware,
                _subagent_profile,
                matched_classes=_subagent_matched_classes,
                matched_names=_subagent_matched_names,
            )
            _verify_excluded_middleware_coverage(
                _subagent_profile,
                _subagent_matched_classes,
                _subagent_matched_names,
                required_classes=_REQUIRED_MIDDLEWARE_CLASSES,
                required_names=_REQUIRED_MIDDLEWARE_NAMES,
            )

            # 合并父级的 interrupt_on 与子 agent 自身的配置；再叠上从权限派生的 fs 中断
            subagent_interrupt_on = spec.get("interrupt_on", interrupt_on)
            subagent_interrupt_on = _merge_fs_interrupt_on(
                _build_interrupt_on_from_permissions(subagent_permissions or []),
                subagent_interrupt_on,
            )

            # 除非子 agent 显式声明自己的 tools，否则继承父级的 tools。
            # 描述会被改写；排除由中间件统一处理。
            raw_subagent_tools = spec.get("tools") if "tools" in spec else tools
            subagent_tools = _apply_tool_description_overrides(
                raw_subagent_tools,
                _subagent_profile.tool_description_overrides,
            )

            processed_spec: SubAgent = {
                **spec,
                "model": subagent_model,
                "tools": subagent_tools or [],
                "middleware": subagent_middleware,
            }
            processed_spec["system_prompt"] = _apply_profile_prompt(_subagent_profile, spec["system_prompt"])
            if subagent_interrupt_on is not None:
                processed_spec["interrupt_on"] = subagent_interrupt_on
            inline_subagents.append(processed_spec)

    # 除非 harness profile 禁用了默认通用子 agent 或者调用方已经提供了自己的，
    # 否则自动添加默认通用子 agent —— 调用方传入显式 spec 是覆写默认的方式。
    # 在这些情况下跳过也能避免构造一个随后被丢弃的基于工厂的 `extra_middleware`。
    gp_profile = _profile.general_purpose_subagent or GeneralPurposeSubagentProfile()
    if gp_profile.enabled is not False and not any(spec["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for spec in inline_subagents):
        gp_middleware: list[AgentMiddleware[Any, Any, Any]] = [
            TodoListMiddleware(),
            FilesystemMiddleware(
                backend=backend,
                custom_tool_descriptions=_profile.tool_description_overrides,
                _permissions=permissions,
            ),
            create_summarization_middleware(model, backend),
            PatchToolCallsMiddleware(),
        ]
        if skills is not None:
            gp_middleware.append(SkillsMiddleware(backend=backend, sources=skills))

        # 追加 harness profile 中间件（若有）
        gp_middleware.extend(_profile.materialize_extra_middleware())

        # 在所有工具注入中间件运行之后，再剥离 excluded_tools
        if _profile.excluded_tools:
            gp_middleware.append(_ToolExclusionMiddleware(excluded=_profile.excluded_tools))
        _append_prompt_caching_middleware(gp_middleware)

        gp_middleware = _apply_excluded_middleware(
            gp_middleware,
            _profile,
            matched_classes=_main_matched_classes,
            matched_names=_main_matched_names,
        )

        general_purpose_spec: SubAgent = {
            **GENERAL_PURPOSE_SUBAGENT,
            "model": model,
            "tools": _tools or [],
            "middleware": gp_middleware,
        }
        if gp_profile.description is not None:
            general_purpose_spec["description"] = gp_profile.description
        if gp_profile.system_prompt is not None:
            # GP 专属的覆写优先于 `profile.base_system_prompt`；只有 profile 后缀
            # 会再叠加在它之上。
            gp_prompt = gp_profile.system_prompt
            if _profile.system_prompt_suffix is not None:
                gp_prompt = gp_prompt + "\n\n" + _profile.system_prompt_suffix
            general_purpose_spec["system_prompt"] = gp_prompt
        else:
            general_purpose_spec["system_prompt"] = _apply_profile_prompt(_profile, GENERAL_PURPOSE_SUBAGENT["system_prompt"])
        # 把从权限派生的 fs 中断与顶层 interrupt_on 合并
        gp_interrupt_on = _merge_fs_interrupt_on(
            _build_interrupt_on_from_permissions(permissions or []),
            interrupt_on,
        )
        if gp_interrupt_on is not None:
            general_purpose_spec["interrupt_on"] = gp_interrupt_on

        # 默认通用子 agent 放在最前，符合调用方期望的最常用入口位置
        inline_subagents.insert(0, general_purpose_spec)

    # 构造主 agent 中间件栈
    deepagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        TodoListMiddleware(),
    ]
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    deepagent_middleware.append(
        FilesystemMiddleware(
            backend=backend,
            custom_tool_descriptions=_profile.tool_description_overrides,
            _permissions=permissions,
        )
    )
    # 只有存在内联同步子 agent 时才挂载 SubAgentMiddleware（避免多余的 task 工具）
    sub_agent_middleware: SubAgentMiddleware | None = None
    if inline_subagents:
        sub_agent_middleware = SubAgentMiddleware(
            backend=backend,
            subagents=inline_subagents,
            # 覆写 task 工具的描述。值中应包含 `{available_agents}` —— 一个会被
            # 替换为子 agent 名称/描述列表的格式化占位符。缺少它模型就看不到
            # 有哪些子 agent 可用。传 None（默认）使用内置模板。如果 task 工具
            # 被改名，残留的占位符键会被静默忽略。
            task_description=_profile.tool_description_overrides.get("task"),
            state_schema=state_schema,
        )
        deepagent_middleware.append(sub_agent_middleware)
    deepagent_middleware.extend(
        [
            create_summarization_middleware(model, backend),
            PatchToolCallsMiddleware(),
        ]
    )

    if async_subagents:
        # 这里的 async 含义是非阻塞运行这些子 agent，
        # 目前支持通过 LangSmith deployments 部署的 agent。
        deepagent_middleware.append(AsyncSubAgentMiddleware(async_subagents=async_subagents))

    if middleware:
        deepagent_middleware.extend(middleware)
    # harness profile 中间件放在用户中间件与 memory 之间，
    # 这样 memory 更新（会改动系统提示）不会作废 Anthropic prompt 缓存的前缀。
    deepagent_middleware.extend(_profile.materialize_extra_middleware())
    if _profile.excluded_tools:
        deepagent_middleware.append(_ToolExclusionMiddleware(excluded=_profile.excluded_tools))
    _append_prompt_caching_middleware(deepagent_middleware)
    if memory is not None:
        # MemoryMiddleware 仅在请求模型是 Anthropic 时才施加 cache_control 断点，
        # 因此可以无条件启用而不会引入副作用。
        deepagent_middleware.append(
            MemoryMiddleware(
                backend=backend,
                sources=memory,
                add_cache_control=True,
            )
        )
    # 合并从权限派生的 fs 中断与用户传入的 interrupt_on
    main_interrupt_on = _merge_fs_interrupt_on(
        _build_interrupt_on_from_permissions(permissions or []),
        interrupt_on,
    )
    if main_interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=main_interrupt_on))
    # 应用 profile 的 excluded_middleware，并累计命中以便后续校验
    deepagent_middleware = _apply_excluded_middleware(
        deepagent_middleware,
        _profile,
        matched_classes=_main_matched_classes,
        matched_names=_main_matched_names,
    )
    # 汇总所有中间件声明的私有 state 字段名，下发给 SubAgentMiddleware，
    # 使其在跨子 agent 调用时不会把这些内部字段泄露出去
    private_state_keys = private_state_field_names(*(mw.state_schema for mw in deepagent_middleware if getattr(mw, "state_schema", None) is not None))
    if sub_agent_middleware is not None:
        sub_agent_middleware.private_state_keys = private_state_keys
    # 校验主 profile 的每条排除项在主 agent 栈或 GP 子 agent 栈中至少命中一处。
    # 若两边都没命中，几乎一定是写错或 profile 已过期。
    _verify_excluded_middleware_coverage(
        _profile,
        _main_matched_classes,
        _main_matched_names,
        required_classes=_REQUIRED_MIDDLEWARE_CLASSES,
        required_names=_REQUIRED_MIDDLEWARE_NAMES,
    )

    # 依据 profile 决定使用其 base_system_prompt 还是 SDK 默认的 BASE
    base_prompt = _apply_profile_prompt(_profile, BASE_AGENT_PROMPT)
    if system_prompt is None:
        # 没有用户指令时，prompt 仅为 base（可能再叠加 profile 后缀）
        final_system_prompt: str | SystemMessage = base_prompt
    elif isinstance(system_prompt, SystemMessage):
        # 保留 SystemMessage 上原有的 cache_control 标记，把 base 作为额外文本块追加
        final_system_prompt = SystemMessage(content_blocks=[*system_prompt.content_blocks, {"type": "text", "text": f"\n\n{base_prompt}"}])
    else:
        # 字符串形式直接拼接，base 紧跟在用户指令之后
        final_system_prompt = system_prompt + "\n\n" + base_prompt

    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=_tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
        state_schema=state_schema if state_schema is not None else DeepAgentState,
    ).with_config(
        {
            "recursion_limit": 9_999,
            "metadata": {
                "ls_integration": "deepagents",
                "lc_versions": {"deepagents": __version__},
                "lc_agent_name": name,
            },
        }
    )
