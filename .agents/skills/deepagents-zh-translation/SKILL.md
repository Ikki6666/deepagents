---
name: deepagents-zh-translation
description: 维护 Deep Agents 个人中文翻译 fork 的分支、同步与发布工作流。当需要同步 upstream、创建或切换 zh / zh-tag / 主题分支、推送到 Ikki6666/deepagents，或在不同 release tag 之间迁移翻译时使用。
---

# Deep Agents 中文翻译 Fork

只用于当前个人 fork 的分支与同步；上游 PR 仍按仓库的 `<github-username>/<scope>/<description>` 规范。

## 远程

- `origin`：`git@github.com:Ikki6666/deepagents.git`，个人 fork，所有翻译提交推这里。
- `upstream`：`git@github.com:langchain-ai/deepagents.git`，官方仓库，只 fetch。
- `main` 跟踪 `upstream/main`，`pushRemote=origin`；`remote.pushDefault=origin`。

## 分支模型

| 用途 | 分支 | 生命周期 | 基准 |
| --- | --- | --- | --- |
| 官方镜像 | `main` | 永久 | 与 `upstream/main` 完全一致，不允许本地提交 |
| 主线中文翻译 | `zh` | 长期 | 从 `main` 同步；翻译提交都落在这里 |
| 指定版本中文翻译 | `zh-tag/<package>-<version>` | 该版本长期 | 指定 release tag；不合并 `main` |
| 临时主题 | `zh-<scope>/<short-description>` | 短期 | 从 `zh` 切出，完成后合回 `zh` |
| 存档 | `zh-archive-YYYYMMDD` 或 tag `archive/zh-YYYYMMDD` | 永久 | 退役分支的快照 |

当前示例：`zh`、`zh-tag/deepagents-0.7.42`（tag `deepagents==0.7.42`）、`zh-archive-20260629`。

## 硬性规则

- `main` 永远不做本地提交；它的内容必须与 `upstream/main` 一致。
- 不把 `main` 或 `zh` merge 进 `zh-tag/...`；tag 分支只做该 tag 上的翻译提交。
- 不创建 `zh/...` 分支。Git 不能同时存在 `refs/heads/zh` 和 `refs/heads/zh/...`；用 `zh-tag/...` 或 `zh-<scope>/...`。
- 已推送的 `zh`、`zh-tag/...`、主题分支不要 rebase 或 force-push；用 merge 或 cherry-pick。
- 新 release tag 建新分支，不移动旧 tag 分支。
- 做翻译前先确认 `main` 与 `origin/main` 一致；不一致先快进。

## 同步 main

```bash
git switch main
git fetch https://github.com/langchain-ai/deepagents.git \
  main:refs/remotes/upstream/main --no-tags
git merge --ff-only upstream/main
git push
```

这里用 HTTPS 是有意的：当前环境 SSH fetch 很慢。`main` 的 `pushRemote=origin`，所以 `git push` 会推到 fork 的 `main`。

## 更新 zh

```bash
git switch zh
git merge main
git push
```

每次开主题分支前先做这一步。解决冲突时保留翻译意图，代码标识符和行为不要改。

## 主题分支

```bash
git switch zh
git merge main
git switch -c zh-sdk/graph-docstrings
# 翻译、提交
git push -u origin zh-sdk/graph-docstrings

git switch zh
git merge zh-sdk/graph-docstrings
git push
git branch -d zh-sdk/graph-docstrings
git push origin --delete zh-sdk/graph-docstrings
```

小改动可以直接提交到 `zh`；批量、实验或并行工作再用主题分支。

## tag 基准分支

```bash
git fetch --tags https://github.com/langchain-ai/deepagents.git
git branch zh-tag/deepagents-0.7.42 'deepagents==0.7.42'
git push -u origin zh-tag/deepagents-0.7.42
```

tag 名里的 `==` 在分支名里替换成 `-`。新建后确认分支指向 tag commit：

```bash
git rev-parse 'deepagents==0.7.42^{commit}'
git rev-parse zh-tag/deepagents-0.7.42
```

把翻译迁到新 tag：新建 `zh-tag/<package>-<new-version>`，再 cherry-pick 或 `git diff`/`apply` 需要的翻译提交；不要整条 merge 旧 tag 分支。

## 命名

- 主线：`zh`
- tag：`zh-tag/<package>-<version>`，如 `zh-tag/deepagents-code-0.2.15`
- 主题：`zh-<scope>/<short-description>`，如 `zh-sdk/graph-docstrings`、`zh-code/slash-commands`
- 多人主题：`<github-username>/zh/<scope>/<short-description>`
- 存档分支：`zh-archive-YYYYMMDD`
- 存档 tag：`archive/zh-YYYYMMDD`；现有的 `archive/YYYYMMDD` 是 main 快照

不要用纯日期活动分支、`main-zh`、`chinese`、`translation`，也不要按模块建长期分支（如 `zh-sdk`）。

## 提交信息

- 翻译内容：`docs(zh): ...`
- 翻译工具/流程：`chore(zh): ...`
- 官方同步：`docs(zh): sync upstream YYYY-MM-DD (N commits)`

## 交付前检查

```bash
git branch -vv
git status -sb
git rev-list --left-right --count main...upstream/main   # 期望 0 0
git rev-list --left-right --count main...origin/main     # 期望 0 0
git ls-remote --heads origin refs/heads/zh refs/heads/zh-tag/deepagents-0.7.42
```

新建 tag 分支时还要确认 HEAD 与 tag commit 相同。

## 本地杂项

`.codex/`、`langchain-event-streaming-tutorial.md`、`libs/deepagents/scripts/event_streaming_*.py`、`分析/` 是本地草稿；除非用户明确要求，不要加进翻译提交。
