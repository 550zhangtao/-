# main 分支保护：改代码必须由用户审批

> 约定（2026-10-23）：**此后任何代码改动都必须经过用户本人的运行/审批才能进入
> `origin/main`。** AI 会话可以改工作区、可以提交到 `agent/*` 分支，但不能自行推送 main。

## 1. 本地闸门（已启用）

仓库内提供了 `.githooks/pre-push` + `scripts/approve.sh`：

```bash
bash scripts/approve.sh install   # 一次性：git config core.hooksPath .githooks
bash scripts/approve.sh           # 审批当前 HEAD（会列出待推送提交/改动，要求确认）
git push origin main              # 审批通过后才允许推送
bash scripts/approve.sh show      # 查看审批状态
bash scripts/approve.sh revoke    # 撤销
```

规则：

1. 推 `main` 时，本次**新增的每个提交**都必须出现在 `.git/approval` 的
   `approved <sha>` 列表里，否则拒绝；审批记录不存在也拒绝；
2. 审批只覆盖审批那一刻的 HEAD——**审批后再 `git commit`，新提交自动变回未审批**，
   必须再跑一次 `approve.sh`；
3. 推非 `main` 分支不受影响（`agent/*` 可以直接推，用于备份/评审）；
4. `git push --no-verify` 是逃生门，用于"首次安装闸门"这类引导动作，日常不要用。

`.git/approval` 在 `.git/` 内，不会进版本库；`git config core.hooksPath` 也是本地配置。

首次安装闸门时自身也要能推上去，所以**引导流程**是：

```bash
# 一次性（由用户执行）：把闸门本身推上去
git push --no-verify origin main
# 之后
bash scripts/approve.sh install
```

## 2. GitHub 侧（建议在网页里开，1 分钟）

本地钩子只能拦住本机这个 clone。真正的"只有我能改"建议在 GitHub 仓库设置里开：

**Settings → Branches → Add branch ruleset**（或 Branch protection rule）：

- Branch name pattern：`main`
- ☑ Require a pull request before merging
  - ☑ Require approvals：`1`
  - ☑ Dismiss stale pull request approvals when new commits are pushed
- ☑ Require status checks to pass（如有 CI）
- ☑ Do not allow bypassing the above settings
  （个人仓库里这条会连管理员一起约束，真正做到"只有你审批才能合"）
- ☑ Restrict who can push to matching branches（可选，只留自己）

效果：AI/其他 clone 只能推 `agent/*` 分支或开 PR，**合并进 `main` 必须由你在网页上点
Approve + Merge**。

## 3. 日常协作流程（AI 会话）

1. AI 在 `agent/<主题>` 分支上改代码并提交，**不推 main**；
2. AI 给出改动摘要 + 验证结果（AGENTS 第 1 节的"三处落地 + 板端绘制自检"）；
3. 用户 review 后运行 `bash scripts/approve.sh` 并 `git push origin main`，
   或在 GitHub PR 里点 Approve/Merge。

## 4. 回滚

```bash
git log --oneline             # 找到上一版
git checkout -b revert/<topic> <good-sha>
git push origin revert/<topic>
```

或 `git revert <sha>` 后走同样的审批流程。**不要** `git push --force` 到 main。
