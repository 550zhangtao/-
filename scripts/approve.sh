#!/usr/bin/env bash
# approve.sh —— 修改/推送审批闸门（本地）
#
# 背景：本仓库约定"此后需要有用户运行才能改动"。`.githooks/pre-push` 会拦住
# 任何推往 origin/main 的提交，直到用户显式运行本脚本审批一次。
#
#   bash scripts/approve.sh install   # 一次性：启用 .githooks 钩子
#   bash scripts/approve.sh           # 审批：列出待推送提交，写 .git/approval
#   bash scripts/approve.sh show      # 看当前审批状态
#   bash scripts/approve.sh revoke    # 撤销审批
#
# 审批只覆盖**当时已经存在的提交**：审批后再 commit 的新提交会自动失效，
# 需要再跑一次 approve。这是故意的——"改完要我看过才算数"。
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
APPROVAL="$ROOT/.git/approval"
HOOK_DIR="$ROOT/.githooks"

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

cmd="${1:-approve}"
case "$cmd" in
    -h|--help|help) usage ;;
    install)
        git config core.hooksPath .githooks
        chmod +x "$HOOK_DIR/pre-push" 2>/dev/null || true
        echo "[approve] 已启用 .githooks（core.hooksPath=.githooks）"
        ;;
    show)
        if [ -f "$APPROVAL" ]; then
            cat "$APPROVAL"
        else
            echo "[approve] 尚未审批（$(git rev-parse --short=12 HEAD) 未授权推送 origin/main）"
        fi
        ;;
    revoke)
        rm -f "$APPROVAL"
        echo "[approve] 已撤销审批"
        ;;
    approve)
        head="$(git rev-parse HEAD)"
        branch="$(git rev-parse --abbrev-ref HEAD)"
        range="HEAD"
        if git rev-parse --verify -q "refs/remotes/origin/main" >/dev/null; then
            range="origin/main..HEAD"
        fi
        echo "分支: $branch"
        echo "待推送提交:"
        git log --oneline --no-decorate $range | sed 's/^/  /'
        echo
        echo "变更文件:"
        git diff --stat $range | sed 's/^/  /'
        echo
        if [ -t 0 ]; then
            read -r -p "确认允许这版代码进入 origin/main？[y/N] " ans
            case "$ans" in
                y|Y|yes|YES) ;;
                *) echo "[approve] 取消"; exit 1 ;;
            esac
        fi
        # 累积写入：head= 是当前 HEAD；approved 行是"已审批提交"白名单。
        # 之前审批过的提交保留，这样一次 push 多个已审批提交不会被误拦。
        {
            echo "head=$(git rev-parse HEAD)"
            echo "branch=$branch"
            echo "by=${USER:-$(id -un)}"
            echo "at=$(date '+%Y-%m-%d %H:%M:%S %z')"
            for sha in $(git rev-list $range); do
                echo "approved $sha"
            done
        } > "$APPROVAL.new"
        mv "$APPROVAL.new" "$APPROVAL"
        echo "[approve] OK：HEAD=$(git rev-parse --short=12 HEAD) 已授权。"
        echo "          计划推送时用：git push origin $branch"
        ;;
    *)
        echo "未知命令: $cmd" >&2
        usage
        ;;
esac
