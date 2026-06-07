#!/usr/bin/env bash
set -euo pipefail

R=/sys/fs/resctrl

[ "$(id -u)" -eq 0 ] || {
    echo "run with sudo"
    exit 1
}

mountpoint -q "$R" || mount -t resctrl resctrl "$R"

# すべての subgroup の task を root group に戻す
find "$R" \
    -path "$R/info" -prune -o \
    -path "$R/mon_data" -prune -o \
    -name tasks -type f -print |
while read -r f; do
    [ "$f" = "$R/tasks" ] && continue
    while read -r t; do
        [ -n "$t" ] && echo "$t" > "$R/tasks" 2>/dev/null || true
    done < "$f"
done

# monitor group / control group を深い順に削除
find "$R" -mindepth 1 -depth -type d \
    ! -path "$R/info*" \
    ! -path "$R/mon_data*" \
    -exec rmdir {} + 2>/dev/null || true

echo "resctrl cleanup done"