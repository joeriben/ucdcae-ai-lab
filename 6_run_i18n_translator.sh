#!/bin/bash
# 6 — Run i18n Batch Translator (nightly, unattended)
# Processes pending work orders + auto-detects missing keys
# Runs without permission prompts — safe for AFK / cron usage
#
# Usage:
#   ./6_run_i18n_translator.sh           # foreground (interactive terminal)
#   ./6_run_i18n_translator.sh --nightly # tmux session + log file (cron/AFK)

cd "$(dirname "$0")"

LOGDIR="logs/i18n"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/$(date +%Y%m%d_%H%M%S).log"

PROMPT='You are the i18n batch translator. Follow the workflow in .claude/agents/i18n-translator.md exactly:
1. Run the auto-audit (compare en.ts keys vs all 8 target files)
2. Process all pending work orders from WORK_ORDERS.md
3. Translate any missing keys found by the audit
4. Run apostrophe validation, type-check, build
5. Move completed WOs, commit with chore(i18n): prefix
Target languages: de/tr/ko/uk/fr/es/he/ar (8 total).'

if [[ "$1" == "--nightly" ]]; then
    echo "=== i18n Nightly Translator ==="
    echo "Log: $LOGFILE"
    echo "tmux session: i18n-nightly"

    tmux new-session -d -s i18n-nightly \
        "cd $(pwd) && \
         claude -p \
           --permission-mode bypassPermissions \
           --model sonnet \
           --allowedTools 'Read,Edit,Write,Bash,Glob,Grep' \
           \"$PROMPT\" \
         2>&1 | tee $LOGFILE; \
         echo ''; \
         echo '=== Done. Attach with: tmux attach -t i18n-nightly ==='; \
         sleep 86400" \
    && echo "Started. Attach with: tmux attach -t i18n-nightly" \
    || echo "Failed to start tmux session (already running?)"
else
    echo "=== i18n Batch Translator (interactive) ==="
    echo "Log: $LOGFILE"

    claude -p \
      --permission-mode bypassPermissions \
      --model sonnet \
      --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
      "$PROMPT" \
    2>&1 | tee "$LOGFILE"
fi
