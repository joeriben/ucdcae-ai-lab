#!/bin/bash
# 6 — Run i18n Batch Translator
# Processes all pending work orders in src/i18n/WORK_ORDERS.md
# Translates English keys into DE/TR/KO/UK/FR

cd "$(dirname "$0")"

echo "=== i18n Batch Translator ==="
echo "Checking for pending work orders..."

if ! grep -q "^### WO-" public/ai4artsed-frontend/src/i18n/WORK_ORDERS.md 2>/dev/null; then
    echo "No pending work orders found. Nothing to do."
    exit 0
fi

echo "Starting Claude i18n-translator agent..."
claude -p "Process all pending i18n work orders. Read WORK_ORDERS.md, translate all pending entries into de/tr/ko/uk/fr, run type-check, and commit."
