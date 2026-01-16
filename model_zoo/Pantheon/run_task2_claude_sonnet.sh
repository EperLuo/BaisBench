#!/usr/bin/env bash
set -euo pipefail

mkdir -p output_task2_claude_sonnet

command -v pantheon-cli >/dev/null 2>&1 || { echo "[ERROR] pantheon-cli not found in PATH"; exit 1; }

i=0
for file in prompt_task2/*.txt; do
  i=$((i+1))
  echo "======================================"
  echo "RUN $i : $file"
  echo "START: $(date)"
  echo "======================================"

  raw="./output_task2_claude_sonnet/run_process_${i}.raw.log"
  clean="./output_task2_claude_sonnet/run_process_${i}.clean.log"

  script -q -f -c "pantheon-cli" "$raw" <<EOF
$(cat "$file")

EOF

  perl -pe 's/\r/\n/g; s/\x1b\[[0-?]*[ -\/]*[@-~]//g; s/\x1b\][^\x07]*(\x07|\x1b\\)//g; s/\x1b[@-Z\\-_]//g;' "$raw" \
  | grep -vE 'Processing\.\.\.|Running run_python_code' \
  > "$clean"

  echo "END: $(date)"
done
