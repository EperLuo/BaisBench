#!/usr/bin/env bash
set -euo pipefail

# === 你需要改的地方：器官文件夹的父目录（里面每个子文件夹名就是器官名）===
ORGANS_ROOT="./examples"   # 例如：./examples 下有 Adipose/ Brain/ Liver/ ...

# === 固定参数（按你原命令）===
MODEL="gpt-4.1"
EXECUTE="True"

# === OpenAI Key：用环境变量提供（不要写进脚本）===
# : "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before running.}"
OPENAI_API_KEY="sk-kIjA9Jgja2TPF6UD0f9b788028684aB7B9B282F2A582FcD8"

# 记录总开始时间
total_start=$(date +%s)

# 遍历所有子目录（每个子目录名=器官名）
for organ_dir in "$ORGANS_ROOT"/*/; do
  # 如果没有匹配目录，bash 可能会原样返回；做个保护
  [[ -d "$organ_dir" ]] || continue

  organ="$(basename "${organ_dir%/}")"
  config="${organ_dir}/config.yaml"
  outdir="${organ_dir}/output"

  # 只处理存在 config.yaml 的器官目录（避免扫到无关目录）
  if [[ ! -f "$config" ]]; then
    echo "[SKIP] $organ (no config.yaml at $config)"
    continue
  fi

  mkdir -p "$outdir"

  raw_log="${outdir}/autoba.log"
  clean_log="${outdir}/autoba_clean.log"

  echo "============================================================"
  echo "[RUN ] Organ: $organ"
  echo "Config: $config"
  echo "Logs : $raw_log / $clean_log"
  echo "Start: $(date)"
  organ_start=$(date +%s)

  # 1) 运行 app.py，并把 stdout+stderr 记录到 autoba.log（同时打印到终端）
  python app.py \
    --config "$config" \
    --openai "$OPENAI_API_KEY" \
    --model "$MODEL" \
    --execute "$EXECUTE" \
    2>&1 | tee "$raw_log"

  # 2) 清洗日志（去 \r、去 ANSI、过滤两类噪声行）
  perl -pe 's/\r/\n/g; s/\x1b\[[0-?]*[ -\/]*[@-~]//g; s/\x1b\][^\x07]*(\x07|\x1b\\)//g; s/\x1b[@-Z\\-_]//g;' "$raw_log" \
    | grep -vE 'Processing\.\.\.|Running run_python_code' \
    > "$clean_log"

  organ_end=$(date +%s)
  organ_elapsed=$((organ_end - organ_start))

  echo "[DONE] Organ: $organ | Elapsed: ${organ_elapsed}s | End: $(date)"
done

total_end=$(date +%s)
echo "============================================================"
echo "[ALL DONE] Total elapsed: $((total_end - total_start))s"



# python app.py --config ./examples/Adipose/config.yaml --openai sk-4fZXCtTZCuQrWU9JE06999C49e9241489f2bE77fB7229290 --model gpt-4.1-2025-04-14 --execute True 2>&1 | tee ./examples/Adipose/output/autoba.log

# perl -pe 's/\r/\n/g; s/\x1b\[[0-?]*[ -\/]*[@-~]//g; s/\x1b\][^\x07]*(\x07|\x1b\\)//g; s/\x1b[@-Z\\-_]//g;' "examples/Adipose/output/autoba.log" \
#   | grep -vE 'Processing\.\.\.|Running run_python_code' \
#   > "examples/Adipose/output/autoba_clean.log"