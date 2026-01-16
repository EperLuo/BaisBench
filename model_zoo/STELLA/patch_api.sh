#!/bin/bash
# ----------------------------------------
# STELLA API base URL patch script
# 替换 OpenRouter 接口为 https://api.openai-next.com/v1/
# ----------------------------------------

set -e

# 新的 API 地址
NEW_URL="https://api.openai-next.com/v1/"

echo "🔍 正在备份并替换 STELLA 中的 OpenRouter 地址为：$NEW_URL"
echo

# 要修改的文件列表
FILES=(
  "predefined_tools.py"
  "stella_core.py"
  "memory_manager.py"
  "Knowledge_base.py"
  "new_tools/database_tools.py"
  "new_tools/llm.py"
)

for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    echo "📄 修改文件: $f"
    cp "$f" "$f.bak"  # 备份原文件
    sed -i "s|https://api.openai-next.com/v1/|$NEW_URL|g" "$f"
  else
    echo "⚠️ 未找到文件: $f"
  fi
done

echo
echo "✅ 替换完成！所有原文件已备份为 *.bak"
echo "请在 .env 文件中设置："
echo
echo "  OPENAI_API_KEY=你的_api_key"
echo "  BASE_URL=$NEW_URL"
echo
echo "运行前记得执行:  source .env"
