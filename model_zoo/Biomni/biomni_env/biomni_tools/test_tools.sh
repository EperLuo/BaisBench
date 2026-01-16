#!/bin/bash
echo "Testing if tools are in the PATH..."
echo "Current PATH: $PATH"
echo ""
echo "Looking for tools in: /data/lep/BaisBench/model_zoo/Biomni/biomni_env/biomni_tools/bin"
ls -la "/data/lep/BaisBench/model_zoo/Biomni/biomni_env/biomni_tools/bin"
echo ""
echo "Checking for path caching issues..."
for tool in $(ls "/data/lep/BaisBench/model_zoo/Biomni/biomni_env/biomni_tools/bin"); do
  which $tool 2>/dev/null | grep -q "/afs/cs.stanford.edu" && {
    echo "WARNING: $tool is still pointing to the old AFS location!"
    echo "Run 'hash -r' (bash) or 'rehash' (zsh) to clear the command cache."
    break
  }
done
echo ""
for tool in $(ls "/data/lep/BaisBench/model_zoo/Biomni/biomni_env/biomni_tools/bin"); do
  if command -v $tool &> /dev/null; then
    echo "$tool: $(which $tool)"
  else
    echo "$tool: NOT FOUND IN PATH"
  fi
done
