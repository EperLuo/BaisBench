#!/bin/bash
# Script to fix path caching issues
echo "Fixing path caching issues..."
# Remove any old paths first to avoid duplicates
PATH=$(echo $PATH | tr ':' '\n' | grep -v "biomni_tools/bin" | tr '\n' ':' | sed 's/:$//')
export PATH="/data/lep/BaisBench/model_zoo/Biomni/biomni_env/biomni_tools/bin:$PATH"
# Clear the shell's command hash table to force it to re-search the PATH
hash -r 2>/dev/null || rehash 2>/dev/null || true
echo "Path fixed. Try running your command again."
