#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, runpy, os

# 确保 STELLA 主目录在路径中
sys.path.append("/home/liyaru/software/stella/STELLA-main")

# ==== 运行 stella_core.py，让它自己创建 manager_agent ====
print("⚙️  Loading and initializing STELLA core system...")
runpy.run_module("stella_core", run_name="__main__")

# ==== 再次导入已经初始化好的 manager_agent ====
from stella_core import manager_agent



task = """
Task:  
Please load the single-cell transcriptomics data file located at the following path, perform a comprehensive biological analysis, and generate a well-structured, detailed research paper.
File path:  
/home/liyaru/software/stella/STELLA-main/data_to_text/E-MTAB_Embryos_process.h5ad
Data description:  
- Format: .h5ad, compliant with the AnnData standard.  
- Contains: expression matrix, cell metadata (obs), and gene metadata (var).  
- Dataset name: **E-MTAB_Embryos**, derived from an embryo development-related study.
Analysis requirements:  
1. Data preprocessing:  
  - Normalization: perform normalization (e.g., TPM, log1p transformation) and identify highly variable genes.  
  - Dimensionality reduction and clustering: run PCA, UMAP/t-SNE visualization, and perform Leiden/Louvain clustering.
2. Cell type annotation:  
  - Annotate clusters using known marker genes or automated tools (e.g., scType, SingleR, CellTypist).  
  - Provide annotation evidence (marker gene expression maps, heatmaps, dot plots, etc.).
3. Developmental trajectory analysis (if applicable):  
  - Construct differentiation trajectories (e.g., PAGA, Monocle3, Palantir) to identify potential developmental paths and key regulatory nodes.  
  - Visualize pseudotime dynamics and identify genes dynamically expressed across development.

All outputs (figures, scripts, text) should be saved to the current directory
"""

print("🚀 Running STELLA manager agent...")
result = manager_agent.run(task)
print("✅ Task completed.")
print(result)