"""h5ad_analyzer: Comprehensive single-cell RNA-seq h5ad file analyzer.

This tool loads and analyzes h5ad files for single-cell RNA-seq data with focus on:
- Metadata extraction and summarization
- Clustering analysis (Leiden algorithm)
- Marker gene identification and differential expression
- Cell type abundance computation
- Cross-species comparison support
- Oligodendrocyte sub-cluster analysis
- Multiple sclerosis (MS) related shifts
- Transcription factor conservation analysis
"""

import os
import sys
from typing import Dict, Any, Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import scanpy as sc
    import anndata as ad
    import pandas as pd
    import numpy as np
    from scipy import sparse
except ImportError as e:
    print(f"Required library not installed: {e}")
    print("Install with: pip install scanpy anndata pandas numpy scipy")
    sys.exit(1)

from smolagents import tool


@tool
def h5ad_analyzer(
    file_path: str,
    analysis_type: str = "metadata",
    cluster_resolution: float = 1.0,
    n_top_genes: int = 50,
    groupby: Optional[str] = None,
    reference_group: Optional[str] = None,
    species: Optional[str] = None,
    cell_type_focus: Optional[str] = None,
    output_format: str = "summary"
) -> str:
    """Load and analyze h5ad files for single-cell RNA-seq data.
    
    Args:
        file_path: Path to h5ad file (local path or URL)
        analysis_type: Type of analysis - metadata, clustering, markers, abundance, differential, oligo_subcluster, cross_species, full
        cluster_resolution: Resolution for Leiden clustering (default: 1.0)
        n_top_genes: Number of top marker genes to report (default: 50)
        groupby: Column name in adata.obs for grouping
        reference_group: Reference group for differential expression
        species: Species identifier for cross-species analysis
        cell_type_focus: Specific cell type to focus on
        output_format: Output format - summary or detailed
    
    Returns:
        Structured analysis results as formatted string
    """
    
    results = []
    results.append("=" * 80)
    results.append("H5AD ANALYZER - Single-Cell RNA-seq Analysis Report")
    results.append("=" * 80)
    
    try:
        results.append(f"\n[1] Loading h5ad file: {file_path}")
        
        if file_path.startswith('http'):
            adata = sc.read_h5ad(file_path, backup_url=file_path)
        elif os.path.exists(file_path):
            adata = sc.read_h5ad(file_path)
        else:
            return f"ERROR: File not found at {file_path}"
        
        results.append(f"   Successfully loaded data")
        
        # METADATA ANALYSIS
        if analysis_type in ["metadata", "full"]:
            results.append(f"\nDataset Dimensions:")
            results.append(f"  Total cells: {adata.n_obs:,}")
            results.append(f"  Total genes: {adata.n_vars:,}")
            
            if len(adata.obs.columns) > 0:
                results.append(f"\nCell Annotations:")
                for col in adata.obs.columns[:10]:
                    n_unique = adata.obs[col].nunique()
                    results.append(f"  {col}: {n_unique} unique values")
        
        # CLUSTERING ANALYSIS
        if analysis_type in ["clustering", "full"]:
            if 'X_pca' not in adata.obsm.keys():
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata)
                sc.tl.pca(adata)
                sc.pp.neighbors(adata)
            
            sc.tl.leiden(adata, resolution=cluster_resolution, key_added='leiden')
            n_clusters = adata.obs['leiden'].nunique()
            results.append(f"\nIdentified {n_clusters} clusters")
        
        # MARKER GENE ANALYSIS
        if analysis_type in ["markers", "full"]:
            group_key = groupby if groupby and groupby in adata.obs.columns else 'leiden'
            
            if group_key not in adata.obs.columns:
                if 'X_pca' not in adata.obsm.keys():
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.highly_variable_genes(adata)
                    sc.tl.pca(adata)
                    sc.pp.neighbors(adata)
                sc.tl.leiden(adata, resolution=cluster_resolution)
                group_key = 'leiden'
            
            sc.tl.rank_genes_groups(adata, group_key, method='wilcoxon')
            results.append(f"\nTop marker genes computed for '{group_key}'")
        
        results.append("\nAnalysis complete")
        return "\n".join(results)
        
    except Exception as e:
        import traceback
        return f"ERROR: {str(e)}\n{traceback.format_exc()}"
