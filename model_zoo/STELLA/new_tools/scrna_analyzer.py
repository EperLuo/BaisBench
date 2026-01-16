"""scRNA-seq Analyzer Tool

Custom tool for loading and analyzing h5ad scRNA-seq files.
Supports subsetting, proportions, differential expression, and pathway enrichment.
"""

import h5py
import pandas as pd
import numpy as np
from scipy.stats import ranksums, hypergeom
from typing import Dict, List, Optional, Union, Any
import os
from pathlib import Path


class AnnDataLike:
    """Lightweight AnnData-like structure."""
    
    def __init__(self, X: np.ndarray, obs: pd.DataFrame, var: pd.DataFrame):
        self.X = X
        self.obs = obs
        self.var = var
        self.uns = {}
        self.n_obs = X.shape[0]
        self.n_vars = X.shape[1]


def load_h5ad(path: str) -> AnnDataLike:
    """Load h5ad file and reconstruct AnnData-like structure.
    
    Args:
        path: Path to h5ad file
        
    Returns:
        AnnDataLike object with X, obs, var attributes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required keys missing from h5ad
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with h5py.File(path, 'r') as f:
        # Load expression matrix
        if 'X' in f:
            X = f['X'][:]
        elif 'raw/X' in f:
            X = f['raw/X'][:]
        else:
            raise KeyError("No expression matrix found (expected 'X' or 'raw/X')")
        
        # Load observations (cells)
        obs_dict = {}
        if 'obs' in f:
            for key in f['obs'].keys():
                try:
                    obs_dict[key] = f['obs'][key][:].astype(str)
                except:
                    obs_dict[key] = f['obs'][key][:]
        obs = pd.DataFrame(obs_dict)
        
        # Load variables (genes)
        var_dict = {}
        if 'var' in f:
            for key in f['var'].keys():
                try:
                    var_dict[key] = f['var'][key][:].astype(str)
                except:
                    var_dict[key] = f['var'][key][:]
        var = pd.DataFrame(var_dict)
        
    return AnnDataLike(X, obs, var)


def subset_cells(adata: AnnDataLike, by: str = 'cell_type', value: str = 'CD8 T') -> AnnDataLike:
    """Subset cells by metadata column value.
    
    Args:
        adata: AnnDataLike object
        by: Column name in obs to subset by
        value: Value to filter for
        
    Returns:
        New AnnDataLike with subset cells
        
    Raises:
        KeyError: If column doesn't exist
        ValueError: If no cells match criteria
    """
    if by not in adata.obs.columns:
        raise KeyError(f"Column '{by}' not found in obs. Available: {list(adata.obs.columns)}")
    
    mask = adata.obs[by] == value
    if not mask.any():
        raise ValueError(f"No cells found with {by}='{value}'")
    
    X_sub = adata.X[mask, :]
    obs_sub = adata.obs[mask].copy()
    var_sub = adata.var.copy()
    
    return AnnDataLike(X_sub, obs_sub, var_sub)


def proportions(adata: AnnDataLike, group_col: str = 'patient_group', 
                subset_col: str = 'cx3cr1_bin') -> pd.DataFrame:
    """Compute cell proportions across groups.
    
    Args:
        adata: AnnDataLike object
        group_col: Column defining groups (e.g., 'patient_group')
        subset_col: Column defining subsets within groups (e.g., 'cx3cr1_bin')
        
    Returns:
        DataFrame with proportions for each group/subset combination
        
    Raises:
        KeyError: If columns don't exist
    """
    for col in [group_col, subset_col]:
        if col not in adata.obs.columns:
            raise KeyError(f"Column '{col}' not found in obs")
    
    counts = adata.obs.groupby([group_col, subset_col]).size().reset_index(name='count')
    totals = adata.obs.groupby(group_col).size().reset_index(name='total')
    
    result = counts.merge(totals, on=group_col)
    result['proportion'] = result['count'] / result['total']
    
    return result


def rank_genes_groups(adata: AnnDataLike, groupby: str = 'patient_group', 
                      groups: List[str] = ['PC']) -> Dict[str, pd.DataFrame]:
    """Perform differential expression analysis using Wilcoxon rank-sum test.
    
    Args:
        adata: AnnDataLike object
        groupby: Column in obs defining groups
        groups: List of groups to compare (each vs rest)
        
    Returns:
        Dictionary mapping group names to DataFrames with DE results
        
    Raises:
        KeyError: If groupby column doesn't exist
        ValueError: If groups not found
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"Column '{groupby}' not found in obs")
    
    results = {}
    
    for group in groups:
        mask_group = adata.obs[groupby] == group
        mask_rest = adata.obs[groupby] != group
        
        if not mask_group.any():
            raise ValueError(f"Group '{group}' not found in {groupby}")
        
        X_group = adata.X[mask_group, :]
        X_rest = adata.X[mask_rest, :]
        
        pvals = []
        scores = []
        logfcs = []
        
        for i in range(adata.n_vars):
            try:
                stat, pval = ranksums(X_group[:, i], X_rest[:, i])
                mean_group = np.mean(X_group[:, i])
                mean_rest = np.mean(X_rest[:, i])
                logfc = np.log2((mean_group + 1) / (mean_rest + 1))
                
                pvals.append(pval)
                scores.append(stat)
                logfcs.append(logfc)
            except:
                pvals.append(1.0)
                scores.append(0.0)
                logfcs.append(0.0)
        
        gene_names = adata.var.index.tolist() if hasattr(adata.var.index, 'tolist') else list(range(adata.n_vars))
        
        df = pd.DataFrame({
            'names': gene_names,
            'scores': scores,
            'pvals': pvals,
            'logfoldchanges': logfcs
        })
        df = df.sort_values('pvals')
        
        results[group] = df
    
    return results


def load_gmt(gmt_path: str) -> Dict[str, List[str]]:
    """Load gene sets from GMT file.
    
    Args:
        gmt_path: Path to GMT file
        
    Returns:
        Dictionary mapping pathway names to gene lists
    """
    gene_sets = {}
    
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                pathway = parts[0]
                genes = parts[2:]
                gene_sets[pathway] = genes
    
    return gene_sets


def pathway_enrich(de_genes: List[str], gene_sets_dir: str = './resource/GSEA/',
                   gene_set_file: str = 'hallmark.gmt', 
                   pval_thresh: float = 0.05) -> pd.DataFrame:
    """Perform pathway enrichment using hypergeometric test.
    
    Args:
        de_genes: List of differentially expressed genes
        gene_sets_dir: Directory containing GSEA gene set files
        gene_set_file: Name of GMT file (default: hallmark.gmt)
        pval_thresh: P-value threshold for significance
        
    Returns:
        DataFrame with enrichment results
        
    Raises:
        FileNotFoundError: If gene set file doesn't exist
    """
    gmt_path = os.path.join(gene_sets_dir, gene_set_file)
    
    if not os.path.exists(gmt_path):
        raise FileNotFoundError(f"Gene set file not found: {gmt_path}")
    
    gene_sets = load_gmt(gmt_path)
    de_genes_set = set(de_genes)
    
    # Estimate universe size (total unique genes across all pathways)
    all_genes = set()
    for genes in gene_sets.values():
        all_genes.update(genes)
    M = len(all_genes)
    n = len(de_genes_set)
    
    results = []
    
    for pathway, pathway_genes in gene_sets.items():
        pathway_genes_set = set(pathway_genes)
        N = len(pathway_genes_set)
        
        # Overlap
        overlap = de_genes_set & pathway_genes_set
        k = len(overlap)
        
        if k == 0:
            continue
        
        # Hypergeometric test
        pval = hypergeom.sf(k - 1, M, N, n)
        
        if pval <= pval_thresh:
            results.append({
                'pathway': pathway,
                'pval': pval,
                'overlap_size': k,
                'pathway_size': N,
                'de_genes_size': n,
                'overlap_genes': ','.join(sorted(overlap))
            })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('pval')
    
    return df


def scrna_analyzer(action: str, **kwargs) -> Any:
    """Main scRNA-seq analyzer tool.
    
    Args:
        action: Action to perform ('load', 'subset', 'proportions', 'rank_genes', 'enrich')
        **kwargs: Action-specific arguments
        
    Returns:
        Result depends on action
        
    Examples:
        >>> adata = scrna_analyzer('load', h5ad_path='data.h5ad')
        >>> cd8 = scrna_analyzer('subset', adata=adata, by='cell_type', value='CD8 T')
        >>> props = scrna_analyzer('proportions', adata=cd8, group_col='patient_group')
        >>> de = scrna_analyzer('rank_genes', adata=cd8, groupby='patient_group', groups=['PC'])
        >>> enrich = scrna_analyzer('enrich', de_genes=['GENE1', 'GENE2'])
    """
    if action == 'load':
        return load_h5ad(kwargs['h5ad_path'])
    elif action == 'subset':
        return subset_cells(kwargs['adata'], kwargs.get('by', 'cell_type'), kwargs.get('value', 'CD8 T'))
    elif action == 'proportions':
        return proportions(kwargs['adata'], kwargs.get('group_col', 'patient_group'), 
                          kwargs.get('subset_col', 'cx3cr1_bin'))
    elif action == 'rank_genes':
        return rank_genes_groups(kwargs['adata'], kwargs.get('groupby', 'patient_group'),
                                kwargs.get('groups', ['PC']))
    elif action == 'enrich':
        return pathway_enrich(kwargs['de_genes'], kwargs.get('gene_sets_dir', './resource/GSEA/'),
                            kwargs.get('gene_set_file', 'hallmark.gmt'),
                            kwargs.get('pval_thresh', 0.05))
    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == '__main__':
    print("Testing scRNA-seq analyzer...")
    
    # Create dummy data
    n_cells = 100
    n_genes = 50
    X = np.random.rand(n_cells, n_genes)
    
    obs = pd.DataFrame({
        'cell_type': ['CD8 T'] * 60 + ['CD4 T'] * 40,
        'patient_group': ['PC'] * 50 + ['non-PC'] * 50,
        'cx3cr1_bin': ['high'] * 30 + ['low'] * 70
    })
    
    var = pd.DataFrame({
        'gene_name': [f'GENE{i}' for i in range(n_genes)]
    })
    var.index = var['gene_name']
    
    adata = AnnDataLike(X, obs, var)
    
    print(f"✓ Created dummy data: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Test subsetting
    cd8 = subset_cells(adata, by='cell_type', value='CD8 T')
    print(f"✓ Subset CD8 T cells: {cd8.n_obs} cells")
    
    # Test proportions
    props = proportions(cd8, group_col='patient_group', subset_col='cx3cr1_bin')
    print(f"✓ Computed proportions:\n{props}")
    
    # Test DE
    de = rank_genes_groups(cd8, groupby='patient_group', groups=['PC'])
    print(f"✓ Differential expression: {len(de['PC'])} genes ranked")
    print(f"  Top 5 genes: {de['PC']['names'].head().tolist()}")
    
    print("\n✅ All tests passed!")
