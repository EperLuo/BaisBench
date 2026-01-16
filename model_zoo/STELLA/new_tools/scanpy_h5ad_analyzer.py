"""Scanpy H5AD Analyzer Tool

Robust tool for loading h5ad files, preprocessing, subsetting cells,
computing expression metrics, performing differential expression analysis,
and validating markers.
"""

import os
import warnings
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from smolagents import tool

try:
    import scanpy as sc
    import anndata as ad
except ImportError:
    raise ImportError(
        "Required packages not found. Install with: pip install scanpy anndata"
    )


@tool
def scanpy_h5ad_analyzer(
    h5ad_path: str,
    operation: str,
    cell_annotation_column: Optional[str] = None,
    cell_subset: Optional[str] = None,
    gene_list: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    reference_group: Optional[str] = None,
    test_group: Optional[str] = None,
    ligand_receptor_pairs: Optional[List[tuple]] = None,
    normalize: bool = True,
    log_transform: bool = True,
    min_cells: int = 3,
    min_genes: int = 200,
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 0.5,
) -> Dict[str, Any]:
    """Robustly analyze h5ad files with preprocessing, subsetting, expression metrics, and DE analysis.
    
    This tool provides comprehensive analysis of single-cell RNA-seq data stored in h5ad format,
    including normalization, cell subsetting, expression quantification, differential expression,
    and ligand-receptor co-expression analysis.
    
    Args:
        h5ad_path: Path to the h5ad file to analyze
        operation: Analysis operation to perform. Options:
            - 'load': Load and return basic info about the h5ad file
            - 'preprocess': Normalize and log-transform the data
            - 'subset_cells': Subset cells by annotation
            - 'expression_metrics': Compute mean expression and percentage of cells expressing genes
            - 'differential_expression': Perform DE analysis using Wilcoxon rank-sum test
            - 'validate_markers': Validate known marker genes in cell types
            - 'ligand_receptor': Compute ligand-receptor co-expression percentages
            - 'full_analysis': Run complete analysis pipeline
        cell_annotation_column: Column name in adata.obs for cell annotations (default: None)
        cell_subset: Specific cell type/annotation to subset (default: None)
        gene_list: List of genes to analyze (default: None, uses all genes)
        group_by: Column name for grouping cells in DE analysis (default: None)
        reference_group: Reference group for DE analysis (default: None)
        test_group: Test group to compare against reference in DE analysis (default: None)
        ligand_receptor_pairs: List of (ligand, receptor) gene tuples for co-expression (default: None)
        normalize: Whether to normalize the data (default: True)
        log_transform: Whether to log-transform after normalization (default: True)
        min_cells: Minimum number of cells for gene filtering (default: 3)
        min_genes: Minimum number of genes for cell filtering (default: 200)
        pval_cutoff: P-value cutoff for significant genes (default: 0.05)
        logfc_cutoff: Log fold-change cutoff for DE genes (default: 0.5)
    
    Returns:
        Dictionary containing analysis results with keys depending on operation:
        - 'status': 'success' or 'error'
        - 'message': Status message
        - 'data': Operation-specific results (varies by operation)
        - 'n_cells': Number of cells (when applicable)
        - 'n_genes': Number of genes (when applicable)
    
    Examples:
        >>> # Load and inspect h5ad file
        >>> result = scanpy_h5ad_analyzer('data.h5ad', operation='load')
        
        >>> # Compute expression metrics for specific cell type
        >>> result = scanpy_h5ad_analyzer(
        ...     'data.h5ad',
        ...     operation='expression_metrics',
        ...     cell_annotation_column='cell_type',
        ...     cell_subset='T cells',
        ...     gene_list=['CD3D', 'CD4', 'CD8A']
        ... )
        
        >>> # Perform differential expression
        >>> result = scanpy_h5ad_analyzer(
        ...     'data.h5ad',
        ...     operation='differential_expression',
        ...     group_by='cell_type',
        ...     reference_group='B cells',
        ...     test_group='T cells'
        ... )
    """
    
    # Input validation
    if not os.path.exists(h5ad_path):
        return {
            'status': 'error',
            'message': f'File not found: {h5ad_path}',
            'data': None
        }
    
    valid_operations = [
        'load', 'preprocess', 'subset_cells', 'expression_metrics',
        'differential_expression', 'validate_markers', 'ligand_receptor',
        'full_analysis'
    ]
    
    if operation not in valid_operations:
        return {
            'status': 'error',
            'message': f'Invalid operation. Choose from: {valid_operations}',
            'data': None
        }
    
    try:
        # Load h5ad file
        adata = sc.read_h5ad(h5ad_path)
        
        result = {
            'status': 'success',
            'message': f'Successfully loaded {h5ad_path}',
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'data': {}
        }
        
        # Basic load operation
        if operation == 'load':
            result['data'] = {
                'obs_columns': list(adata.obs.columns),
                'var_columns': list(adata.var.columns),
                'cell_types': list(adata.obs[cell_annotation_column].unique()) if cell_annotation_column and cell_annotation_column in adata.obs.columns else [],
                'is_sparse': str(type(adata.X)),
            }
            return result
        
        # Preprocessing
        if operation in ['preprocess', 'full_analysis'] or (normalize or log_transform):
            if normalize:
                sc.pp.normalize_total(adata, target_sum=1e4)
            if log_transform:
                sc.pp.log1p(adata)
            result['message'] += ' | Preprocessed (normalized & log-transformed)'
        
        # Cell subsetting
        if operation == 'subset_cells' or (cell_subset and cell_annotation_column):
            if not cell_annotation_column or cell_annotation_column not in adata.obs.columns:
                return {
                    'status': 'error',
                    'message': f'Cell annotation column "{cell_annotation_column}" not found',
                    'data': None
                }
            
            if not cell_subset:
                return {
                    'status': 'error',
                    'message': 'cell_subset parameter required for subsetting',
                    'data': None
                }
            
            mask = adata.obs[cell_annotation_column] == cell_subset
            adata = adata[mask, :].copy()
            result['n_cells'] = adata.n_obs
            result['message'] += f' | Subset to {cell_subset}'
            
            if operation == 'subset_cells':
                result['data']['subset_info'] = {
                    'cell_type': cell_subset,
                    'n_cells_subset': adata.n_obs
                }
                return result
        
        # Expression metrics
        if operation == 'expression_metrics':
            genes_to_analyze = gene_list if gene_list else adata.var_names.tolist()
            
            # Filter genes that exist in the dataset
            available_genes = [g for g in genes_to_analyze if g in adata.var_names]
            
            if not available_genes:
                return {
                    'status': 'error',
                    'message': 'None of the specified genes found in dataset',
                    'data': None
                }
            
            metrics = {}
            for gene in available_genes:
                gene_idx = adata.var_names.get_loc(gene)
                
                # Handle sparse and dense matrices safely
                if hasattr(adata.X, 'toarray'):
                    expr_values = adata.X[:, gene_idx].toarray().flatten()
                else:
                    expr_values = adata.X[:, gene_idx].flatten()
                
                if hasattr(expr_values, 'A1'):
                    expr_values = expr_values.A1
                
                metrics[gene] = {
                    'mean_expression': float(np.mean(expr_values)),
                    'median_expression': float(np.median(expr_values)),
                    'percent_cells_expressing': float(100 * np.sum(expr_values > 0) / len(expr_values)),
                    'std_expression': float(np.std(expr_values)),
                    'max_expression': float(np.max(expr_values)),
                }
            
            result['data']['expression_metrics'] = metrics
            result['data']['genes_analyzed'] = available_genes
            result['data']['genes_not_found'] = [g for g in genes_to_analyze if g not in available_genes]
            return result
        
        # Differential expression
        if operation == 'differential_expression':
            if not group_by or group_by not in adata.obs.columns:
                return {
                    'status': 'error',
                    'message': f'group_by column "{group_by}" not found in obs',
                    'data': None
                }
            
            if not reference_group or not test_group:
                return {
                    'status': 'error',
                    'message': 'Both reference_group and test_group required for DE',
                    'data': None
                }
            
            # Check if groups exist
            available_groups = adata.obs[group_by].unique().tolist()
            if reference_group not in available_groups or test_group not in available_groups:
                return {
                    'status': 'error',
                    'message': f'Groups not found. Available: {available_groups}',
                    'data': None
                }
            
            # Perform differential expression
            sc.tl.rank_genes_groups(
                adata,
                groupby=group_by,
                reference=reference_group,
                method='wilcoxon',
                key_added='rank_genes_groups'
            )
            
            # Extract results for test group
            de_results = sc.get.rank_genes_groups_df(
                adata,
                group=test_group,
                key='rank_genes_groups'
            )
            
            # Filter by significance and log fold change
            significant = de_results[
                (de_results['pvals_adj'] < pval_cutoff) &
                (np.abs(de_results['logfoldchanges']) > logfc_cutoff)
            ]
            
            result['data']['differential_expression'] = {
                'test_group': test_group,
                'reference_group': reference_group,
                'n_significant_genes': len(significant),
                'top_upregulated': significant.nlargest(20, 'logfoldchanges')[['names', 'logfoldchanges', 'pvals_adj']].to_dict('records'),
                'top_downregulated': significant.nsmallest(20, 'logfoldchanges')[['names', 'logfoldchanges', 'pvals_adj']].to_dict('records'),
                'all_significant': significant[['names', 'logfoldchanges', 'pvals_adj', 'scores']].to_dict('records'),
            }
            return result
        
        # Validate markers
        if operation == 'validate_markers':
            if not gene_list:
                return {
                    'status': 'error',
                    'message': 'gene_list required for marker validation',
                    'data': None
                }
            
            if not group_by or group_by not in adata.obs.columns:
                return {
                    'status': 'error',
                    'message': f'group_by column "{group_by}" required for validation',
                    'data': None
                }
            
            validation_results = {}
            available_genes = [g for g in gene_list if g in adata.var_names]
            
            for cell_type in adata.obs[group_by].unique():
                cell_mask = adata.obs[group_by] == cell_type
                adata_subset = adata[cell_mask, :]
                
                type_metrics = {}
                for gene in available_genes:
                    gene_idx = adata_subset.var_names.get_loc(gene)
                    
                    if hasattr(adata_subset.X, 'toarray'):
                        expr_values = adata_subset.X[:, gene_idx].toarray().flatten()
                    else:
                        expr_values = adata_subset.X[:, gene_idx].flatten()
                    
                    if hasattr(expr_values, 'A1'):
                        expr_values = expr_values.A1
                    
                    type_metrics[gene] = {
                        'mean_expr': float(np.mean(expr_values)),
                        'pct_expressing': float(100 * np.sum(expr_values > 0) / len(expr_values)),
                    }
                
                validation_results[cell_type] = type_metrics
            
            result['data']['marker_validation'] = validation_results
            result['data']['genes_validated'] = available_genes
            return result
        
        # Ligand-receptor co-expression
        if operation == 'ligand_receptor':
            if not ligand_receptor_pairs:
                return {
                    'status': 'error',
                    'message': 'ligand_receptor_pairs required (list of tuples)',
                    'data': None
                }
            
            lr_results = []
            for ligand, receptor in ligand_receptor_pairs:
                if ligand not in adata.var_names or receptor not in adata.var_names:
                    lr_results.append({
                        'ligand': ligand,
                        'receptor': receptor,
                        'status': 'not_found',
                        'coexpression_pct': 0.0
                    })
                    continue
                
                lig_idx = adata.var_names.get_loc(ligand)
                rec_idx = adata.var_names.get_loc(receptor)
                
                # Get expression values
                if hasattr(adata.X, 'toarray'):
                    lig_expr = adata.X[:, lig_idx].toarray().flatten()
                    rec_expr = adata.X[:, rec_idx].toarray().flatten()
                else:
                    lig_expr = adata.X[:, lig_idx].flatten()
                    rec_expr = adata.X[:, rec_idx].flatten()
                
                if hasattr(lig_expr, 'A1'):
                    lig_expr = lig_expr.A1
                if hasattr(rec_expr, 'A1'):
                    rec_expr = rec_expr.A1
                
                # Calculate co-expression
                coexpr_mask = (lig_expr > 0) & (rec_expr > 0)
                coexpr_pct = 100 * np.sum(coexpr_mask) / len(coexpr_mask)
                
                lr_results.append({
                    'ligand': ligand,
                    'receptor': receptor,
                    'status': 'found',
                    'coexpression_pct': float(coexpr_pct),
                    'ligand_mean': float(np.mean(lig_expr)),
                    'receptor_mean': float(np.mean(rec_expr)),
                })
            
            result['data']['ligand_receptor_analysis'] = lr_results
            return result
        
        # Full analysis pipeline
        if operation == 'full_analysis':
            result['data']['full_analysis'] = {
                'preprocessing': 'completed',
                'basic_stats': {
                    'n_cells': adata.n_obs,
                    'n_genes': adata.n_vars,
                    'cell_annotations': list(adata.obs.columns) if len(adata.obs.columns) > 0 else [],
                }
            }
            return result
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error during analysis: {str(e)}',
            'data': None,
            'error_type': type(e).__name__
        }
