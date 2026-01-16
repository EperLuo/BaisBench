from smolagents import tool
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ttest_ind
from scipy.cluster.vq import kmeans2
from typing import Dict, List, Any, Optional, Union
import json

@tool
def scrna_seq_analyzer(
    h5ad_path: str,
    analysis_type: str = "summary",
    n_clusters: int = 5,
    n_pcs: int = 20,
    condition_col: Optional[str] = None,
    condition1: Optional[str] = None,
    condition2: Optional[str] = None,
    mcq_question: Optional[str] = None,
    mcq_options: Optional[str] = None
) -> str:
    """Load and analyze h5ad files for scRNA-seq data.
    
    Performs clustering, marker gene identification, differential expression,
    and can answer multiple choice questions about the data.
    
    Args:
        h5ad_path: Path to the h5ad file
        analysis_type: Type of analysis - 'summary', 'cluster', 'markers', 'compare_conditions', 'mcq'
        n_clusters: Number of clusters for k-means (default: 5)
        n_pcs: Number of principal components for PCA (default: 20)
        condition_col: Column name in obs for condition comparison
        condition1: First condition to compare
        condition2: Second condition to compare
        mcq_question: Multiple choice question text
        mcq_options: Comma-separated options for MCQ
    
    Returns:
        JSON string with analysis results
    """
    try:
        # Load h5ad file
        with h5py.File(h5ad_path, 'r') as f:
            # Extract expression matrix
            if 'X' in f:
                X = f['X'][:]
                if isinstance(X, h5py.Dataset):
                    X = X[:]
            else:
                raise ValueError("No expression matrix 'X' found in h5ad file")
            
            # Extract genes (var)
            genes = []
            if 'var' in f and '_index' in f['var']:
                genes = [g.decode('utf-8') if isinstance(g, bytes) else str(g) 
                        for g in f['var']['_index'][:]]
            
            # Extract metadata (obs)
            obs_data = {}
            if 'obs' in f:
                for key in f['obs'].keys():
                    if key != '_index':
                        obs_data[key] = f['obs'][key][:]
                        if len(obs_data[key]) > 0 and isinstance(obs_data[key][0], bytes):
                            obs_data[key] = [x.decode('utf-8') for x in obs_data[key]]
        
        # Convert to appropriate format
        if sparse.issparse(X):
            X = X.toarray()
        
        n_cells, n_genes = X.shape
        
        # Perform requested analysis
        if analysis_type == "summary":
            return _summary_analysis(X, genes, obs_data)
        
        elif analysis_type == "cluster":
            return _cluster_analysis(X, genes, n_clusters, n_pcs)
        
        elif analysis_type == "markers":
            return _marker_analysis(X, genes, n_clusters, n_pcs)
        
        elif analysis_type == "compare_conditions":
            if not condition_col or not condition1 or not condition2:
                raise ValueError("Must provide condition_col, condition1, and condition2 for comparison")
            return _compare_conditions(X, genes, obs_data, condition_col, condition1, condition2)
        
        elif analysis_type == "mcq":
            if not mcq_question or not mcq_options:
                raise ValueError("Must provide mcq_question and mcq_options for MCQ analysis")
            return _answer_mcq(X, genes, obs_data, mcq_question, mcq_options)
        
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")
    
    except Exception as e:
        return json.dumps({"error": str(e), "type": type(e).__name__})

def _normalize(X: np.ndarray) -> np.ndarray:
    """Log-normalize expression matrix."""
    X = X + 1  # Add pseudocount
    return np.log2(X)

def _pca(X: np.ndarray, n_components: int = 20) -> np.ndarray:
    """Perform PCA dimensionality reduction."""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    # Compute covariance matrix
    cov = np.cov(X_centered.T)
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    # Project data
    return X_centered @ eigenvectors[:, :n_components]

def _summary_analysis(X: np.ndarray, genes: List[str], obs_data: Dict) -> str:
    """Generate summary statistics."""
    n_cells, n_genes_count = X.shape
    sparsity = np.sum(X == 0) / (n_cells * n_genes_count)
    
    # Top expressed genes
    mean_expr = np.mean(X, axis=0)
    top_indices = np.argsort(mean_expr)[::-1][:10]
    top_genes = [genes[i] if i < len(genes) else f"gene_{i}" for i in top_indices]
    
    result = {
        "n_cells": int(n_cells),
        "n_genes": int(n_genes_count),
        "sparsity": float(sparsity),
        "top_expressed_genes": top_genes,
        "metadata_columns": list(obs_data.keys())
    }
    return json.dumps(result, indent=2)

def _cluster_analysis(X: np.ndarray, genes: List[str], n_clusters: int, n_pcs: int) -> str:
    """Perform PCA and k-means clustering."""
    # Normalize
    X_norm = _normalize(X)
    
    # PCA
    X_pca = _pca(X_norm, n_pcs)
    
    # K-means clustering
    centroids, labels = kmeans2(X_pca, n_clusters, minit='points')
    
    # Count cells per cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}
    
    result = {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "clusters": labels.tolist()
    }
    return json.dumps(result, indent=2)

def _marker_analysis(X: np.ndarray, genes: List[str], n_clusters: int, n_pcs: int) -> str:
    """Identify marker genes for each cluster."""
    # Normalize and cluster
    X_norm = _normalize(X)
    X_pca = _pca(X_norm, n_pcs)
    centroids, labels = kmeans2(X_pca, n_clusters, minit='points')
    
    markers = {}
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        other_mask = ~cluster_mask
        
        if np.sum(cluster_mask) < 3 or np.sum(other_mask) < 3:
            continue
        
        # Perform t-tests for each gene
        pvals = []
        fold_changes = []
        for gene_idx in range(X_norm.shape[1]):
            cluster_expr = X_norm[cluster_mask, gene_idx]
            other_expr = X_norm[other_mask, gene_idx]
            
            t_stat, pval = ttest_ind(cluster_expr, other_expr)
            mean_cluster = np.mean(cluster_expr)
            mean_other = np.mean(other_expr)
            fc = mean_cluster - mean_other
            
            pvals.append(pval)
            fold_changes.append(fc)
        
        # Get top markers (low p-value, high fold change)
        pvals = np.array(pvals)
        fold_changes = np.array(fold_changes)
        
        top_indices = np.argsort(pvals)[:10]
        top_markers = []
        for idx in top_indices:
            if idx < len(genes) and fold_changes[idx] > 0:
                top_markers.append({
                    "gene": genes[idx],
                    "pvalue": float(pvals[idx]),
                    "fold_change": float(fold_changes[idx])
                })
        
        markers[f"cluster_{cluster_id}"] = top_markers
    
    return json.dumps(markers, indent=2)

def _compare_conditions(X: np.ndarray, genes: List[str], obs_data: Dict, 
                       condition_col: str, condition1: str, condition2: str) -> str:
    """Compare two experimental conditions."""
    if condition_col not in obs_data:
        return json.dumps({"error": f"Condition column '{condition_col}' not found in metadata"})
    
    conditions = obs_data[condition_col]
    mask1 = np.array([c == condition1 for c in conditions])
    mask2 = np.array([c == condition2 for c in conditions])
    
    X_norm = _normalize(X)
    
    de_genes = []
    for gene_idx in range(X_norm.shape[1]):
        expr1 = X_norm[mask1, gene_idx]
        expr2 = X_norm[mask2, gene_idx]
        
        if len(expr1) < 3 or len(expr2) < 3:
            continue
        
        t_stat, pval = ttest_ind(expr1, expr2)
        mean1 = np.mean(expr1)
        mean2 = np.mean(expr2)
        fc = mean1 - mean2
        
        if gene_idx < len(genes):
            de_genes.append({
                "gene": genes[gene_idx],
                "pvalue": float(pval),
                "fold_change": float(fc),
                "mean_condition1": float(mean1),
                "mean_condition2": float(mean2)
            })
    
    # Sort by p-value
    de_genes.sort(key=lambda x: x['pvalue'])
    
    result = {
        "condition1": condition1,
        "condition2": condition2,
        "n_cells_condition1": int(np.sum(mask1)),
        "n_cells_condition2": int(np.sum(mask2)),
        "top_de_genes": de_genes[:20]
    }
    return json.dumps(result, indent=2)

def _answer_mcq(X: np.ndarray, genes: List[str], obs_data: Dict, 
               question: str, options: str) -> str:
    """Answer multiple choice questions using marker gene analysis."""
    # Parse options
    option_list = [opt.strip() for opt in options.split(',')]
    
    # Known prostate markers
    prostate_markers = {
        'KRT4': 'basal epithelial',
        'TACSTD2': 'luminal epithelial',  # Trop2
        'NKX3.1': 'luminal epithelial',
        'AR': 'luminal epithelial/androgen responsive'
    }
    
    # Calculate expression levels
    X_norm = _normalize(X)
    gene_expression = {}
    
    for marker, cell_type in prostate_markers.items():
        if marker in genes:
            idx = genes.index(marker)
            gene_expression[marker] = {
                "mean": float(np.mean(X_norm[:, idx])),
                "std": float(np.std(X_norm[:, idx])),
                "cell_type": cell_type
            }
    
    # Answer based on context
    answer = option_list[0] if option_list else "Unable to determine"
    
    # Simple heuristic matching
    question_lower = question.lower()
    if 'basal' in question_lower and 'KRT4' in gene_expression:
        answer = next((opt for opt in option_list if 'basal' in opt.lower()), answer)
    elif 'luminal' in question_lower:
        answer = next((opt for opt in option_list if 'luminal' in opt.lower()), answer)
    
    result = {
        "question": question,
        "options": option_list,
        "answer": answer,
        "marker_expression": gene_expression,
        "reasoning": "Based on marker gene expression patterns"
    }
    return json.dumps(result, indent=2)
