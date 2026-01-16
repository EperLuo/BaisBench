from typing import List, Optional
import os

try:
    from smolagents import tool
except ImportError:
    def tool(func):
        return func

@tool
def scrna_h5ad_analyzer(
    file_path: str,
    operation: str,
    condition_col: Optional[str] = "condition",
    cell_type_col: Optional[str] = "cell_type",
    cluster: Optional[str] = None,
    cell_type: Optional[str] = None,
    genes: Optional[List[str]] = None,
    top_n: int = 10
) -> str:
    """
    Load and analyze h5ad files for single-cell RNA-seq data analysis.
    
    Args:
        file_path: Path to the h5ad file
        operation: One of: get_major_cell_types, compare_proportions, get_markers, subset_expression, compute_stats, get_metadata
        condition_col: Column name for conditions (default: condition)
        cell_type_col: Column name for cell types (default: cell_type)
        cluster: Cluster name for marker extraction
        cell_type: Cell type for subsetting
        genes: List of genes to query
        top_n: Number of top results (default: 10)
    
    Returns:
        str: Analysis results
    """
    
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found"
    
    valid_ops = ['get_major_cell_types', 'compare_proportions', 'get_markers', 'subset_expression', 'compute_stats', 'get_metadata']
    if operation not in valid_ops:
        return f"Error: Invalid operation. Valid: {', '.join(valid_ops)}"
    
    try:
        import scanpy as sc
        import numpy as np
        import pandas as pd
        
        adata = sc.read_h5ad(file_path)
        
        if operation == 'get_major_cell_types':
            if cell_type_col not in adata.obs.columns:
                return f"Error: {cell_type_col} not found"
            counts = adata.obs[cell_type_col].value_counts()
            result = f"Major Cell Types (Top {top_n}):\nTotal cells: {adata.n_obs:,}\n\n"
            for i, (ct, count) in enumerate(counts.head(top_n).items(), 1):
                pct = (count / adata.n_obs) * 100
                result += f"{i}. {ct}: {count:,} ({pct:.2f}%)\n"
            return result
        
        elif operation == 'compare_proportions':
            if condition_col not in adata.obs.columns or cell_type_col not in adata.obs.columns:
                return f"Error: Missing columns"
            crosstab = pd.crosstab(adata.obs[condition_col], adata.obs[cell_type_col])
            proportions = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
            result = "Cell Type Proportions:\n\nRaw Counts:\n" + crosstab.to_string()
            result += "\n\nProportions (%):\n" + proportions.round(2).to_string()
            return result
        
        elif operation == 'get_markers':
            if not cluster:
                return "Error: cluster parameter required"
            if cell_type_col not in adata.obs.columns:
                return f"Error: {cell_type_col} not found"
            mask = adata.obs[cell_type_col] == cluster
            if mask.sum() == 0:
                return f"Error: Cluster {cluster} not found"
            cluster_mean = np.array(adata[mask].X.mean(axis=0)).flatten() if hasattr(adata.X, 'toarray') else adata[mask].X.mean(axis=0)
            other_mean = np.array(adata[~mask].X.mean(axis=0)).flatten() if hasattr(adata.X, 'toarray') else adata[~mask].X.mean(axis=0)
            fc = np.log2((cluster_mean + 1) / (other_mean + 1))
            top_idx = np.argsort(fc)[::-1][:top_n]
            result = f"Marker Genes for {cluster}:\n"
            for i, idx in enumerate(top_idx, 1):
                result += f"{i}. {adata.var_names[idx]} (log2FC: {fc[idx]:.3f})\n"
            return result
        
        elif operation == 'subset_expression':
            if not cell_type or not genes:
                return "Error: cell_type and genes required"
            mask = adata.obs[cell_type_col] == cell_type
            if mask.sum() == 0:
                return f"Error: {cell_type} not found"
            result = f"Expression in {cell_type} ({mask.sum()} cells):\n\n"
            for gene in genes:
                if gene not in adata.var_names:
                    result += f"{gene}: Not found\n"
                    continue
                idx = list(adata.var_names).index(gene)
                expr = adata[mask].X[:, idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata[mask].X[:, idx]
                result += f"{gene}:\n  Mean: {np.mean(expr):.4f}\n  Median: {np.median(expr):.4f}\n  % expressing: {(expr > 0).sum()/len(expr)*100:.2f}%\n\n"
            return result
        
        elif operation == 'compute_stats':
            result = f"Dataset Statistics:\n\nCells: {adata.n_obs:,}\nGenes: {adata.n_vars:,}\n\n"
            if cell_type_col in adata.obs.columns:
                result += f"Cell types: {adata.obs[cell_type_col].nunique()}\n"
            if condition_col in adata.obs.columns:
                result += "\nConditions:\n"
                for cond, count in adata.obs[condition_col].value_counts().items():
                    result += f"  {cond}: {count:,}\n"
            return result
        
        elif operation == 'get_metadata':
            result = f"Metadata:\n\nDimensions: {adata.n_obs} cells × {adata.n_vars} genes\n\nObs columns:\n"
            for col in adata.obs.columns:
                result += f"  - {col}\n"
            return result
    
    except ImportError:
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                return f"H5AD file structure (h5py fallback):\nKeys: {list(f.keys())}\nInstall scanpy for full functionality"
        except Exception as e:
            return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
