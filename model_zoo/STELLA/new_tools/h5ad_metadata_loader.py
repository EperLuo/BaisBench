from smolagents import tool
import h5py
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import os


def _load_categorical_column(group, col_name: str) -> pd.Series:
    """Load a categorical column from h5ad file."""
    col_group = group[col_name]
    if 'categories' in col_group.attrs or 'categories' in col_group:
        # Load codes and categories
        codes = col_group['codes'][:] if 'codes' in col_group else col_group[:]
        if 'categories' in col_group:
            categories = col_group['categories'][:]
            if hasattr(categories[0], 'decode'):
                categories = [cat.decode('utf-8') for cat in categories]
        else:
            categories = col_group.attrs['categories']
        return pd.Series(pd.Categorical.from_codes(codes, categories))
    else:
        data = col_group[:]
        if hasattr(data[0], 'decode'):
            data = [d.decode('utf-8') for d in data]
        return pd.Series(data)


def _load_dataframe_from_group(h5_group) -> pd.DataFrame:
    """Recursively load a DataFrame from an HDF5 group."""
    data_dict = {}
    index_data = None
    
    # Load index if present
    if '_index' in h5_group:
        index_data = h5_group['_index'][:]
        if hasattr(index_data[0], 'decode'):
            index_data = [idx.decode('utf-8') for idx in index_data]
    
    # Load columns
    for col_name in h5_group.keys():
        if col_name.startswith('_'):
            continue
        
        try:
            col_data = h5_group[col_name]
            if isinstance(col_data, h5py.Group):
                # Handle categorical or complex columns
                data_dict[col_name] = _load_categorical_column(h5_group, col_name)
            else:
                # Handle simple arrays
                arr = col_data[:]
                if arr.dtype.kind in ['O', 'S', 'U']:
                    if hasattr(arr[0], 'decode'):
                        arr = [a.decode('utf-8') if a else '' for a in arr]
                data_dict[col_name] = arr
        except Exception as e:
            print(f"Warning: Could not load column {col_name}: {e}")
    
    df = pd.DataFrame(data_dict)
    if index_data is not None:
        df.index = index_data
    
    return df


@tool
def h5ad_metadata_loader(
    h5ad_path: str,
    action: str = "inspect",
    gene_list: Optional[str] = None,
    cluster_col: Optional[str] = None,
    cluster_name: Optional[str] = None,
    cell_type_col: Optional[str] = None,
    cell_type_pattern: Optional[str] = None,
    age_col: Optional[str] = None
) -> str:
    """Load and process h5ad file metadata without scanpy/anndata.
    
    This tool loads h5ad files using h5py and provides various operations:
    - Load observation (cell) metadata
    - Load variable (gene) metadata
    - Subset to specific cell types (e.g., fibroblasts)
    - Compute average gene expression in clusters
    - Extract age distributions
    
    Args:
        h5ad_path: Path to the h5ad file
        action: Action to perform - 'inspect', 'load_obs', 'load_var', 
                'get_cell_subset', 'avg_expression', 'get_ages'
        gene_list: Comma-separated gene names for expression calculation
        cluster_col: Column name for cluster information
        cluster_name: Specific cluster name to filter
        cell_type_col: Column name containing cell type information
        cell_type_pattern: Pattern to match for cell type (e.g., 'fibroblast')
        age_col: Column name containing age information
    
    Returns:
        String containing the results of the requested operation
    """
    
    if not os.path.exists(h5ad_path):
        return f"Error: File not found: {h5ad_path}"
    
    try:
        with h5py.File(h5ad_path, 'r') as f:
            
            if action == "inspect":
                # Inspect file structure
                result = ["H5AD File Structure:"]
                result.append(f"\nTop-level keys: {list(f.keys())}")
                
                if 'obs' in f:
                    obs_keys = list(f['obs'].keys())
                    result.append(f"\nObservation (obs) columns ({len(obs_keys)}): {obs_keys[:20]}")
                    if len(obs_keys) > 20:
                        result.append(f"... and {len(obs_keys) - 20} more")
                
                if 'var' in f:
                    var_keys = list(f['var'].keys())
                    result.append(f"\nVariable (var) columns ({len(var_keys)}): {var_keys[:20]}")
                    if len(var_keys) > 20:
                        result.append(f"... and {len(var_keys) - 20} more")
                
                if 'X' in f:
                    result.append(f"\nExpression matrix (X) shape: {f['X'].shape}")
                
                return "\n".join(result)
            
            elif action == "load_obs":
                # Load observation metadata
                if 'obs' not in f:
                    return "Error: 'obs' group not found in h5ad file"
                
                obs_df = _load_dataframe_from_group(f['obs'])
                result = [f"Loaded obs DataFrame with shape: {obs_df.shape}"]
                result.append(f"\nColumns: {list(obs_df.columns)}")
                result.append(f"\nFirst 5 rows:\n{obs_df.head().to_string()}")
                result.append(f"\nData types:\n{obs_df.dtypes.to_string()}")
                return "\n".join(result)
            
            elif action == "load_var":
                # Load variable metadata
                if 'var' not in f:
                    return "Error: 'var' group not found in h5ad file"
                
                var_df = _load_dataframe_from_group(f['var'])
                result = [f"Loaded var DataFrame with shape: {var_df.shape}"]
                result.append(f"\nColumns: {list(var_df.columns)}")
                result.append(f"\nFirst 10 rows:\n{var_df.head(10).to_string()}")
                result.append(f"\nData types:\n{var_df.dtypes.to_string()}")
                return "\n".join(result)
            
            elif action == "get_cell_subset":
                # Get subset of cells matching pattern
                if 'obs' not in f:
                    return "Error: 'obs' group not found in h5ad file"
                
                if not cell_type_col or not cell_type_pattern:
                    return "Error: Both cell_type_col and cell_type_pattern are required"
                
                obs_df = _load_dataframe_from_group(f['obs'])
                
                if cell_type_col not in obs_df.columns:
                    return f"Error: Column '{cell_type_col}' not found. Available: {list(obs_df.columns)}"
                
                # Case-insensitive pattern matching
                mask = obs_df[cell_type_col].astype(str).str.contains(cell_type_pattern, case=False, na=False)
                subset_df = obs_df[mask]
                
                result = [f"Found {len(subset_df)} cells matching '{cell_type_pattern}' in '{cell_type_col}'"]
                result.append(f"\nSubset shape: {subset_df.shape}")
                result.append(f"\nFirst 10 matching cells:\n{subset_df.head(10).to_string()}")
                
                if len(subset_df) > 0:
                    result.append(f"\nUnique values in {cell_type_col}:")
                    result.append(subset_df[cell_type_col].value_counts().to_string())
                
                return "\n".join(result)
            
            elif action == "avg_expression":
                # Compute average expression in cluster
                if not gene_list or not cluster_col or not cluster_name:
                    return "Error: gene_list, cluster_col, and cluster_name are required"
                
                if 'obs' not in f or 'var' not in f or 'X' not in f:
                    return "Error: Required groups not found in h5ad file"
                
                obs_df = _load_dataframe_from_group(f['obs'])
                var_df = _load_dataframe_from_group(f['var'])
                
                if cluster_col not in obs_df.columns:
                    return f"Error: Cluster column '{cluster_col}' not found. Available: {list(obs_df.columns)}"
                
                # Filter cells in cluster
                cluster_mask = obs_df[cluster_col].astype(str) == str(cluster_name)
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) == 0:
                    return f"Error: No cells found in cluster '{cluster_name}'"
                
                # Parse gene list
                genes = [g.strip() for g in gene_list.split(',')]
                
                # Find gene indices
                gene_indices = []
                gene_names_found = []
                for gene in genes:
                    if gene in var_df.index:
                        gene_indices.append(var_df.index.get_loc(gene))
                        gene_names_found.append(gene)
                
                if len(gene_indices) == 0:
                    return f"Error: None of the genes found in var index. Genes requested: {genes}"
                
                # Load expression data (handle sparse if needed)
                X_data = f['X']
                if hasattr(X_data, 'shape'):
                    # Dense matrix
                    expr_subset = X_data[cluster_indices, :][:, gene_indices]
                    avg_expr = np.mean(expr_subset, axis=0)
                else:
                    return "Error: Sparse matrix support not yet implemented"
                
                result = [f"Average expression in cluster '{cluster_name}' ({len(cluster_indices)} cells):"]
                for i, gene in enumerate(gene_names_found):
                    result.append(f"{gene}: {avg_expr[i]:.4f}")
                
                return "\n".join(result)
            
            elif action == "get_ages":
                # Extract age distribution
                if 'obs' not in f:
                    return "Error: 'obs' group not found in h5ad file"
                
                if not age_col:
                    return "Error: age_col parameter is required"
                
                obs_df = _load_dataframe_from_group(f['obs'])
                
                if age_col not in obs_df.columns:
                    return f"Error: Age column '{age_col}' not found. Available: {list(obs_df.columns)}"
                
                age_data = obs_df[age_col]
                result = [f"Age distribution from column '{age_col}':"]
                result.append(f"\nValue counts:\n{age_data.value_counts().sort_index().to_string()}")
                result.append(f"\nTotal cells: {len(age_data)}")
                result.append(f"\nMissing values: {age_data.isna().sum()}")
                
                # Try numeric statistics if applicable
                try:
                    numeric_ages = pd.to_numeric(age_data, errors='coerce')
                    if not numeric_ages.isna().all():
                        result.append(f"\nNumeric statistics:")
                        result.append(f"Mean: {numeric_ages.mean():.2f}")
                        result.append(f"Median: {numeric_ages.median():.2f}")
                        result.append(f"Min: {numeric_ages.min():.2f}")
                        result.append(f"Max: {numeric_ages.max():.2f}")
                except:
                    pass
                
                return "\n".join(result)
            
            else:
                return f"Error: Unknown action '{action}'. Valid actions: inspect, load_obs, load_var, get_cell_subset, avg_expression, get_ages"
    
    except Exception as e:
        return f"Error processing h5ad file: {str(e)}\n\nPlease check that the file is a valid h5ad format."
