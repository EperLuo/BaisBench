"""H5AD File Validator Tool

Inspects H5AD file structure and provides detailed metadata analysis.
"""

import json
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import warnings
from smolagents import tool


@tool
def h5ad_validator(file_path: str, sample_size: int = 10) -> str:
    """Inspect H5AD file structure and generate a comprehensive JSON report.
    
    This tool analyzes H5AD files (AnnData format) to extract metadata, dimensions,
    sparsity information, and provides recommendations for common biological terms.
    
    Args:
        file_path: Path to the H5AD file to validate
        sample_size: Number of sample entries to extract from var names (default: 10)
    
    Returns:
        JSON string containing comprehensive file analysis including:
        - File metadata (dimensions, sparsity)
        - obs keys and sample values
        - var keys and sample gene names
        - Categorical metadata (stages, cell types, etc.)
        - Recommendations for common biological terms
        - Data structure information
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return json.dumps({
                "error": f"File not found: {file_path}",
                "status": "failed"
            }, indent=2)
        
        report = {
            "file_path": str(file_path),
            "status": "success",
            "structure": {},
            "obs_metadata": {},
            "var_metadata": {},
            "data_info": {},
            "categorical_analysis": {},
            "recommendations": {}
        }
        
        with h5py.File(file_path, 'r') as f:
            report["structure"]["groups"] = list(f.keys())
            
            if 'X' in f:
                report["data_info"]["X"] = _analyze_X_matrix(f['X'])
            
            if 'obs' in f:
                report["obs_metadata"] = _analyze_obs(f['obs'], sample_size)
            
            if 'var' in f:
                report["var_metadata"] = _analyze_var(f['var'], sample_size)
            
            report["categorical_analysis"] = _extract_categorical_info(f)
            
            report["recommendations"] = _generate_recommendations(
                report["obs_metadata"],
                report["var_metadata"],
                report["categorical_analysis"]
            )
        
        return json.dumps(report, indent=2, default=str)
    
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "failed"
        }, indent=2)


def _decode_bytes(data: Any) -> Any:
    """Decode bytes objects to strings, handle arrays."""
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='ignore')
    elif isinstance(data, np.ndarray):
        if data.dtype.kind == 'S' or data.dtype.kind == 'O':
            try:
                return [_decode_bytes(item) for item in data]
            except:
                return data.tolist()
        return data.tolist()
    return data


def _analyze_X_matrix(X_group: h5py.Group) -> Dict[str, Any]:
    """Analyze the X matrix for shape, sparsity, and data type."""
    info = {}
    
    try:
        if 'data' in X_group and 'indices' in X_group and 'indptr' in X_group:
            info["format"] = "sparse_csr"
            info["shape"] = list(X_group.attrs.get('shape', [None, None]))
            
            data = X_group['data'][:]
            info["dtype"] = str(data.dtype)
            info["non_zero_elements"] = len(data)
            
            if info["shape"][0] and info["shape"][1]:
                total_elements = info["shape"][0] * info["shape"][1]
                info["sparsity"] = round(1 - (len(data) / total_elements), 4)
                info["density"] = round(len(data) / total_elements, 4)
        else:
            info["format"] = "dense"
            if hasattr(X_group, 'shape'):
                info["shape"] = list(X_group.shape)
                info["dtype"] = str(X_group.dtype)
                info["sparsity"] = 0.0
                info["density"] = 1.0
    except Exception as e:
        info["error"] = str(e)
    
    return info


def _analyze_obs(obs_group: h5py.Group, sample_size: int) -> Dict[str, Any]:
    """Analyze obs (observations/cells) metadata."""
    metadata = {
        "keys": [],
        "sample_data": {},
        "dimensions": {}
    }
    
    try:
        metadata["keys"] = list(obs_group.keys())
        
        for key in metadata["keys"]:
            try:
                data = obs_group[key]
                if hasattr(data, 'shape'):
                    metadata["dimensions"][key] = list(data.shape)
                    
                    if len(data) > 0:
                        sample_indices = min(sample_size, len(data))
                        sample = data[:sample_indices]
                        metadata["sample_data"][key] = _decode_bytes(sample)
                        
                        if 'categories' in data.attrs:
                            categories = data.attrs['categories']
                            metadata["sample_data"][f"{key}_categories"] = _decode_bytes(categories)
            except Exception as e:
                metadata["sample_data"][key] = f"Error: {str(e)}"
        
        if '_index' in obs_group:
            try:
                index_data = obs_group['_index'][:sample_size]
                metadata["sample_data"]["_index"] = _decode_bytes(index_data)
            except:
                pass
    
    except Exception as e:
        metadata["error"] = str(e)
    
    return metadata


def _analyze_var(var_group: h5py.Group, sample_size: int) -> Dict[str, Any]:
    """Analyze var (variables/genes) metadata."""
    metadata = {
        "keys": [],
        "sample_data": {},
        "dimensions": {},
        "sample_gene_names": []
    }
    
    try:
        metadata["keys"] = list(var_group.keys())
        
        for key in metadata["keys"]:
            try:
                data = var_group[key]
                if hasattr(data, 'shape'):
                    metadata["dimensions"][key] = list(data.shape)
                    
                    if len(data) > 0:
                        sample_indices = min(sample_size, len(data))
                        sample = data[:sample_indices]
                        metadata["sample_data"][key] = _decode_bytes(sample)
            except Exception as e:
                metadata["sample_data"][key] = f"Error: {str(e)}"
        
        if '_index' in var_group:
            try:
                index_data = var_group['_index'][:sample_size]
                metadata["sample_gene_names"] = _decode_bytes(index_data)
            except:
                pass
    
    except Exception as e:
        metadata["error"] = str(e)
    
    return metadata


def _extract_categorical_info(h5_file: h5py.File) -> Dict[str, Any]:
    """Extract and decode categorical metadata from obs."""
    categorical_data = {
        "cell_types": [],
        "stages": [],
        "clusters": [],
        "other_categories": {}
    }
    
    try:
        if 'obs' in h5_file:
            obs = h5_file['obs']
            
            for key in obs.keys():
                try:
                    data = obs[key]
                    
                    if 'categories' in data.attrs:
                        categories = _decode_bytes(data.attrs['categories'])
                        
                        if hasattr(data, '__iter__'):
                            codes = data[:]
                            unique_codes = np.unique(codes)
                            
                            if isinstance(categories, list) and len(categories) > 0:
                                unique_values = [categories[int(c)] if int(c) < len(categories) else f"code_{c}" 
                                               for c in unique_codes if int(c) >= 0]
                            else:
                                unique_values = categories
                            
                            key_lower = key.lower()
                            if 'cell' in key_lower and 'type' in key_lower:
                                categorical_data["cell_types"] = unique_values
                            elif 'stage' in key_lower or 'time' in key_lower:
                                categorical_data["stages"] = unique_values
                            elif 'cluster' in key_lower or 'louvain' in key_lower or 'leiden' in key_lower:
                                categorical_data["clusters"] = unique_values
                            else:
                                categorical_data["other_categories"][key] = unique_values
                except Exception as e:
                    continue
    
    except Exception as e:
        categorical_data["error"] = str(e)
    
    return categorical_data


def _generate_recommendations(obs_metadata: Dict, var_metadata: Dict, 
                             categorical_analysis: Dict) -> Dict[str, Any]:
    """Generate recommendations for common biological terms and mappings."""
    recommendations = {
        "cell_type_mappings": {},
        "gene_mappings": {},
        "analysis_suggestions": []
    }
    
    cell_type_terms = ['endothelial', 'epithelial', 'fibroblast', 'immune', 
                       'lymphocyte', 'macrophage', 'neuron', 'stem']
    
    if categorical_analysis.get("cell_types"):
        for term in cell_type_terms:
            matches = [ct for ct in categorical_analysis["cell_types"] 
                      if term.lower() in str(ct).lower()]
            if matches:
                recommendations["cell_type_mappings"][term] = matches
    
    common_genes = ['VEGFA', 'PECAM1', 'CD31', 'VWF', 'ACTB', 'GAPDH', 
                    'TP53', 'MYC', 'CD34', 'KDR']
    
    if var_metadata.get("sample_gene_names"):
        gene_names = var_metadata["sample_gene_names"]
        for gene in common_genes:
            matches = [g for g in gene_names if gene.upper() in str(g).upper()]
            if matches:
                recommendations["gene_mappings"][gene] = matches
    
    if obs_metadata.get("keys"):
        if any('cluster' in k.lower() for k in obs_metadata["keys"]):
            recommendations["analysis_suggestions"].append(
                "Clustering information detected - suitable for cluster-based analysis"
            )
        if any('stage' in k.lower() or 'time' in k.lower() for k in obs_metadata["keys"]):
            recommendations["analysis_suggestions"].append(
                "Temporal/stage information detected - suitable for trajectory analysis"
            )
    
    if categorical_analysis.get("cell_types"):
        recommendations["analysis_suggestions"].append(
            f"Found {len(categorical_analysis['cell_types'])} cell types - suitable for cell type-specific analysis"
        )
    
    return recommendations
