"""H5AD Single Cell Analyzer Tool

Analyzes h5ad files without scanpy dependency using h5py for direct file access.
Provides metadata extraction, cell subsetting, differential expression, and pathway scoring.
"""

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Union, Tuple
import os
from smolagents import tool


# Predefined gene sets
GENE_SETS = {
    'angiogenic': ['VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR', 'FLT4', 'ANGPT1', 'ANGPT2', 'TEK', 'WNT2B'],
    'glycolysis': [
        'SLC2A1', 'GPI', 'PFKM', 'PFKL', 'PFKP', 'ALDOA', 'ALDOB', 'ALDOC',
        'TPI1', 'GAPDH', 'PGK1', 'PGAM1', 'ENO1', 'ENO2', 'PKM', 'LDHA', 'LDHB'
    ],
    'oxphos': [
        'NDUFB8', 'SDHB', 'UQCRC2', 'COX5B', 'ATP5F1A', 'COX6C', 'NDUFA4',
        'COX7C', 'UQCRH', 'NDUFAB1', 'COX4I1', 'ATP5PB', 'NDUFS5', 'ATP5MC1'
    ],
    'fetal_signature': [
        'AFP', 'EPCAM', 'KRT19', 'SOX9', 'DLK1', 'SALL4', 'IGF2', 'H19'
    ],
    'adult_signature': [
        'ALB', 'CYP3A4', 'CYP2E1', 'HNF4A', 'APOA1', 'APOB', 'TTR', 'TF'
    ]
}


class H5ADAnalyzer:
    """Helper class for h5ad file analysis."""
    
    def __init__(self, h5ad_path: str):
        """Initialize analyzer with h5ad file path.
        
        Args:
            h5ad_path: Path to the h5ad file
        """
        if not os.path.exists(h5ad_path):
            raise FileNotFoundError(f"File not found: {h5ad_path}")
        
        self.h5ad_path = h5ad_path
        self.file = None
        self.obs = None
        self.var = None
        self.X = None
        self.gene_names = None
    
    def __enter__(self):
        """Context manager entry."""
        self.file = h5py.File(self.h5ad_path, 'r')
        self._load_metadata()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.file:
            self.file.close()
    
    def _load_metadata(self):
        """Load obs and var DataFrames from h5ad file."""
        # Load obs (cell metadata)
        if 'obs' in self.file:
            obs_data = {}
            obs_group = self.file['obs']
            for key in obs_group.keys():
                data = obs_group[key][:]
                # Handle categorical data
                if hasattr(data[0], 'decode'):
                    data = [d.decode('utf-8') if isinstance(d, bytes) else d for d in data]
                obs_data[key] = data
            
            # Get cell names
            if '_index' in self.file['obs']:
                index = self.file['obs']['_index'][:]
                index = [i.decode('utf-8') if isinstance(i, bytes) else i for i in index]
            else:
                index = range(len(list(obs_data.values())[0]) if obs_data else 0)
            
            self.obs = pd.DataFrame(obs_data, index=index)
        
        # Load var (gene metadata)
        if 'var' in self.file:
            var_data = {}
            var_group = self.file['var']
            for key in var_group.keys():
                data = var_group[key][:]
                if hasattr(data[0], 'decode'):
                    data = [d.decode('utf-8') if isinstance(d, bytes) else d for d in data]
                var_data[key] = data
            
            # Get gene names
            if '_index' in self.file['var']:
                gene_names = self.file['var']['_index'][:]
                self.gene_names = [g.decode('utf-8') if isinstance(g, bytes) else g for g in gene_names]
            else:
                self.gene_names = list(range(len(list(var_data.values())[0]) if var_data else 0))
            
            self.var = pd.DataFrame(var_data, index=self.gene_names)
        
        # Load expression matrix X
        if 'X' in self.file:
            X_data = self.file['X']
            if isinstance(X_data, h5py.Dataset):
                self.X = X_data[:]
            elif 'data' in X_data:  # Sparse matrix
                # For sparse matrices, we'll load as dense for simplicity
                data = X_data['data'][:]
                indices = X_data['indices'][:]
                indptr = X_data['indptr'][:]
                shape = X_data.attrs.get('shape', (len(self.obs), len(self.var)))
                
                # Reconstruct sparse matrix as dense
                self.X = np.zeros(shape)
                for i in range(len(indptr) - 1):
                    start, end = indptr[i], indptr[i + 1]
                    self.X[i, indices[start:end]] = data[start:end]
    
    def get_metadata_summary(self) -> Dict:
        """Extract metadata summary.
        
        Returns:
            Dictionary with metadata information
        """
        summary = {
            'n_cells': len(self.obs) if self.obs is not None else 0,
            'n_genes': len(self.var) if self.var is not None else 0,
            'obs_columns': list(self.obs.columns) if self.obs is not None else [],
            'var_columns': list(self.var.columns) if self.var is not None else []
        }
        
        # Add categorical summaries
        if self.obs is not None:
            for col in self.obs.columns:
                if self.obs[col].dtype == 'object' or len(self.obs[col].unique()) < 50:
                    summary[f'obs_{col}_categories'] = self.obs[col].value_counts().to_dict()
        
        return summary
    
    def subset_cells(self, column: str, values: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
        """Subset cells based on metadata column.
        
        Args:
            column: Column name in obs
            values: List of values to keep
        
        Returns:
            Tuple of (expression matrix subset, obs subset)
        """
        if self.obs is None:
            raise ValueError("No obs metadata available")
        
        if column not in self.obs.columns:
            raise ValueError(f"Column {column} not found in obs. Available: {self.obs.columns.tolist()}")
        
        mask = self.obs[column].isin(values)
        X_subset = self.X[mask, :]
        obs_subset = self.obs[mask]
        
        return X_subset, obs_subset
    
    def compute_differential_expression(
        self,
        group_column: str,
        group1_values: List[str],
        group2_values: List[str],
        top_n: int = 50
    ) -> pd.DataFrame:
        """Compute differential expression using t-test.
        
        Args:
            group_column: Column name for grouping
            group1_values: Values for group 1
            group2_values: Values for group 2
            top_n: Number of top genes to return
        
        Returns:
            DataFrame with DE results (gene, logFC, p_value, adj_p_value)
        """
        # Subset cells
        X_group1, _ = self.subset_cells(group_column, group1_values)
        X_group2, _ = self.subset_cells(group_column, group2_values)
        
        if X_group1.shape[0] == 0 or X_group2.shape[0] == 0:
            raise ValueError("One or both groups have no cells")
        
        # Compute log fold change and t-test
        mean1 = np.mean(X_group1, axis=0)
        mean2 = np.mean(X_group2, axis=0)
        
        # Add pseudocount to avoid log(0)
        logFC = np.log2((mean1 + 1) / (mean2 + 1))
        
        # T-test
        t_stats, p_values = stats.ttest_ind(X_group1, X_group2, axis=0, equal_var=False)
        
        # Bonferroni correction for multiple testing
        adj_p_values = np.minimum(p_values * len(p_values), 1.0)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'gene': self.gene_names,
            'logFC': logFC,
            'mean_group1': mean1,
            'mean_group2': mean2,
            't_statistic': t_stats,
            'p_value': p_values,
            'adj_p_value': adj_p_values
        })
        
        # Sort by absolute logFC and filter
        results['abs_logFC'] = np.abs(results['logFC'])
        results = results.sort_values('abs_logFC', ascending=False)
        
        return results.head(top_n)
    
    def score_pathway(
        self,
        gene_set_name: str,
        custom_genes: Optional[List[str]] = None
    ) -> pd.Series:
        """Score cells for a pathway/gene set.
        
        Args:
            gene_set_name: Name of predefined gene set or 'custom'
            custom_genes: List of custom genes if gene_set_name is 'custom'
        
        Returns:
            Series with pathway scores per cell
        """
        if gene_set_name == 'custom' and custom_genes is None:
            raise ValueError("Must provide custom_genes when gene_set_name is 'custom'")
        
        if gene_set_name != 'custom':
            if gene_set_name not in GENE_SETS:
                raise ValueError(f"Gene set {gene_set_name} not found. Available: {list(GENE_SETS.keys())}")
            genes = GENE_SETS[gene_set_name]
        else:
            genes = custom_genes
        
        # Find gene indices
        gene_indices = [i for i, g in enumerate(self.gene_names) if g in genes]
        
        if len(gene_indices) == 0:
            raise ValueError(f"None of the genes found in dataset")
        
        # Compute mean expression across gene set
        X_subset = self.X[:, gene_indices]
        scores = np.mean(X_subset, axis=1)
        
        return pd.Series(scores, index=self.obs.index, name=f'{gene_set_name}_score')
    
    def compare_signatures(
        self,
        group_column: str,
        group1_values: List[str],
        group2_values: List[str],
        signature1_name: str = 'fetal_signature',
        signature2_name: str = 'adult_signature'
    ) -> Dict:
        """Compare two signatures between two groups.
        
        Args:
            group_column: Column name for grouping
            group1_values: Values for group 1 (e.g., fetal)
            group2_values: Values for group 2 (e.g., adult)
            signature1_name: Name of first signature
            signature2_name: Name of second signature
        
        Returns:
            Dictionary with comparison statistics
        """
        # Score both signatures
        score1 = self.score_pathway(signature1_name)
        score2 = self.score_pathway(signature2_name)
        
        # Subset by groups
        _, obs1 = self.subset_cells(group_column, group1_values)
        _, obs2 = self.subset_cells(group_column, group2_values)
        
        results = {
            f'{signature1_name}_group1_mean': score1.loc[obs1.index].mean(),
            f'{signature1_name}_group2_mean': score1.loc[obs2.index].mean(),
            f'{signature2_name}_group1_mean': score2.loc[obs1.index].mean(),
            f'{signature2_name}_group2_mean': score2.loc[obs2.index].mean(),
        }
        
        # T-tests
        t_stat1, p_val1 = stats.ttest_ind(
            score1.loc[obs1.index],
            score1.loc[obs2.index]
        )
        t_stat2, p_val2 = stats.ttest_ind(
            score2.loc[obs1.index],
            score2.loc[obs2.index]
        )
        
        results[f'{signature1_name}_t_statistic'] = t_stat1
        results[f'{signature1_name}_p_value'] = p_val1
        results[f'{signature2_name}_t_statistic'] = t_stat2
        results[f'{signature2_name}_p_value'] = p_val2
        
        return results


@tool
def h5ad_single_cell_analyzer(
    h5ad_path: str,
    operation: str,
    group_column: Optional[str] = None,
    group1_values: Optional[str] = None,
    group2_values: Optional[str] = None,
    gene_set_name: Optional[str] = None,
    custom_genes: Optional[str] = None,
    top_n: int = 50
) -> str:
    """Load and analyze h5ad files without scanpy.
    
    This tool provides comprehensive single-cell analysis including:
    - Metadata extraction (cell types, stages)
    - Cell subsetting
    - Differential expression (logFC via t-test)
    - Pathway scoring (glycolysis/OXPHOS gene sets)
    - Signature mapping for fetal vs adult comparisons
    
    Args:
        h5ad_path: Path to the h5ad file
        operation: Operation to perform. Options:
            - 'metadata': Extract metadata summary
            - 'subset': Subset cells by criteria
            - 'differential_expression': Compute DE between groups
            - 'pathway_score': Score cells for a pathway
            - 'compare_signatures': Compare fetal vs adult signatures
        group_column: Column name for grouping (required for subset, DE, compare)
        group1_values: Comma-separated values for group 1 (required for subset, DE, compare)
        group2_values: Comma-separated values for group 2 (required for DE, compare)
        gene_set_name: Gene set name (required for pathway_score, compare). 
            Options: 'angiogenic', 'glycolysis', 'oxphos', 'fetal_signature', 'adult_signature', 'custom'
        custom_genes: Comma-separated custom gene list (required if gene_set_name='custom')
        top_n: Number of top genes for DE (default: 50)
    
    Returns:
        String containing analysis results in a formatted manner
    
    Examples:
        # Extract metadata
        h5ad_single_cell_analyzer('data.h5ad', operation='metadata')
        
        # Differential expression between fetal and adult
        h5ad_single_cell_analyzer(
            'data.h5ad',
            operation='differential_expression',
            group_column='stage',
            group1_values='fetal',
            group2_values='adult',
            top_n=50
        )
        
        # Score glycolysis pathway
        h5ad_single_cell_analyzer(
            'data.h5ad',
            operation='pathway_score',
            gene_set_name='glycolysis'
        )
        
        # Compare fetal vs adult signatures
        h5ad_single_cell_analyzer(
            'data.h5ad',
            operation='compare_signatures',
            group_column='stage',
            group1_values='fetal',
            group2_values='adult',
            gene_set_name='fetal_signature'
        )
    """
    try:
        with H5ADAnalyzer(h5ad_path) as analyzer:
            
            if operation == 'metadata':
                summary = analyzer.get_metadata_summary()
                result = "=== H5AD Metadata Summary ===\n"
                result += f"Number of cells: {summary['n_cells']}\n"
                result += f"Number of genes: {summary['n_genes']}\n"
                result += f"\nObs columns: {', '.join(summary['obs_columns'])}\n"
                result += f"Var columns: {', '.join(summary['var_columns'])}\n"
                
                # Add categorical summaries
                for key, value in summary.items():
                    if key.startswith('obs_') and key.endswith('_categories'):
                        col_name = key.replace('obs_', '').replace('_categories', '')
                        result += f"\n{col_name} distribution:\n"
                        for cat, count in list(value.items())[:10]:  # Top 10
                            result += f"  {cat}: {count}\n"
                
                return result
            
            elif operation == 'subset':
                if not group_column or not group1_values:
                    raise ValueError("subset operation requires group_column and group1_values")
                
                values = [v.strip() for v in group1_values.split(',')]
                X_subset, obs_subset = analyzer.subset_cells(group_column, values)
                
                result = f"=== Cell Subset Results ===\n"
                result += f"Selected {X_subset.shape[0]} cells out of {analyzer.X.shape[0]} total\n"
                result += f"Criteria: {group_column} in {values}\n"
                return result
            
            elif operation == 'differential_expression':
                if not all([group_column, group1_values, group2_values]):
                    raise ValueError("DE requires group_column, group1_values, and group2_values")
                
                g1_vals = [v.strip() for v in group1_values.split(',')]
                g2_vals = [v.strip() for v in group2_values.split(',')]
                
                de_results = analyzer.compute_differential_expression(
                    group_column, g1_vals, g2_vals, top_n
                )
                
                result = f"=== Differential Expression Results ===\n"
                result += f"Comparison: {g1_vals} vs {g2_vals}\n"
                result += f"Top {top_n} genes by absolute log fold change:\n\n"
                result += de_results.to_string(index=False)
                return result
            
            elif operation == 'pathway_score':
                if not gene_set_name:
                    raise ValueError("pathway_score requires gene_set_name")
                
                custom_gene_list = None
                if gene_set_name == 'custom':
                    if not custom_genes:
                        raise ValueError("custom gene set requires custom_genes parameter")
                    custom_gene_list = [g.strip() for g in custom_genes.split(',')]
                
                scores = analyzer.score_pathway(gene_set_name, custom_gene_list)
                
                result = f"=== Pathway Scoring Results ===\n"
                result += f"Gene set: {gene_set_name}\n"
                if gene_set_name != 'custom':
                    result += f"Genes: {', '.join(GENE_SETS[gene_set_name])}\n"
                result += f"\nScore statistics:\n"
                result += f"  Mean: {scores.mean():.4f}\n"
                result += f"  Std: {scores.std():.4f}\n"
                result += f"  Min: {scores.min():.4f}\n"
                result += f"  Max: {scores.max():.4f}\n"
                return result
            
            elif operation == 'compare_signatures':
                if not all([group_column, group1_values, group2_values]):
                    raise ValueError("compare_signatures requires group_column, group1_values, and group2_values")
                
                g1_vals = [v.strip() for v in group1_values.split(',')]
                g2_vals = [v.strip() for v in group2_values.split(',')]
                
                # Use default signatures or custom
                sig1 = gene_set_name or 'fetal_signature'
                sig2 = 'adult_signature'  # Default second signature
                
                comparison = analyzer.compare_signatures(
                    group_column, g1_vals, g2_vals, sig1, sig2
                )
                
                result = f"=== Signature Comparison Results ===\n"
                result += f"Groups: {g1_vals} vs {g2_vals}\n\n"
                
                for key, value in comparison.items():
                    if isinstance(value, (int, float)):
                        result += f"{key}: {value:.6f}\n"
                    else:
                        result += f"{key}: {value}\n"
                
                return result
            
            else:
                raise ValueError(
                    f"Unknown operation: {operation}. "
                    f"Valid options: metadata, subset, differential_expression, "
                    f"pathway_score, compare_signatures"
                )
    
    except Exception as e:
        return f"Error in h5ad_single_cell_analyzer: {str(e)}\n\nPlease check your inputs and file format."
