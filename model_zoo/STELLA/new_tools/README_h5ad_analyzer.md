# h5ad_analyzer Tool

## Overview
Comprehensive single-cell RNA-seq analysis tool for h5ad files using scanpy and anndata.

## Features
- Metadata extraction and dataset statistics
- Leiden clustering analysis
- Marker gene identification
- Cell type abundance analysis
- Differential expression analysis
- Oligodendrocyte sub-cluster analysis
- Cross-species gene conservation
- MS-related analysis support

## Installation
```bash
pip install scanpy anndata pandas numpy scipy
```

## Usage
```python
from h5ad_analyzer import h5ad_analyzer

# Basic metadata analysis
result = h5ad_analyzer("path/to/file.h5ad", analysis_type="metadata")

# Full comprehensive analysis
result = h5ad_analyzer("path/to/file.h5ad", analysis_type="full")

# Clustering with custom resolution
result = h5ad_analyzer("path/to/file.h5ad", analysis_type="clustering", cluster_resolution=0.5)
```

## Parameters
- `file_path`: Path to h5ad file (local or URL)
- `analysis_type`: metadata, clustering, markers, abundance, differential, oligo_subcluster, cross_species, full
- `cluster_resolution`: Leiden clustering resolution (default: 1.0)
- `n_top_genes`: Number of top marker genes (default: 50)
- `groupby`: Column for grouping analysis
- `reference_group`: Reference for differential expression
- `species`: Species identifier
- `cell_type_focus`: Specific cell type to analyze
- `output_format`: summary or detailed

## Returns
Formatted string with analysis results including statistics, cluster information, and marker genes.
