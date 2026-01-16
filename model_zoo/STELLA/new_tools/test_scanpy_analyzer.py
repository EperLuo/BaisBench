#!/usr/bin/env python3
"""Comprehensive test script for scanpy_h5ad_analyzer tool"""

import sys
import os
import numpy as np

try:
    import scanpy as sc
    import anndata as ad
except ImportError:
    print("ERROR: scanpy and anndata required. Install with: pip install scanpy anndata")
    sys.exit(1)

# Import the tool
sys.path.insert(0, './new_tools')
from scanpy_h5ad_analyzer import scanpy_h5ad_analyzer

def create_test_data(filepath='./new_tools/test_data.h5ad'):
    """Create synthetic h5ad file for testing"""
    print("Creating synthetic test data...")
    np.random.seed(42)
    
    # Create 500 cells, 100 genes
    X = np.random.negative_binomial(5, 0.3, (500, 100)).astype(np.float32)
    
    # Gene names including known markers
    genes = [f'Gene_{i}' for i in range(80)] + [
        'CD3D', 'CD4', 'CD8A', 'MS4A1', 'CD19', 'CD14', 'FCGR3A', 'IL7R', 
        'CCR7', 'S100A8', 'GNLY', 'NKG7', 'FCER1A', 'CST3', 'PPBP',
        'CD79A', 'CD79B', 'LYZ', 'TGFB1', 'CXCL8'
    ]
    
    # Cell types
    cell_types = np.array(['T cells']*200 + ['B cells']*150 + ['Monocytes']*100 + ['NK cells']*50)
    
    # Boost marker expression in specific cell types
    X[:200, 80:83] *= 5  # T cell markers
    X[200:350, [83, 84, 95, 96]] *= 5  # B cell markers
    X[350:450, [85, 97]] *= 5  # Monocyte markers
    X[450:, [90, 91]] *= 5  # NK markers
    
    # Create AnnData
    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = [f'Cell_{i}' for i in range(500)]
    adata.obs['cell_type'] = cell_types
    adata.obs['sample'] = np.random.choice(['Sample_A', 'Sample_B'], 500)
    
    adata.write_h5ad(filepath)
    print(f"✓ Test data saved: {filepath}\n")
    return filepath

def run_tests():
    """Run all tests"""
    print("="*70)
    print("SCANPY H5AD ANALYZER - COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    test_file = create_test_data()
    tests_passed = 0
    tests_total = 7
    
    # Test 1: Load
    print("[TEST 1/7] Load Operation")
    print("-" * 70)
    result = scanpy_h5ad_analyzer(test_file, operation='load', cell_annotation_column='cell_type')
    if result['status'] == 'success':
        print(f"✓ Status: {result['status']}")
        print(f"✓ Cells: {result['n_cells']}, Genes: {result['n_genes']}")
        print(f"✓ Cell types: {result['data']['cell_types']}")
        tests_passed += 1
    else:
        print(f"✗ FAILED: {result['message']}")
    print()
    
    # Test 2: Expression metrics
    print("[TEST 2/7] Expression Metrics")
    print("-" * 70)
    result = scanpy_h5ad_analyzer(
        test_file, operation='expression_metrics',
        cell_annotation_column='cell_type', cell_subset='T cells',
        gene_list=['CD3D', 'CD4', 'CD8A', 'MS4A1']
    )
    if result['status'] == 'success':
        print(f"✓ Status: {result['status']}")
        for gene, metrics in result['data']['expression_metrics'].items():
            print(f"  {gene}: mean={metrics['mean_expression']:.2f}, pct={metrics['percent_cells_expressing']:.1f}%")
        tests_passed += 1
    else:
        print(f"✗ FAILED: {result['message']}")
    print()
    
    # Test 3: Differential expression
    print("[TEST 3/7] Differential Expression (T cells vs B cells)")
    print("-" * 70)
    result = scanpy_h5ad_analyzer(
        test_file, operation='differential_expression',
        group_by='cell_type', reference_group='B cells', test_group='T cells'
    )
    if result['status'] == 'success':
        print(f"✓ Status: {result['status']}")
        print(f"✓ Significant genes: {result['data']['differential_expression']['n_significant_genes']}")
        print("  Top upregulated in T cells:")
        for g in result['data']['differential_expression']['top_upregulated'][:3]:
            print(f"    {g['names']}: logFC={g['logfoldchanges']:.2f}, p-adj={g['pvals_adj']:.2e}")
        tests_passed += 1
    else:
        print(f"✗ FAILED: {result['message']}")
    print()
    
    # Test 4: Marker validation
    print("[TEST 4/7] Marker Validation")
    print("-" * 70)
    result = scanpy_h5ad_analyzer(
        test_file, operation='validate_markers',
        gene_list=['CD3D', 'MS4A1', 'CD14', 'NKG7'], group_by='cell_type'
    )
    if result['status'] == 'success':
        print(f"✓ Status: {result['status']}")
        for ctype in ['T cells', 'B cells']:
            print(f"  {ctype}:")
            for gene in ['CD3D', 'MS4A1']:
                m = result['data']['marker_validation'][ctype][gene]
                print(f"    {gene}: {m['pct_expressing']:.1f}% cells")
        tests_passed += 1
    else:
        print(f"✗ FAILED: {result['message']}")
    print()
    
    # Test 5: Ligand-receptor
    print("[TEST 5/7] Ligand-Receptor Co-expression")
    print("-" * 70)
    result = scanpy_h5ad_analyzer(
        test_file, operation='ligand_receptor',
        ligand_receptor_pairs=[('TGFB1', 'CD3D'), ('CXCL8', 'CD14')]
    )
    if result['status'] == 'success':
        print(f"✓ Status: {result['status']}")
        for lr in result['data']['ligand_receptor_analysis']:
            print(f"  {lr['ligand']}-{lr['receptor']}: {lr['coexpression_pct']:.1f}% co-expressing")
        tests_passed += 1
    else:
        print(f"✗ FAILED: {result['message']}")
    print()
    
    # Test 6: Cell subsetting
    print("[TEST 6/7] Cell Subsetting")
    print("-" * 70)
    result = scanpy_h5ad_analyzer(
        test_file, operation='subset_cells',
        cell_annotation_column='cell_type', cell_subset='Monocytes'
    )
    if result['status'] == 'success':
        print(f"✓ Status: {result['status']}")
        print(f"✓ Subset cells: {result['n_cells']}")
        tests_passed += 1
    else:
        print(f"✗ FAILED: {result['message']}")
    print()
    
    # Test 7: Error handling
    print("[TEST 7/7] Error Handling")
    print("-" * 70)
    result = scanpy_h5ad_analyzer('nonexistent.h5ad', operation='load')
    if result['status'] == 'error':
        print(f"✓ Correctly detected error: {result['message']}")
        tests_passed += 1
    else:
        print("✗ FAILED: Should have returned error")
    print()
    
    # Summary
    print("="*70)
    print(f"TEST RESULTS: {tests_passed}/{tests_total} PASSED")
    if tests_passed == tests_total:
        print("✓ ALL TESTS PASSED - Tool is production ready!")
    else:
        print(f"✗ {tests_total - tests_passed} test(s) failed")
    print("="*70)
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n✓ Cleaned up: {test_file}")
    
    return tests_passed == tests_total

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
