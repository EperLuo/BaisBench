import h5py
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from h5ad_validator import h5ad_validator

def create_sample_h5ad(filename='test_sample.h5ad'):
    print('Creating sample H5AD file...')
    n_obs = 100
    n_vars = 50
    
    with h5py.File(filename, 'w') as f:
        X_group = f.create_group('X')
        non_zero = 1500
        data = np.random.randn(non_zero).astype(np.float32)
        indices = np.random.randint(0, n_vars, non_zero).astype(np.int32)
        indptr = np.sort(np.random.randint(0, non_zero, n_obs + 1)).astype(np.int32)
        indptr[0] = 0
        indptr[-1] = non_zero
        
        X_group.create_dataset('data', data=data)
        X_group.create_dataset('indices', data=indices)
        X_group.create_dataset('indptr', data=indptr)
        X_group.attrs['shape'] = [n_obs, n_vars]
        
        obs_group = f.create_group('obs')
        cell_names = [f'CELL_{i:04d}'.encode('utf-8') for i in range(n_obs)]
        obs_group.create_dataset('_index', data=np.array(cell_names, dtype='S'))
        
        cell_type_categories = np.array([b'Endothelial cells', b'Fibroblasts', b'Immune cells', b'Epithelial cells'])
        cell_type_codes = np.random.randint(0, 4, n_obs).astype(np.int8)
        ct_dataset = obs_group.create_dataset('cell_type', data=cell_type_codes)
        ct_dataset.attrs['categories'] = cell_type_categories
        
        stage_categories = np.array([b'E10.5', b'E12.5', b'E14.5', b'E16.5'])
        stage_codes = np.random.randint(0, 4, n_obs).astype(np.int8)
        stage_dataset = obs_group.create_dataset('developmental_stage', data=stage_codes)
        stage_dataset.attrs['categories'] = stage_categories
        
        clusters = np.random.randint(0, 8, n_obs).astype(np.int32)
        obs_group.create_dataset('louvain', data=clusters)
        
        var_group = f.create_group('var')
        gene_names = [b'VEGFA', b'PECAM1', b'VWF', b'CD34', b'KDR', b'ACTB', b'GAPDH', b'TP53', b'MYC'] + [f'GENE_{i:04d}'.encode('utf-8') for i in range(n_vars - 9)]
        var_group.create_dataset('_index', data=np.array(gene_names[:n_vars], dtype='S'))
        highly_variable = np.random.choice([True, False], n_vars)
        var_group.create_dataset('highly_variable', data=highly_variable)
    
    print(f'Sample H5AD file created: {filename}')
    return filename

def test_validator():
    print('Testing H5AD Validator Tool')
    sample_file = create_sample_h5ad()
    result = h5ad_validator(sample_file, sample_size=10)
    report = json.loads(result)
    print(json.dumps(report, indent=2))
    print('Test completed successfully!')

if __name__ == '__main__':
    test_validator()
