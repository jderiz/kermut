# %% [markdown]
# # Kermut
# Evaluate how to use the model, how to save trained models and load them. and how to run inference.

# %%


# %%
import pandas as pd
dms = pd.read_csv('data/DMS_substitutions.csv')

# %%
gb1 = pd.read_csv('/home/iwe7/haipr/benchmark/BindingGYM/input/Binding_substitutions_DMS/GB1_IgG-Fc_fitness_1FCC_2016.csv')
gb1.columns
""" 
Index(['POI', 'DMS_score', 'wildtype_sequence', 'mutant', 'mutated_sequence',
       'chain_id', 'pdb_file', 'mutant_pdb'],
      dtype='object')
"""
# %%
bg = pd.read_csv('/home/iwe7/haipr/benchmark/BindingGYM/input/BindingGYM.csv')
print(bg.columns)
gb1_bg = bg[bg.DMS_filename == 'GB1_IgG-Fc_fitness_1FCC_2016.csv']
"""
Index(['POI', 'DMS_id', 'DMS_filename', 'wildtype_sequence', 'chain_id',
       'pdb_file'],
      dtype='object')
"""
# %%
dms = pd.concat([dms, gb1_bg.rename(columns={'DMS_filename': 'raw_DMS_filename'})])

# %%
dms.columns
"""
Index(['DMS_index', 'DMS_id', 'DMS_filename', 'UniProt_ID', 'taxon',
       'source_organism', 'target_seq', 'seq_len', 'includes_multiple_mutants',
       'DMS_total_number_mutants', 'DMS_number_single_mutants',
       'DMS_number_multiple_mutants', 'DMS_binarization_cutoff',
       'DMS_binarization_method', 'first_author', 'title', 'year', 'jo',
       'region_mutated', 'molecule_name', 'selection_assay', 'selection_type',
       'MSA_filename', 'MSA_start', 'MSA_end', 'MSA_len', 'MSA_bitscore',
       'MSA_theta', 'MSA_num_seqs', 'MSA_perc_cov', 'MSA_num_cov', 'MSA_N_eff',
       'MSA_Neff_L', 'MSA_Neff_L_category', 'MSA_num_significant',
       'MSA_num_significant_L', 'raw_DMS_filename', 'raw_DMS_phenotype_name',
       'raw_DMS_directionality', 'raw_DMS_mutant_column', 'weight_file_name',
       'pdb_file', 'pdb_range', 'ProteinGym_version', 'raw_mut_offset',
       'coarse_selection_type', 'POI', 'wildtype_sequence', 'chain_id'],
      dtype='object')
"""
# %%
dms[dms.raw_DMS_filename == dms.DMS_filename]

# %% [markdown]
# # Kermut
# Evaluate how to use the model, how to save trained models and load them. and how to run inference.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig
from typing import Dict, List, Tuple, Optional

def process_bindinggym_to_proteingym(
    bg_data: pd.DataFrame,
    dms_id: str,
    target_seq: str,
    pdb_file: str,
    chain_id: str,
    region_mutated: str,
    msa_filename: Optional[str] = None,
    msa_start: Optional[int] = None,
    msa_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert BindingGYM format to ProteinGym format.
    
    Args:
        bg_data: BindingGYM dataframe
        dms_id: Unique identifier for the DMS assay
        target_seq: Reference sequence of the target protein
        pdb_file: PDB file identifier
        chain_id: Chain identifier in the PDB file
        region_mutated: Region of the protein that was mutated
        msa_filename: Optional MSA filename
        msa_start: Optional start position in MSA
        msa_end: Optional end position in MSA
    
    Returns:
        DataFrame in ProteinGym format
    """
    # Create base ProteinGym format
    pg_data = pd.DataFrame()
    
    # Required fields from reference_files_description.md
    pg_data['DMS_id'] = dms_id
    pg_data['DMS_filename'] = f"{dms_id}.csv"
    pg_data['target_seq'] = target_seq
    pg_data['seq_len'] = len(target_seq)
    pg_data['includes_multiple_mutants'] = bg_data['mutant'].str.contains(',').any()
    pg_data['DMS_total_number_mutants'] = len(bg_data)
    pg_data['DMS_number_single_mutants'] = len(bg_data[~bg_data['mutant'].str.contains(',')])
    pg_data['DMS_number_multiple_mutants'] = len(bg_data[bg_data['mutant'].str.contains(',')])
    
    # Calculate binarization cutoff using median method
    pg_data['DMS_binarization_cutoff_ProteinGym'] = bg_data['DMS_score'].median()
    pg_data['DMS_binarization_method'] = 'median'
    
    # Add MSA information if provided
    if msa_filename:
        pg_data['MSA_filename'] = msa_filename
        pg_data['MSA_start'] = msa_start
        pg_data['MSA_end'] = msa_end
    
    # Add raw DMS information
    pg_data['raw_DMS_filename'] = bg_data['DMS_filename'].iloc[0]
    pg_data['raw_DMS_phenotype_name'] = 'DMS_score'
    pg_data['raw_DMS_directionality'] = 1  # Assuming higher score means higher fitness
    pg_data['raw_DMS_mutant_column'] = 'mutant'
    
    # Add PDB information
    pg_data['pdb_file'] = pdb_file
    pg_data['chain_id'] = chain_id
    pg_data['region_mutated'] = region_mutated
    
    return pg_data

def validate_proteingym_format(pg_data: pd.DataFrame) -> bool:
    """
    Validate that the data conforms to ProteinGym format requirements.
    
    Args:
        pg_data: DataFrame in ProteinGym format
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    required_columns = [
        'DMS_id', 'DMS_filename', 'target_seq', 'seq_len',
        'includes_multiple_mutants', 'DMS_total_number_mutants',
        'DMS_number_single_mutants', 'DMS_number_multiple_mutants',
        'DMS_binarization_cutoff_ProteinGym', 'DMS_binarization_method',
        'raw_DMS_filename', 'raw_DMS_phenotype_name',
        'raw_DMS_directionality', 'raw_DMS_mutant_column'
    ]
    
    # Check all required columns exist
    if not all(col in pg_data.columns for col in required_columns):
        return False
    
    # Validate data types and ranges
    if not isinstance(pg_data['seq_len'].iloc[0], int):
        return False
    if not isinstance(pg_data['includes_multiple_mutants'].iloc[0], bool):
        return False
    if not isinstance(pg_data['DMS_binarization_cutoff_ProteinGym'].iloc[0], (int, float)):
        return False
    
    return True

# %%
# Load BindingGYM data bg_data = pd.read_csv('path/to/bindinggym_data.csv')
bg_data = pd.read_csv('/home/iwe7/haipr/benchmark/BindingGYM/input/Binding_substitutions_DMS/GB1_IgG-Fc_fitness_1FCC_2016.csv')
 
# Convert to ProteinGym format
pg_data = process_bindinggym_to_proteingym(
    bg_data=bg_data,
    dms_id='GB1_IgG-Fc_2016',
    target_seq=bg_data['wildtype_sequence'].iloc[0],
    pdb_file='1FCC',
    chain_id=bg_data['chain_id'].iloc[0],
    region_mutated='full_sequence'  # Update based on actual region
)

# Validate format
if validate_proteingym_format(pg_data):
    # Save to CSV
    pass
else:
    print("Validation failed - check data format")









# %%
