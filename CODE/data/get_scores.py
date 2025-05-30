import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.config_la import h_init, w_init, d_init, device
from data.atlas_scores.high_imp_regions import get_high_regions

def prepare_regions():
    # Load the JHU WM atlas .nii.gz file
    file_path = './data/atlas_scores/JHU-ICBM-labels-1mm.nii.gz'          
    wm_img = nib.load(file_path)
    wm_atlas_data = wm_img.get_fdata()

    # Load the AAL3v1 atlas .nii.gz file
    file_path = './data/atlas_scores/AAL3v1_1mm.nii.gz' 
    img = nib.load(file_path)
    aal_atlas = img.get_fdata()

    # Print or explore the data
    print("AAL atlas:",aal_atlas.shape)  # Prints the shape of the 3D/4D data
    print("JHU WM atlas:",wm_atlas_data.shape)  # Prints the shape of the 3D/4D data

    #zoom atlas
    # Get atlas and data arrays
    atlas_data = aal_atlas.copy()

    # Calculate the zoom factors for resampling the atlas
    zoom_factors = (
        h_init / atlas_data.shape[0],
        w_init / atlas_data.shape[1],
        d_init / atlas_data.shape[2]
    )

    # Resample the atlas data
    aal_atlas_resampled_data = zoom(atlas_data, zoom_factors, order=0)  
    aal_atlas_data_r = np.array(aal_atlas_resampled_data, dtype=int)
    wm_atlas_data_r = np.array(wm_atlas_data, dtype=int)

    # Lists of high relevance regions
    AAL3_high_relevance, JHU_WM_high_relevance = get_high_regions()
    wm_high_relevance = set(JHU_WM_high_relevance)
    aal_high_relevance = set(AAL3_high_relevance)

    # Create masks for high relevance and non-zero image regions
    wm_high_relevance_mask = np.isin(wm_atlas_data_r, list(wm_high_relevance))
    aal_high_relevance_mask = np.isin(aal_atlas_data_r, list(aal_high_relevance))
    high_relevance_mask = np.logical_or(wm_high_relevance_mask, aal_high_relevance_mask)

    return torch.tensor(high_relevance_mask, dtype=torch.bool)


def process_batch(batch_data, high_relevance_mask, device):
    high_relevance_mask = high_relevance_mask.to(device)
    high_relevance_mask_expanded = high_relevance_mask.unsqueeze(0).expand(batch_data.shape[0],-1, -1, -1, -1)

    # Use reshape instead of view to avoid stride issues.
    min_values_per_sample = batch_data.reshape(batch_data.shape[0], -1).min(dim=1, keepdim=True)[0]
    min_values_per_sample = min_values_per_sample.view(batch_data.shape[0], 1, 1, 1, 1)
    
    # Create masks for non-minimum image regions
    non_minimum_smri_mask = batch_data != min_values_per_sample

    # Combine masks to identify low-importance regions
    low_relevance_mask = non_minimum_smri_mask.to(device) & ~high_relevance_mask_expanded

    # Initialize atlas data scores matrix
    atlas_data_scores = torch.zeros_like(batch_data, dtype=torch.int)

    # Set high and low relevance values
    atlas_data_scores[high_relevance_mask_expanded] = 2
    atlas_data_scores[low_relevance_mask] = 1

    return atlas_data_scores


if __name__ == "__main__":
    high_relevance_mask = prepare_regions()

    # Load a batch of sample data (28 samples, assumed to be loaded as tensors)
    batch_file_paths = ['./DATASET/nii_files_preprocessed_all/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114556321_S13408_I118671.nii.gz',
                        './DATASET/nii_files_preprocessed_all/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001120532722_S21856_I118692.nii.gz',
                        './DATASET/nii_files_preprocessed_all/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219173850420_S21856_I40966.nii.gz']
    batch_data = []

    for file_path in batch_file_paths:
        sample_img = nib.load(file_path)
        sample_data = sample_img.get_fdata()
        batch_data.append(torch.tensor(sample_data))

    batch_data = torch.stack(batch_data)

    atlas_data_scores_batch = process_batch(batch_data, high_relevance_mask)

    # Count occurrences for each sample in the batch
    count_1 = (atlas_data_scores_batch == 1).sum(dim=(1, 2, 3))
    count_2 = (atlas_data_scores_batch == 2).sum(dim=(1, 2, 3))
    count_0 = (atlas_data_scores_batch == 0).sum(dim=(1, 2, 3))
    total_count = atlas_data_scores_batch.numel() // atlas_data_scores_batch.shape[0]

    for i in range(atlas_data_scores_batch.shape[0]):
        print(f"Sample {i+1}:")
        print("Number of 2s:", count_2[i].item())
        print("Number of 1s:", count_1[i].item())
        print("Number of 0s:", count_0[i].item())
        print("Total count:", total_count)
        print("Ratio of 2s/total: ", 100 * count_2[i].item() / total_count)
        print("Ratio of 1s/total: ", 100 * count_1[i].item() / total_count)
        print("Ratio of 0s/total: ", 100 * count_0[i].item() / total_count)
