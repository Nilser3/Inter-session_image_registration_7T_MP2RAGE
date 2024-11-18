"""

Compute metrics of similarity between two images 
Tha images must share the same resolution, dimension and spacing (coregistered images).

Details of flags: 

-ses2_file : Path to the session 2 NIfTI file (fixed image)
-ses1reg_file : Path to the registered session 1 NIfTI file (registered image)
-acquisition : Acquisition type (i.e. T1w , T2w)
-method : Registration method used (i.e. SCT_rigid, ANTS_syn)
-mask_file : Optional path to a binary mask NIfTI file (i.e. SC mask, Brain mask)
-o : Path to save the output CSV file (i.e. intersession_t2w.csv)

Usage:
    python calculate_similarity_metrics.py -ses2_file   -ses1reg_file   -acquisition  -method    -mask_file  -o

Authors: Nilser Laines Medina
Date: 2024-06-20

"""

import numpy as np
import nibabel as nib
import pandas as pd
import argparse
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from skimage.metrics import normalized_mutual_information as nmi_sk
from sklearn.preprocessing import KBinsDiscretizer
import copy
import os

def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def apply_mask(image, mask):
    return image * mask

def calculate_nmi(image1, image2, bins=32):
    # Flatten the images
    image1 = image1.ravel()
    image2 = image2.ravel()
    
    # Discretize the images
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    image1_disc = discretizer.fit_transform(image1[:, None]).ravel()
    image2_disc = discretizer.fit_transform(image2[:, None]).ravel()
    
    # Compute the joint histogram
    joint_hist, _, _ = np.histogram2d(image1_disc, image2_disc, bins=bins)
    
    # Compute the joint probability distribution
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Compute the marginal probabilities
    prob_image1 = np.sum(joint_prob, axis=1)
    prob_image2 = np.sum(joint_prob, axis=0)
    
    # Compute entropies
    entropy_image1 = -np.sum(prob_image1 * np.log(prob_image1 + 1e-10))
    entropy_image2 = -np.sum(prob_image2 * np.log(prob_image2 + 1e-10))
    entropy_joint = -np.sum(joint_prob * np.log(joint_prob + 1e-10))
    
    # Compute mutual information and normalized mutual information
    mutual_information = entropy_image1 + entropy_image2 - entropy_joint
    nmi = 2 * mutual_information / (entropy_image1 + entropy_image2)
    
    return nmi

def main(ses2_file, ses1reg_file, acquisition, method, mask_file=None, output_csv=None):
    image_ses2 = load_nifti(ses2_file)
    image_ses1_reg = load_nifti(ses1reg_file)

    if mask_file:
        mask = load_nifti(mask_file)
        image_ses2 = apply_mask(image_ses2, mask)
        image_ses1_reg = apply_mask(image_ses1_reg, mask)
        masked = True
    else:
        masked = False

    # Calculate similarity metrics
    correlation_coefficient, _ = pearsonr(image_ses2.ravel(), image_ses1_reg.ravel())
    ssim_index, _ = ssim(image_ses2, image_ses1_reg, data_range=1, full=True)  # Add data_range=1

    mse = np.mean((image_ses2 - image_ses1_reg) ** 2)
    ncc = np.corrcoef(image_ses2.ravel(), image_ses1_reg.ravel())[0, 1]
    nmi = calculate_nmi(image_ses2, image_ses1_reg)

    data = {
        'Subject': [os.path.basename(ses2_file)],
        'Acq': [acquisition],
        'Method': [method],
        'Cross-Correlation Index (CC - Opt.val: 1)': [correlation_coefficient],
        'Structural Similarity Index (SSIM - Opt.val: 1)': [ssim_index],
        'Mean Squared Error (MSE - Opt.val: 0)': [mse],
        'Normalized Cross-Correlation (NCC - Opt.val: 1)': [ncc],
        'Normalized Mutual Information (NMI - Opt.val: 1)': [nmi],
        'Masked': [masked]
    }

    df = pd.DataFrame(data)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
    else:
        print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate image similarity metrics.")
    parser.add_argument('-ses2_file', required=True, help="Path to the session 2 NIfTI file.")
    parser.add_argument('-ses1reg_file', required=True, help="Path to the registered session 1 NIfTI file.")
    parser.add_argument('-acquisition', required=True, help="Acquisition type.")
    parser.add_argument('-method', required=True, help="Registration method used.")
    parser.add_argument('-mask_file', help="Optional path to a binary mask NIfTI file.")
    parser.add_argument('-o', '--output_csv', help="Path to save the output CSV file.")
    args = parser.parse_args()

    main(args.ses2_file, args.ses1reg_file, args.acquisition, args.method, args.mask_file, args.output_csv)