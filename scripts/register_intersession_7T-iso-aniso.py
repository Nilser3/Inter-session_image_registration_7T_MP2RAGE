"""
Fully automated registration of inter-session images 7T MP2RAGE (UNIT1 isotropic or anisotropic images)

Dependencies: 
    SCT v6.4 and later versions 

To install: 
    sct_deepseg -install-task seg_sc_contrast_agnostic
    sct_deepseg -install seg_spinal_rootlets_t2w -custom-url https://github.com/ivadomed/model-spinal-rootlets/releases/download/r20240915/model-spinal-rootlets-mp2rage-r20240915.zip
    sct_deepseg -install-task seg_ms_lesion_mp2rage

Details of flags: 
    -ses01              Path to the session 1 MRI image (UNIT1 contrast)
    -ses02              Path to the session 2 MRI image (UNIT1 contrast)
    -level_up           Upper cervical level for calculating the center of mass.
    -level_down         Lower cervical level for calculating the center of mass.
    -output_folder      Output folder for processed images.
    -anisotropic        Flag to process anisotropic data (default: False).

Usage:
    python register_intersession_7T-iso-aniso.py -ses01 SES01.nii.gz -ses02 SES02.nii.gz -level_up LEVEL_UP -level_down LEVEL_DOWN -output_folder OUTPUT_FOLDER -anisotropic

Authors : Nilser Laines Medina
Date    : 2024-11-14

"""

import os
import argparse
import subprocess
import nibabel as nib
import numpy as np
import warnings
from scipy.ndimage import center_of_mass

# Main function to perform the inter-session registration
def main(ses01, ses02, level_up, level_down, output_folder, anisotropic=False):
    print("Creating output directory if it does not exist.")
    os.makedirs(output_folder, exist_ok=True)

    # Set orientation adjustment to RPI 
    subprocess.run(f"sct_image -i {ses01} -setorient RPI -o {output_folder}/Image_ses01.nii.gz", shell=True)
    subprocess.run(f"sct_image -i {ses02} -setorient RPI -o {output_folder}/Image_ses02.nii.gz", shell=True)

    # Segmentation of rootlets and spinal cord
    subprocess.run(f"sct_deepseg -i {output_folder}/Image_ses01.nii.gz -task seg_spinal_rootlets_t2w -o {output_folder}/Image_ses01_rootlets.nii.gz", shell=True)
    subprocess.run(f"sct_deepseg -i {output_folder}/Image_ses02.nii.gz -task seg_spinal_rootlets_t2w -o {output_folder}/Image_ses02_rootlets.nii.gz", shell=True)
    subprocess.run(f"sct_deepseg -i {output_folder}/Image_ses01.nii.gz -task seg_sc_contrast_agnostic -o {output_folder}/Image_ses01_sc.nii.gz", shell=True)
    subprocess.run(f"sct_deepseg -i {output_folder}/Image_ses02.nii.gz -task seg_sc_contrast_agnostic -o {output_folder}/Image_ses02_sc.nii.gz", shell=True)
    
    # Check if the data is anisotropic or isotropic
    if anisotropic:
        dilation_size = "100x100x2"
        # Function to calculate centers of mass for specified labels
        def calculate_centers_of_mass(input_path, input_path_sc, output_dir, labels):
            print(f"Loading NIfTI data from {input_path}")
            nifti_data_roots = nib.load(input_path)
            data_roots = np.array(nifti_data_roots.get_fdata())
            multi_label_image = np.zeros(data_roots.shape)

            nifti_data_sc = nib.load(input_path_sc)
            data_sc = np.array(nifti_data_sc.get_fdata())
            d=[]
            for label in labels:
                print(f"Calculating center of mass for label {label}")
                coords = center_of_mass(data_roots == label)
                if not np.isnan(coords).any():
                    x_center, y_center, z_center = map(round, coords)
                    print(f"Label {label} center of mass found at ({x_center}, {y_center}, {z_center})")
                    multi_label_image[x_center, y_center, z_center] = label
                    d.append(z_center)
                else:
                    warnings.warn(f"No center of mass found for label {label}. It may be absent in the image.")

            label_str = "_".join(map(str, labels))
            data_sc_cropped = np.zeros_like(data_sc)
            data_sc_cropped[:, :, d[1] -4: d[0]+8] = data_sc[:, :, d[1] -4: d[0]+8]
            print("Crop on SC from" , d[1] -4,"to" ,d[0] +8)

            input_filename = os.path.basename(input_path).split('.')[0]
            output_filename = f"{input_filename}_c{label_str}.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            new_nifti = nib.Nifti1Image(multi_label_image, nifti_data_roots.affine)
            nib.save(new_nifti, output_path)
            print(f"NIfTI rootlets file saved at: {output_path}")

            input_filename_sc = os.path.basename(input_path_sc).split('.')[0]
            output_filename_sc = f"{input_filename_sc}_cropped.nii.gz"
            output_path_sc = os.path.join(output_dir, output_filename_sc)
            new_nifti_sc = nib.Nifti1Image(data_sc_cropped, nifti_data_sc.affine)
            nib.save(new_nifti_sc, output_path_sc)
            print(f"NIfTI sc cropped file saved at: {output_path_sc}")
            
        #Dilate spinal cord masks
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses01_sc.nii.gz -dilate 2 -shape disk -dim 2 -o {output_folder}/Image_ses01_sc_dil.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses02_sc.nii.gz -dilate 2 -shape disk -dim 2 -o {output_folder}/Image_ses02_sc_dil.nii.gz", shell=True)
        
        # Calculate centers of mass
        print("Calculating centers of mass for specified cervical levels.")
        calculate_centers_of_mass(f"{output_folder}/Image_ses01_rootlets.nii.gz", f"{output_folder}/Image_ses01_sc_dil.nii.gz", output_folder, [level_up, level_down])
        calculate_centers_of_mass(f"{output_folder}/Image_ses02_rootlets.nii.gz", f"{output_folder}/Image_ses02_sc_dil.nii.gz", output_folder, [level_up, level_down])
        
        # Crop images based on spinal cord segmentation with dilation 100x100x2
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses01.nii.gz -m {output_folder}/Image_ses01_sc_dil_cropped.nii.gz -o {output_folder}/Image_ses01_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses02.nii.gz -m {output_folder}/Image_ses02_sc_dil_cropped.nii.gz -o {output_folder}/Image_ses02_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses01_rootlets.nii.gz -m {output_folder}/Image_ses01_sc_dil_cropped.nii.gz -o {output_folder}/Image_ses01_rootlets_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses02_rootlets.nii.gz -m {output_folder}/Image_ses02_sc_dil_cropped.nii.gz -o {output_folder}/Image_ses02_rootlets_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses01_sc_dil.nii.gz -m {output_folder}/Image_ses01_sc_dil_cropped.nii.gz -o {output_folder}/Image_ses01_sc_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses02_sc_dil.nii.gz -m {output_folder}/Image_ses02_sc_dil_cropped.nii.gz -o {output_folder}/Image_ses02_sc_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses01_rootlets_c{level_up}_{level_down}.nii.gz -m {output_folder}/Image_ses01_sc_dil_cropped.nii.gz -o {output_folder}/Image_ses01_rootlets_c{level_up}_{level_down}_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses02_rootlets_c{level_up}_{level_down}.nii.gz -m {output_folder}/Image_ses02_sc_dil_cropped.nii.gz -o {output_folder}/Image_ses02_rootlets_c{level_up}_{level_down}_crop.nii.gz -dilate {dilation_size}", shell=True)
        
        # # Segmentation of MS lesions
        # subprocess.run(f"sct_deepseg -i {output_folder}/Image_ses01_crop.nii.gz -task seg_ms_lesion_mp2rage -o {output_folder}/Image_ses01_ms-lesion.nii.gz", shell=True)
        # subprocess.run(f"sct_deepseg -i {output_folder}/Image_ses02_crop.nii.gz -task seg_ms_lesion_mp2rage -o {output_folder}/Image_ses02_ms-lesion.nii.gz", shell=True)
        
        # Combine spinal cord and rootlets for binary dilation
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses01_sc_crop.nii.gz -add {output_folder}/Image_ses01_rootlets_crop.nii.gz -o {output_folder}/Image_ses01_sc_rootlets.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses02_sc_crop.nii.gz -add {output_folder}/Image_ses02_rootlets_crop.nii.gz -o {output_folder}/Image_ses02_sc_rootlets.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses01_sc_rootlets.nii.gz -bin 0.5 -o {output_folder}/Image_ses01_sc_rootlets_bin.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses02_sc_rootlets.nii.gz -bin 0.5 -o {output_folder}/Image_ses02_sc_rootlets_bin.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses01_sc_rootlets_bin.nii.gz -dilate 1 -o {output_folder}/Image_ses01_sc_rootlets_dil.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses02_sc_rootlets_bin.nii.gz -dilate 1 -o {output_folder}/Image_ses02_sc_rootlets_dil.nii.gz", shell=True)
        
        # Register session images
        print("Starting inter-session anisotropic registration")
        subprocess.run(f"sct_register_multimodal -i {output_folder}/Image_ses01_crop.nii.gz -d {output_folder}/Image_ses02_crop.nii.gz -ilabel {output_folder}/Image_ses01_rootlets_c{level_up}_{level_down}_crop.nii.gz -dlabel {output_folder}/Image_ses02_rootlets_c{level_up}_{level_down}_crop.nii.gz -ofolder label_{output_folder} -param step=0,type=label,dof=Tx_Ty_Tz_Rz", shell=True)
        subprocess.run(f"sct_register_multimodal -i {output_folder}/Image_ses01_crop.nii.gz -d {output_folder}/Image_ses02_crop.nii.gz -ilabel {output_folder}/Image_ses01_rootlets_c{level_up}_{level_down}_crop.nii.gz -dlabel {output_folder}/Image_ses02_rootlets_c{level_up}_{level_down}_crop.nii.gz -iseg {output_folder}/Image_ses01_sc_crop.nii.gz -dseg {output_folder}/Image_ses02_sc_crop.nii.gz -ofolder sc_{output_folder} -param step=0,type=label,dof=Tx_Ty_Tz_Rz:step=1,type=im,algo=affine,metric=MI,iter=30:step=2,type=seg,algo=affine,metric=CC,iter=5:step=3,type=im,algo=syn,metric=MI,iter=50", shell=True)
        subprocess.run(f"sct_register_multimodal -i {output_folder}/Image_ses01_crop.nii.gz -d {output_folder}/Image_ses02_crop.nii.gz -ilabel {output_folder}/Image_ses01_rootlets_c{level_up}_{level_down}_crop.nii.gz -dlabel {output_folder}/Image_ses02_rootlets_c{level_up}_{level_down}_crop.nii.gz -iseg {output_folder}/Image_ses01_sc_rootlets_dil.nii.gz -dseg {output_folder}/Image_ses02_sc_rootlets_dil.nii.gz -ofolder sc-label_{output_folder} -param step=0,type=label,dof=Tx_Ty_Tz_Rz:step=1,type=im,algo=affine,metric=MI,iter=30:step=2,type=seg,algo=affine,metric=CC,iter=5:step=3,type=im,algo=syn,metric=MI,iter=50", shell=True)

        # # Apply transformation to MS lesion segmentation
        # print("Applying transformation to MS lesion segmentation.")
        # subprocess.run(f"sct_apply_transfo -i {output_folder}/Image_ses01_ms-lesion.nii.gz -d {output_folder}/Image_ses02_crop.nii.gz -w {output_folder}/warp_Image_ses01_crop2Image_ses02_crop.nii.gz -o {output_folder}/Image_ses01_ms-lesion_warped_2_ses02.nii.gz -x nn", shell=True)
    
    else: #for isotropic data
        dilation_size = "40x40x12" 

        # Function to calculate centers of mass for specified labels
        def calculate_centers_of_mass(input_path, output_dir, labels):
            print(f"Loading NIfTI data from {input_path}")
            nifti_data_roots = nib.load(input_path)
            data_roots = np.array(nifti_data_roots.get_fdata())
            
            multi_label_image = np.zeros(data_roots.shape)
            for label in labels:
                print(f"Calculating center of mass for label {label}")
                coords = center_of_mass(data_roots == label)
                if not np.isnan(coords).any():
                    x_center, y_center, z_center = map(round, coords)
                    print(f"Label {label} center of mass found at ({x_center}, {y_center}, {z_center})")
                    multi_label_image[x_center, y_center, z_center] = label
                else:
                    warnings.warn(f"No center of mass found for label {label}. It may be absent in the image.")

            label_str = "_".join(map(str, labels))
            input_filename = os.path.basename(input_path).split('.')[0]
            output_filename = f"{input_filename}_c{label_str}.nii.gz"
            output_path = os.path.join(output_dir, output_filename)

            new_nifti = nib.Nifti1Image(multi_label_image, nifti_data_roots.affine)
            nib.save(new_nifti, output_path)
            print(f"NIfTI file saved at: {output_path}")
            
        #Dilate spinal cord masks 
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses01_sc.nii.gz -dilate 1 -o {output_folder}/Image_ses01_sc_dil.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses02_sc.nii.gz -dilate 1 -o {output_folder}/Image_ses02_sc_dil.nii.gz", shell=True)

        # Crop images based on spinal cord segmentation with dilation
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses01.nii.gz -m {output_folder}/Image_ses01_sc_dil.nii.gz -o {output_folder}/Image_ses01_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses02.nii.gz -m {output_folder}/Image_ses02_sc_dil.nii.gz -o {output_folder}/Image_ses02_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses01_rootlets.nii.gz -m {output_folder}/Image_ses01_sc_dil.nii.gz -o {output_folder}/Image_ses01_rootlets_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses02_rootlets.nii.gz -m {output_folder}/Image_ses02_sc_dil.nii.gz -o {output_folder}/Image_ses02_rootlets_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses01_sc_dil.nii.gz -m {output_folder}/Image_ses01_sc_dil.nii.gz -o {output_folder}/Image_ses01_sc_crop.nii.gz -dilate {dilation_size}", shell=True)
        subprocess.run(f"sct_crop_image -i {output_folder}/Image_ses02_sc_dil.nii.gz -m {output_folder}/Image_ses02_sc_dil.nii.gz -o {output_folder}/Image_ses02_sc_crop.nii.gz -dilate {dilation_size}", shell=True)
        
        # # Segmentation of MS lesions
        # subprocess.run(f"sct_deepseg -i {output_folder}/Image_ses01_crop.nii.gz -task seg_ms_lesion_mp2rage -o {output_folder}/Image_ses01_ms-lesion.nii.gz", shell=True)
        # subprocess.run(f"sct_deepseg -i {output_folder}/Image_ses02_crop.nii.gz -task seg_ms_lesion_mp2rage -o {output_folder}/Image_ses02_ms-lesion.nii.gz", shell=True)
        
        print("Calculating centers of mass for specified cervical levels.")
        calculate_centers_of_mass(f"{output_folder}/Image_ses01_rootlets_crop.nii.gz", output_folder, [level_up, level_down])
        calculate_centers_of_mass(f"{output_folder}/Image_ses02_rootlets_crop.nii.gz", output_folder, [level_up, level_down])

        # Combine spinal cord and rootlets for binary dilation
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses01_sc_crop.nii.gz -add {output_folder}/Image_ses01_rootlets_crop.nii.gz -o {output_folder}/Image_ses01_sc_rootlets.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses02_sc_crop.nii.gz -add {output_folder}/Image_ses02_rootlets_crop.nii.gz -o {output_folder}/Image_ses02_sc_rootlets.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses01_sc_rootlets.nii.gz -bin 0.5 -o {output_folder}/Image_ses01_sc_rootlets_bin.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses02_sc_rootlets.nii.gz -bin 0.5 -o {output_folder}/Image_ses02_sc_rootlets_bin.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses01_sc_rootlets_bin.nii.gz -dilate 1 -o {output_folder}/Image_ses01_sc_rootlets_dil.nii.gz", shell=True)
        subprocess.run(f"sct_maths -i {output_folder}/Image_ses02_sc_rootlets_bin.nii.gz -dilate 1 -o {output_folder}/Image_ses02_sc_rootlets_dil.nii.gz", shell=True)

        # Register session images
        print("Starting inter-session registration using centers of mass and binary dilation images.")
        subprocess.run(f"sct_register_multimodal -i {output_folder}/Image_ses01_crop.nii.gz -d {output_folder}/Image_ses02_crop.nii.gz -ilabel {output_folder}/Image_ses01_rootlets_crop_c{level_up}_{level_down}.nii.gz -dlabel {output_folder}/Image_ses02_rootlets_crop_c{level_up}_{level_down}.nii.gz -ofolder label_{output_folder} -param step=0,type=label,dof=Tx_Ty_Tz_Rz", shell=True)
        subprocess.run(f"sct_register_multimodal -i {output_folder}/Image_ses01_crop.nii.gz -d {output_folder}/Image_ses02_crop.nii.gz -ilabel {output_folder}/Image_ses01_rootlets_crop_c{level_up}_{level_down}.nii.gz -dlabel {output_folder}/Image_ses02_rootlets_crop_c{level_up}_{level_down}.nii.gz -iseg {output_folder}/Image_ses01_sc_crop.nii.gz -dseg {output_folder}/Image_ses02_sc_crop.nii.gz -ofolder sc_{output_folder} -param step=0,type=label,dof=Tx_Ty_Tz_Rz:step=1,type=im,algo=affine,metric=MI,iter=30:step=2,type=seg,algo=slicereg,metric=CC,iter=2:step=3,type=im,algo=syn,metric=MI,iter=50", shell=True)
        subprocess.run(f"sct_register_multimodal -i {output_folder}/Image_ses01_crop.nii.gz -d {output_folder}/Image_ses02_crop.nii.gz -ilabel {output_folder}/Image_ses01_rootlets_crop_c{level_up}_{level_down}.nii.gz -dlabel {output_folder}/Image_ses02_rootlets_crop_c{level_up}_{level_down}.nii.gz -iseg {output_folder}/Image_ses01_sc_rootlets_dil.nii.gz -dseg {output_folder}/Image_ses02_sc_rootlets_dil.nii.gz -ofolder sc-label_{output_folder} -param step=0,type=label,dof=Tx_Ty_Tz_Rz:step=1,type=im,algo=affine,metric=MI,iter=30:step=2,type=seg,algo=slicereg,metric=CC,iter=2:step=3,type=im,algo=syn,metric=MI,iter=50", shell=True)
        
        # # Apply transformation to MS lesion segmentation
        # print("Applying transformation to MS lesion segmentation.")
        # subprocess.run(f"sct_apply_transfo -i {output_folder}/Image_ses01_ms-lesion.nii.gz -d {output_folder}/Image_ses02_crop.nii.gz -w {output_folder}/warp_Image_ses01_crop2Image_ses02_crop.nii.gz -o {output_folder}/Image_ses01_ms-lesion_warped_2_ses02.nii.gz -x nn", shell=True)
    
    print("Processing completed successfully.")

# Argument parser for command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inter-session registration script for spinal rootlets, supporting both isotropic and anisotropic MRI data.")
    parser.add_argument("-ses01", type=str, help="Path to the session 1 MRI image.")
    parser.add_argument("-ses02", type=str, help="Path to the session 2 MRI image.")
    parser.add_argument("-level_up", type=int, help="Upper cervical level for calculating the center of mass.")
    parser.add_argument("-level_down", type=int, help="Lower cervical level for calculating the center of mass.")
    parser.add_argument("-output_folder", type=str, help="Output folder for processed images.")
    parser.add_argument("-anisotropic", action="store_true", help="Flag to process anisotropic data (default: False).")
    args = parser.parse_args()

    main(args.ses01, args.ses02, args.level_up, args.level_down, args.output_folder, args.anisotropic)
