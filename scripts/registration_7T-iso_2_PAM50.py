"""
Fully automated registration of 7T MP2RAGE isotropic to PAM50

Dependencies: 
    SCT v6.5 and later versions 

To install: 
    sct_deepseg seg_sc_contrast_agnostic -install
    sct_deepseg rootlets_t2 -install -custom-url https://github.com/ivadomed/model-spinal-rootlets/releases/download/r20240915/model-spinal-rootlets-mp2rage-r20240915.zip
    sct_deepseg lesion_ms_mp2rage -install

Details of flags: 
    - UNI: Path to the UNI image (MP2RAGE).
    - T1map: Optional path to the T1 map image.
    - output_folder: Output directory for results.
    - template_path: Path to the PAM50 template (T1).
Usage:
    python registration_7T-iso_2_PAM50.py -UNI Image_UNI.nii.gz  -output_folder OUT -T1map [optional] Image_T1map.nii.gz --template_path [optional]

Authors : Nilser Laines Medina
Date    : 2025-04-02

"""
import os
import argparse
import subprocess

def run_command(command):
    """
    Run a shell command and raise an error if it fails.
    """
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"\n!! Command failed: {command}")

def main(UNI, T1map, output_folder, template_path):

    print("\n--- Starting preprocessing pipeline ---")
    os.makedirs(output_folder, exist_ok=True)

    # Convert UNI image to RPI
    run_command(f"sct_image -i {UNI} -setorient RPI -o {output_folder}/Image_UNI.nii.gz")

    # Convert T1map if provided
    if T1map:
        run_command(f"sct_image -i {T1map} -setorient RPI -o {output_folder}/Image_T1map.nii.gz")

    # Segment spinal structures
    run_command(f"sct_deepseg rootlets_t2 -i {output_folder}/Image_UNI.nii.gz -o {output_folder}/Image_UNI_rootlets.nii.gz")
    run_command(f"sct_deepseg spinalcord -i {output_folder}/Image_UNI.nii.gz -o {output_folder}/Image_UNI_sc.nii.gz")

    # Dilate spinal cord mask
    dilation_size = "40x40x12"
    run_command(f"sct_maths -i {output_folder}/Image_UNI_sc.nii.gz -dilate 1 -o {output_folder}/Image_UNI_sc_dil.nii.gz")

    # Crop around spinal cord mask
    run_command(f"sct_crop_image -i {output_folder}/Image_UNI.nii.gz -m {output_folder}/Image_UNI_sc_dil.nii.gz -o {output_folder}/Image_UNI_crop.nii.gz -dilate {dilation_size}")
    run_command(f"sct_crop_image -i {output_folder}/Image_UNI_rootlets.nii.gz -m {output_folder}/Image_UNI_sc_dil.nii.gz -o {output_folder}/Image_UNI_rootlets_crop.nii.gz -dilate {dilation_size}")
    run_command(f"sct_crop_image -i {output_folder}/Image_UNI_sc_dil.nii.gz -m {output_folder}/Image_UNI_sc_dil.nii.gz -o {output_folder}/Image_UNI_sc_crop.nii.gz -dilate {dilation_size}")

    # Segment lesions
    run_command(f"sct_deepseg lesion_ms_mp2rage -i {output_folder}/Image_UNI_crop.nii.gz -o {output_folder}/Image_UNI_ms-lesion.nii.gz")

    # Register to PAM50
    run_command(
        f"sct_register_to_template -i {output_folder}/Image_UNI_crop.nii.gz "
        f"-s {output_folder}/Image_UNI_sc_crop.nii.gz "
        f"-lrootlet {output_folder}/Image_UNI_rootlets_crop.nii.gz "
        f"-ofolder {output_folder}/PAM50"
    )

    # Apply warps
    warp_field = f"{output_folder}/PAM50/warp_anat2template.nii.gz"
    run_command(
        f"sct_apply_transfo -i {output_folder}/Image_UNI_ms-lesion.nii.gz "
        f"-d {template_path} -w {warp_field} "
        f"-o {output_folder}/PAM50/Image_ms-lesion_warped_2_PAM50.nii.gz -x nn"
    )

    if T1map:
        run_command(
            f"sct_apply_transfo -i {output_folder}/Image_T1map.nii.gz "
            f"-d {template_path} -w {warp_field} "
            f"-o {output_folder}/PAM50/T1map_warped_2_PAM50.nii.gz"
        )

    print("\n✅ Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register 7T MP2RAGE isotropic images to the PAM50 spinal cord template."
    )
    parser.add_argument("-UNI", required=True, type=str, help="Path to UNI-non-den image (7T MP2RAGE isotropic).")
    parser.add_argument("-T1map", type=str, default=None, help="Optional path to T1map image (7T MP2RAGE isotropic).")
    parser.add_argument("-output_folder", required=True, type=str, help="Output folder for processed images.")
    parser.add_argument(
        "--template_path",
        type=str,
        default="/home/nlaines/DEV/spinalcordtoolbox/data/PAM50/template/PAM50_t1.nii.gz",
        help="Path to the PAM50 T1 template image. Default: %(default)s"
    )
    args = parser.parse_args()
    main(args.UNI, args.T1map, args.output_folder, args.template_path)
