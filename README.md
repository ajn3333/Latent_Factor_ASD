# Polar Latent Dirichlet Allocation (Polar-LDA)

This repository contains scripts to run **Polar Latent Dirichlet Allocation (Polar-LDA)** analyses on both **structural** and **functional MRI** data.

The codebase was originally developed to run analyses on the **ABIDE II dataset**. However, with appropriate adaptations to data paths and configuration files, it can theoretically be applied to **any MRI/fMRI dataset** that follows a compatible structure.

---

## Computing Environment

All scripts in this repository are designed to run on a **High Performance Computing (HPC) cluster**. Job submission and execution are handled via scheduler scripts (e.g., SLURM).

---

## External Dependencies

Several external repositories are required to run the full pipeline:

### [CBIG (Preprocessing)](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/preprocessing/CBIG_fMRI_Preproc2016)
- The preprocessing pipeline relies on the **CBIG repository** for fMRI preprocessing.
- Appropriate CBIG dependencies must be installed and configured.
- The CBIG codebase can be found on the **Thomas Yeo Lab GitHub page**.

### [MIND (Similarity Matrices)](https://github.com/isebenius/MIND)
- Generation of **MIND similarity matrices** requires the **MIND repository**.
- This repository is maintained on the **isebenius GitHub page**.


---

## Repository Structure

The repository is organized as follows:

### [/Jobs](/Jobs)
- Main entry directory for the pipeline.
- Contains **SLURM job scripts** used to run preprocessing and all major analysis steps on the HPC cluster.

### [/CBIG_Config](/CBIG_Config) and [/Slice_time_files](/Slice_time_files)
- Contain essential **configuration files** required for preprocessing the **ABIDE II dataset**.
- These files primarily specify the correct **slice timing correction parameters** for each ABIDE II participant.

### [/MIND](/MIND)
- Contains Python code used to compute **MIND similarity matrices** for each participant.

### [/PLDA](/PLDA)
- Contains the main Python implementation for running the **Polar-LDA analysis**.
- The Polar-LDA implementation in this repository is an **adaptation of the original Polar-LDA code developed by the Thomas Yeo Lab**.

---

## Notes
- Path definitions and dataset-specific parameters may need to be updated when applying this pipeline to datasets other than ABIDE II.

