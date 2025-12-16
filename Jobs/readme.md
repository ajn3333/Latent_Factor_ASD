# HPC Job Scripts

This directory serves as the **main entry point of the repository**. It contains the shell job scripts used to run the analysis pipeline on a **High Performance Computing (HPC) cluster**.

Each job file corresponds to a major processing or analysis step, from preprocessing to downstream modeling.

---

## External Dependencies

Several external repositories are required to run the full pipeline:

### CBIG (Preprocessing)
- The preprocessing pipeline relies on the **CBIG repository** for fMRI preprocessing.
- Appropriate CBIG dependencies must be installed and configured.
- The CBIG codebase can be found on the **Thomas Yeo Lab GitHub page**.

### MIND (Similarity Matrices)
- Generation of **MIND similarity matrices** requires the **MIND repository**.
- This repository is maintained on the **isebenius GitHub page**.

---

## Contents

### `preproc_CBIG.job`
This is the **main preprocessing job script**.

**Purpose**
- Runs the preprocessing pipeline on raw MRI data.

**Required inputs**
- `y_values.txt`  
  A text file containing a list of participant folders. Each participant directory must follow the **BIDS structure**, including:
  ```
  sub-XX/
    ├── anat/
    └── func/
  ```

- `x_values.txt`  
  A text file containing auxiliary information associated with each participant.

**ABIDE II specific note**
- For the **ABIDE II dataset**, `x_values.txt` specifies the **collection site** for each participant.
- This information is used to select the appropriate **slice timing** during the slice timing correction step.

---

### `parcel.job`
This job script is used to **parcellate structural MRI data**.

**Purpose**
- Applies the **Schaefer 400 parcellation** to the structural data for each participant.

**Output**
- Parcellated structural representations for downstream analyses.

---


### `MIND.job`
This job script computes **MIND similarity matrices**.

**Purpose**
- Calculates the **MIND (Morphometric INverse Divergence)** similarity matrix for each participant.
- These matrices can be used as inputs for subsequent analyses.

---

### `PLDA.job`
This is the **main Polar-LDA analysis job**.

**Purpose**
- Runs the **Polar Latent Dirichlet Allocation (Polar-LDA)** model.

**Important configuration**
- You must specify the correct input directory containing either:
  - Functional connectivity matrices, or
  - Similarity matrices

- This path must be set inside:
  ```
  PLDA/polar_LDA.py
  ```

---

## Notes
- All job scripts are designed to be submitted to an HPC scheduler.
- Ensure that paths, input files, and environment modules are properly configured before submission.

