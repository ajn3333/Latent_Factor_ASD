import sys
import os
import numpy as np
from MIND import compute_MIND

def main(path_to_surf_dir, features, parcellation, output_dir):
    # Returns a dataframe of regions X regions containing the final MIND network.
    MIND = compute_MIND(path_to_surf_dir, features, parcellation)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the MIND result as a plain text file
    output_file = os.path.join(output_dir, f"{os.path.basename(path_to_surf_dir)}.txt")
    np.savetxt(output_file, MIND.values)
    print(f"MIND result saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <base_path> <subject_id> <parcellation> <output_dir>")
        sys.exit(1)

    base_path = sys.argv[1]  # Base path to surfer directory
    subject_id = sys.argv[2]  # Subject ID
    parcellation = sys.argv[3]  # Parcellation to use
    output_dir = sys.argv[4]  # Output directory for saving the result
    
    path_to_surf_dir = os.path.join(base_path, subject_id)  # Concatenate base path with subject ID
    features = ['CT','MC','Vol','SD','SA']  # Features to include

    main(path_to_surf_dir, features, parcellation, output_dir)