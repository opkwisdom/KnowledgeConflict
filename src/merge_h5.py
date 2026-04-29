import h5py
import os

def merge_h5_files(input_dir: str, output_file: str):
    subdirs = ["short_out_0", "short_out_1", "short_out_2", "short_out_3"]
    with h5py.File(output_file, 'w') as merged_h5:
        for subdir in subdirs:
            subdir_path = os.path.join(input_dir, subdir)
            if not os.path.isdir(subdir_path):
                print(f"Directory {subdir_path} does not exist. Skipping.")
                continue
            
            for filename in os.listdir(subdir_path):
                if filename.endswith('.h5'):
                    file_path = os.path.join(subdir_path, filename)
                    print(f"Merging {file_path} into {output_file}...")
                    with h5py.File(file_path, 'r') as h5_file:
                        for key in h5_file.keys():
                            if key in merged_h5:
                                print(f"Warning: Key {key} already exists in merged file. Overwriting.")
                                del merged_h5[key]
                            h5_file.copy(key, merged_h5)

def main():
    input_dir = "/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama"
    output_file = "/workspaces/kvzip_nlplab/checkpoint/genloss_precompute_table/llama/hotpotqa-w_short_loss_precompute_table.h5"
    merge_h5_files(input_dir, output_file)


if __name__ == "__main__":
    main()