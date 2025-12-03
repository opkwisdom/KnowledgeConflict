from omegaconf import OmegaConf
import yaml
import sys
import glob
import os
from utils import load_qa_dataset

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    print(config)

    x = glob.glob(os.path.join(config.data.data_dir, '*.jsonl'))
    print(x)
    # dataset = load_qa_dataset(config.data.data_dir)