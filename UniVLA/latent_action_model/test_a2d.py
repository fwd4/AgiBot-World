import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import importlib
from prismatic.vla.datasets.dataset_a2d import LAMStage1Dataset, setup_distributed

# Debugging helper function
def inspect_dataset_item(item):
    print("\nDataset Item Contents:")
    print(f"Video Shape: {item['videos'].shape}")
    print(f"Random Video Length: {item['random_video_len']}")
    print(f"Control Frequency: {item['ctrl_freqs']}")

# Add config directory to system path
file_path = Path("latent_action_model/config/lam-a2d.py")
sys.path.insert(0, str(file_path.parent))
data_cfg = importlib.import_module(file_path.stem)
dataset_args=data_cfg.DatasetArguments()
data_training_args=data_cfg.DataTrainingArguments()

setup_distributed()

# Create dataset instance
dataset = LAMStage1Dataset(
    # base params
    label_file_dir=dataset_args.meta_json_dir,
    data_root_dir=dataset_args.data_root_dir,
    valid_episode_txt=dataset_args.valid_episode_txt,
    world_size=1, #dist.get_world_size(),
    rank_id=0, #dist.get_rank(),
    online_process_mp_cnt=dataset_args.online_process_mp_cnt,
    # a2d params
    is_train=True,
    image_size=data_training_args.force_image_size,
    pad2square=data_training_args.pad2square,
    normalize_type=data_training_args.normalize_type,

    # is_train=True,
    # image_size=448,
    # pad2square=False,
    # normalize_type="imagenet",
    # # Add your base dataset parameters here
    # data_dir="path/to/your/data",  # Replace with actual data path
)

dataset.generate_task_infos(
    dataset_cfg=dataset_args.dataset_task_cfg,
    task_episode_processors_cfg=dataset_args.episode_processors,
    task_dataset_processors_cfg=dataset_args.dataset_processors,
    task_runtime_processors_cfg=dataset_args.runtime_processors,
    shuffle=True,
    debug_one_episode=False,
    )

# Create dataloader for iteration
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2
)

# Debug loop
def debug_dataset():
    try:
        # Test single item access
        print("Testing single item access:")
        first_item = dataset[0]
        import pdb; pdb.set_trace()
        inspect_dataset_item(first_item)
        
        # Test batch iteration
        print("\nTesting batch iteration:")
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"Batch video shape: {batch['videos'].shape}")
            print(f"Batch ctrl_freqs: {batch['ctrl_freqs']}")
            
            # Only check first few batches
            if batch_idx >= 2:
                break
                
    except Exception as e:
        print(f"Error during dataset debugging: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset()