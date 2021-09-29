import numpy as np
from torch.utils.data import Dataset, DataLoader, dataloader

class HWMPaddedDataset(Dataset):
    def __init__(self, dataset, partition="train", train_ratio=0.8):
        super().__init__()
        self.partition="train"

        assert dataset["observations"].shape[0] == dataset["actions"].shape[0], \
            "Dataset 'observations' and 'actions' samples do not match."
        
        assert dataset["padded_length"] == dataset["observations"].shape[1] == dataset["actions"].shape[1], \
            "Dataset 'observations' and 'actions' padded lengths do not match."

        ratio = train_ratio if partition == "train" else (1. - train_ratio)
        n_episodes = int(dataset["n_episodes"] * ratio)
        
        if self.partition == "train":
            self.observations = dataset["observations"][:n_episodes]
            self.actions = dataset["actions"][:n_episodes]
            # self.terminals = dataset["terminals"][:n_episodes]
            self.depad_masks = dataset["depad_masks"][:n_episodes]
            self.depad_slices = dataset["depad_slices"][:n_episodes]

        else:
            self.observations = dataset["observations"][-n_episodes:]
            self.actions = dataset["actions"][-n_episodes:]
            # self.terminals = dataset["terminals"][-n_episodes:]
            self.depad_masks = dataset["depad_masks"][-n_episodes:]
            self.depad_slices = dataset["depad_slices"][-n_episodes:]
    
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    
    def __len__(self):
        return self.observations.shape[0]
    
    def __getitem__(self, index):
        return self.observations[index], \
               self.actions[index], \
               self.depad_masks[index], \
               self.depad_slices[index]

def full_padded_data_loader_hwm(dataset_path, batch_size, num_workers=4):
    dataset = np.load(dataset_path, allow_pickle=True)

    obs_shape = dataset["observation_shape"]
    act_shape = dataset["act_shape"]

    train_loader = DataLoader(
        HWMPaddedDataset(dataset, partition="train", train_ratio=0.8),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # TODO: maybe also retun the sequence for evaluation of the model
    return train_loader, obs_shape.tolist(), act_shape.tolist()
