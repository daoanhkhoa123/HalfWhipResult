import pandas as pd 
from numpy.typing import NDArray
from typing import Tuple
from torch.utils.data import DataLoader, Dataset

METADATA_PATH = r""

from model.audio import load_audio

class VSAVSmallDataset(Dataset):
    def __init__(self, metadata_path = METADATA_PATH) -> None:
        super().__init__()
        self.metadata = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index) -> Tuple[NDArray, int,int]:
        path = self.metadata.iloc[index]["path"]
        speaker = self.metadata.iloc[index]["speaker_id_num"]
        att_type = self.metadata.iloc[index]["att_type_id"]

        return load_audio(path), speaker, att_type

