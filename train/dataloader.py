import pandas as pd 
from torch import Tensor
from typing import Tuple
from torch.utils.data import DataLoader, Dataset

METADATA_PATH = r""

from model.audio import load_audio, log_mel_spectrogram
import os

class VSAVSmallDataset(Dataset):
    def __init__(self, metadata_path:str = METADATA_PATH, prefix:str="") -> None:
        super().__init__()
        self.metadata = pd.read_csv(metadata_path)
        self.prefix = prefix

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index) -> Tuple[Tensor, int,int]:
        path = self.metadata.iloc[index]["path"]
        path = os.path.join(self.prefix, path)
        audio = log_mel_spectrogram(load_audio(path))
        
        speaker = self.metadata.iloc[index]["speaker_id_num"]
        att_type = self.metadata.iloc[index]["att_type_id"]

        return audio, speaker, att_type
    

