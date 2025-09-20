import pandas as pd 
from torch import Tensor
from typing import Tuple
from torch.utils.data import Dataset

METADATA_PATH = r""

from model.audio import load_audio, log_mel_spectrogram
import os

class VSAVSmallDataset(Dataset):
    def __init__(self, df_path:str = METADATA_PATH, prefix:str="", bonafide_id:int=1) -> None:
        """
        df_path:s path to csv of metadata
        prefix: the folder of VASV dataaset
        bonafide_id: since we only need to check if bonafide or not, check id in att_type_labels.csv
        """
        super().__init__()
        self.metadata = pd.read_csv(df_path)
        self.prefix = prefix
        self.bondafide_id = bonafide_id

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index) -> Tuple[Tensor, int,int]:
        path = self.metadata.iloc[index]["path"]
        path = os.path.join(self.prefix, path)
        audio = log_mel_spectrogram(load_audio(path))
        
        speaker = self.metadata.iloc[index]["speaker_id_num"]
        att_type = self.metadata.iloc[index]["att_type_id"]
        att_type = 0  if att_type == self.bondafide_id else 1 # if bonafide, then 0, otherwise 1
        return audio, speaker, att_type
    

