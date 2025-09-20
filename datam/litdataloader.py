import lightning as lit
import os
import torch
from torch.utils.data import DataLoader
from .dataloader import VSAVSmallDataset
from ..model.audio import pad_or_trim_tensor

class VSAVDataModule(lit.LightningDataModule):
    def __init__(self, metadata_path: str, prefix: str = "", batch_size=64,**dataloader_kwargs) -> None:
        super().__init__()
        self.metadata_path = metadata_path
        self.prefix = prefix
        self.batch_size = batch_size
        self.kwargs = dataloader_kwargs

    def setup(self, stage: str) -> None:
        if stage == "small_fit":
            path = os.path.join(self.metadata_path, "metadata_small_train.csv")
            self.train_dataset = VSAVSmallDataset(path, self.prefix)

        elif stage == "fit":
            path = os.path.join(self.metadata_path, "metadata_train.csv")
            self.train_dataset = VSAVSmallDataset(path, self.prefix)

        elif stage == "validate":
            path = os.path.join(self.metadata_path, "metadata_val.csv")
            self.val_dataset = VSAVSmallDataset(path, self.prefix)

        elif stage == "test":
            path = os.path.join(self.metadata_path, "metadata_test.csv")
            self.test_dataset = VSAVSmallDataset(path, self.prefix)

        elif stage == "predict":
            path = os.path.join(self.metadata_path, "metadata_test.csv")
            self.predict_dataset = VSAVSmallDataset(path, self.prefix)

        else:
            raise ValueError(f"Unknown stage {stage}")

    @staticmethod
    def collate_fn(batch):
        """Just the collate function warper for audio.pad_or_trim
        
        If you want any behavior changed for audio, change the code inside audio, not here
        """
        audios, speakers, att_types = zip(*batch)
        audios = [pad_or_trim_tensor(audio) for audio in audios]
        audios = torch.stack(audios)
        
        speakers= torch.tensor(speakers, dtype=torch.long)
        att_types = torch.tensor(att_types, dtype= torch.long)
        return audios, speakers, att_types

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn,**self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle= False, collate_fn=self.collate_fn, **self.kwargs)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, **self.kwargs)
