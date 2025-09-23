from typing import Any
import lightning as lit
from torch import Tensor
import torch

from typing import Tuple

from model.whisper import ModelDimensions, Whisper1
from model.loss import CLIPLossCls

class WhipserMixed(lit.LightningModule):
    def __init__(self, model_dims:ModelDimensions) -> None:
        super().__init__()
        self.model = Whisper1(model_dims)
        self.spk_crt = CLIPLossCls()
        self.spf_crt = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        audio, speaker, att_type = batch
        speaker_logits, spoof_logits = self.model(audio)

        spk_loss = self.spk_crt(speaker_logits)
        spf_loss = self.spf_crt(spoof_logits, att_type)
        loss = (spk_loss + spf_loss) /2

        self.log("train/speaker_loss", spk_loss, prog_bar=True)
        self.log("train/spoof_loss", spf_loss, prog_bar=True)
        self.log("train/total_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)