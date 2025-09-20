import argparse
from tqdm import tqdm
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from model.model import ModelDimensions,Whisper1
from model.loss import CLIPLoss
from datamodule.dataloader import VSAVSmallDataset
from scheduler.cosine_scheduler import cosine_schedule_with_warmup

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

@dataclass
class Traintest_config:
    metadata_path:str
    prefix:str
    batch_size:int
    epochs:int
    # not yet supported multiple gpu
    n_gpu:int
    _device:str
    lr:float
    eps:float
    weight_decay:float

    @property
    def device(self):
        return torch.device(self._device)

def train(model_dimensions:ModelDimensions, config:Traintest_config):
    train_dataloader = DataLoader(VSAVSmallDataset(config.metadata_path, config.prefix), config.batch_size, True)
    model = Whisper1(model_dimensions)
    model = model.to(config.device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    step_total = len(train_dataloader) * config.epochs
    n_warmup_steps = int(0.2 * step_total)
    scheduler = cosine_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup_steps, num_traning_steps=step_total)

    # skip gradent accumulaton steps, multiple gpu support
    model.zero_grad()
    for epoch in range(int(config.epochs)):
        for step, batch in tqdm(enumerate(train_dataloader)):
            audio, speaker, att_type = batch
            audio = audio.to(config.device)
            speaker = speaker.to(config.device)
            att_type = att_type.to(config.device)
            speaker_embedding, spoofing_logits = model(audio)

            speaker_loss = CLIPLoss(model, speaker_embedding)
            spoof_loss = torch.nn.functional.cross_entropy(spoofing_logits, att_type)
            loss = (speaker_loss + spoof_loss) /2
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.zero_grad()


def setup():
    parser = argparse.ArgumentParser(description="My training script")

    # ---- Training config ----
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--device", dest="_device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # ---- Model dimensions ----
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_audio_ctx", type=int, default=1500)
    parser.add_argument("--n_audio_state", type=int, default=512)
    parser.add_argument("--n_audio_head", type=int, default=8)
    parser.add_argument("--n_audio_layer", type=int, default=12)
    parser.add_argument("--n_vocab", type=int, default=10000)
    parser.add_argument("--use_positionalencoding", action="store_true")
    parser.add_argument("--n_spkemb_layers", type=int, default=3)

    return parser


def get_config(parser:argparse.ArgumentParser):
    args = parser.parse_args()

    model_cfg = ModelDimensions(
        n_mels=args.n_mels,
        n_audio_ctx=args.n_audio_ctx,
        n_audio_state=args.n_audio_state,
        n_audio_head=args.n_audio_head,
        n_audio_layer=args.n_audio_layer,
        n_vocab=args.n_vocab,
        use_positionalencoding=args.use_positionalencoding,
        n_spkemb_layers=args.n_spkemb_layers,
    )

    train_cfg = Traintest_config(
        metadata_path=args.metadata_path,
        prefix=args.prefix,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_gpu=args.n_gpu,
        _device=args._device,
        lr=args.lr,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    return model_cfg, train_cfg

def setup_logger():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename =f"traintorch_testsmall_{current_time}.txt"

    logging.basicConfig(filename=filename,
                        filemode="w", level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Logger Initialized")
    return filename

def log_configs(model_cfg, train_cfg):
    logging.info("===== Training Configuration =====")
    for k, v in asdict(train_cfg).items():
        logging.info(f"{k}: {v}")

    logging.info("===== Model Dimensions =====")
    for k, v in asdict(model_cfg).items():
        logging.info(f"{k}: {v}")

if __name__ == "__main__":
    parser = setup()
    model_cfg, train_cfg = get_config(parser)

    log_file = setup_logger()
    log_configs(model_cfg, train_cfg)
    logging.info(f"Logging to {log_file}")

    train(model_cfg, train_cfg)