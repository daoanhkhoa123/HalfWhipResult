import argparse

import torch
from torch import nn
from typing import Iterable


from model.model import Whisper1, SpeakerEmbedding, SpoofingClassifier, MultiHeadAttention, ModelDimensions


def test_module(model: nn.Module, input_shape: Iterable[int], batch_size: int = 2) -> None:
    model.eval() 
    device = next(model.parameters()).device  

    x = torch.randn((batch_size, *input_shape), device=device)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        y = model(x)
    
    if isinstance(y, tuple):
        for i, out in enumerate(y):
            print(f"Output {i} shape: {out.shape}")
    else:
        print(f"Output shape: {y.shape}")


def setup():
    parser = argparse.ArgumentParser(description="Set up model dimensions for Whisper1")
    
    parser.add_argument("--n_mels", type=int, default=80, help="Number of Mel bins")
    parser.add_argument("--n_audio_ctx", type=int, default=300, help="Audio context length")
    parser.add_argument("--n_audio_state", type=int, default=256, help="Hidden state dimension")
    parser.add_argument("--n_audio_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_audio_layer", type=int, default=2, help="Number of attention layers")
    parser.add_argument("--use_positionalencoding", type=bool, default=True, help="Use positional encoding")
    parser.add_argument("--n_spkemb_layers", type=int, default=2, help="Number of speaker embedding layers")

    args = parser.parse_args()

    dims = ModelDimensions(
        n_mels=args.n_mels,
        n_audio_ctx=args.n_audio_ctx,
        n_audio_state=args.n_audio_state,
        n_audio_head=args.n_audio_head,
        n_audio_layer=args.n_audio_layer,
        use_positionalencoding=args.use_positionalencoding,
        n_spkemb_layers=args.n_spkemb_layers,
    )

    return dims

if __name__ == "__main__":
    dims = setup()
    print(dims)

    encoder = Whisper1(dims).encoder
    speaker_emb = SpeakerEmbedding(dims.n_audio_state, dims.n_audio_head, dims.n_spkemb_layers)
    spoof_cls = SpoofingClassifier(dims.n_audio_state)
    full_model = Whisper1(dims)

    models = [
        ("AudioEncoder", encoder, (dims.n_mels, dims.n_audio_ctx)),
        ("SpeakerEmbedding", speaker_emb, (dims.n_audio_ctx, dims.n_audio_state)),
        ("SpoofingClassifier", spoof_cls, (dims.n_audio_ctx, dims.n_audio_state)),
        ("Whisper1 (full)", full_model, (dims.n_mels, dims.n_audio_ctx))
    ]

    for name, model, input_shape in models:
        print(f"=== Testing {name} ===")
        test_module(model, input_shape, batch_size=2)
        print()
