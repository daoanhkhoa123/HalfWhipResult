import torch
from torch import nn
from typing import Iterable
from model.model import Whisper1, SpeakerEmbedding, SpoofingClassifier, MultiHeadAttention, ModelDimensions

import torch
from torch import nn
from typing import Iterable

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

if __name__ == "__main__":
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=300,
        n_audio_state=256,
        n_audio_head=4,
        n_audio_layer=2,
        use_positionalencoding=True,
        n_spkemb_layers=2
    )

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
