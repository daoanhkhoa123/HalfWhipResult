from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional, Tuple, Iterable

import numpy  as np
import torch
from torch.nn import functional as fn
from torch import Tensor, nn

try:
    from torch.nn.functional import scaled_dot_product_attention
    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention= None
    SDPA_AVAILABLE = False

@dataclass
class ModelDimensions:
    n_mels:int
    n_audio_ctx:int
    n_audio_state:int
    n_audio_head:int
    n_audio_layer:int
    n_vocab : int
    use_positionalencoding:bool
    n_spkemb_layers:int

def sinusoids(length, chanels, maxa_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert chanels % 2 ==0
    log_timescale_increment = np.log(maxa_timescale)/(chanels//2-1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(chanels//2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state:int, n_head:int) -> None:
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out =  nn.Linear(n_state, n_state)

    def forward(self, input:Tensor, xa:Optional[Tensor] = None,
                mask:Optional[Tensor]= None, kv_cache:Optional[dict] = None
                ):
        q= self.query(input)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(input if xa is None else xa)
            v = self.value(input if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k= kv_cache[self.key]
            v=kv_cache[self.value]

        wv, qk= self.qkv_attention(q,k,v,mask)
        return self.out(wv), qk

    def qkv_attention(self, q:Tensor, k:Tensor, v:Tensor, mask:Optional[Tensor]= None
                    )-> Tuple[Tensor, Optional[Tensor]]:        
        _, n_ctx, n_state = q.shape
        scale = (n_state//self.n_head)     ** -0.25
        # [batch, seq_len, hidden_dim] -> [batch, n_head, seq_len, head_dim]
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0,2,1,3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0,2,1,3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0,2,1,3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa and scaled_dot_product_attention is not None:
            a = scaled_dot_product_attention(q,k,v, is_causal=mask is not None and n_ctx>1)
            # 0 2 1 3 -> n_batch, seq_len, n_head, head_dim -> n_batch, seq_len, hid_dim
            out = a.permute(0,2,1,3).flatten(start_dim=2)
            qk=None

        else:
            # qk is n_batch, n_head, seq_len, seq_len
            qk=(q*scale) @ (k*scale).transpose(-1,-2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = fn.softmax(qk, dim=-1, dtype=q.dtype)
            out = (w@v).permute(0,2,1,3).flatten(start_dim=2)
            qk=qk.detach()

        return out, qk

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state:int, n_head:int, cross_attention:bool = False) -> None:
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        # get key from encoder,cross attetion is only used for decoder
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln=nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(nn.Linear(n_state, n_mlp),
                                 nn.GELU(),
                                 nn.Linear(n_mlp, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, input:Tensor, xa:Optional[Tensor] = None,
                mask:Optional[Tensor] = None, kv_cache:Optional[dict] = None):
        input = input+ self.attn(self.attn_ln(input), mask=mask, kv_cache=kv_cache)[0]
        
        if self.cross_attn is not None and self.cross_attn_ln is not None:
            input = input+ self.cross_attn(self.cross_attn_ln(input), xa, kv_cache=kv_cache)[0]

        input = input + self.mlp(self.mlp_ln(input))
        return input
        
class AudioEncoder(nn.Module):
    def __init__(self, n_mels:int, n_ctx:int,
                 n_state:int, n_head:int, n_layer:int,
                 use_positionencoding:bool=True) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(n_mels, n_state, 
                               kernel_size=3, padding=1),
                               nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv1d(n_state, n_state, 
                               kernel_size=3, stride=2, padding=1),
                               nn.GELU())
        self.use_positionencoding = use_positionencoding
        if use_positionencoding:
            self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        # self.blocks:Iterable[ResidualAttentionBlock] = nn.ModuleList(
        #     [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        # )  
        # type: ignore[assignment], this pylance static error checking is so stupid omg

        # since we omit the decoder, this would be easier to read
        self.blocks = nn.Sequential(*[ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, input:Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        input = self.conv1(input)
        input = self.conv2(input)
        # n_bacth, seq_len, n_mels
        input = input.permute(0, 2, 1)

        if self.use_positionencoding:
            assert input.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            input = input+ self.positional_embedding
        
        input = self.blocks(input)
        input = self.ln_post(input)
        return input

class SpeakerEmbedding(nn.Module):
    def __init__(self, n_state, n_head, n_layer) -> None:
        super().__init__()

        self.blocks = nn.Sequential(*[ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])

    def forward(self, input:Tensor):
        input = self.blocks(input)
        return torch.mean(input, dim=1, dtype=input.dtype)

class SpoofingClassifier(nn.Module):
    def __init__(self, n_state) -> None:
        super().__init__()
        self.blocks = nn.Sequential(nn.Linear(n_state, n_state//2),
                                    nn.GELU(),
                                    nn.Linear(n_state//2, n_state//4),
                                    nn.GELU(),
                                    nn.Linear(n_state//4, 2))
    def forward(self, input:Tensor):
        return self.blocks(input)

class Whisper1(nn.Module):
    def __init__(self, dims:ModelDimensions) -> None:
        super().__init__()    
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            self.dims.use_positionalencoding
        )

        self.speaker_embedding = SpeakerEmbedding(self.dims.n_audio_state, self.dims.n_audio_head, self.dims.n_spkemb_layers)
        self.spoofing_classifier = SpoofingClassifier(self.dims.n_audio_state)
        self.logit_scale = nn.Parameter(torch.ones([]) *  np.log(1/0.07))

    def embed_audio(self, mel:Tensor):
        return self.encoder(mel)

    def forward(self, mel:Tensor)-> Tuple[Tensor, Tensor]:
        features = self.encoder(mel)
        
        spk_emb = self.speaker_embedding(features)
        spoofing_logits = self.spoofing_classifier(features)
        return spk_emb, spoofing_logits

    @property
    def device(self):
        return next(self.parameters()).device
    

