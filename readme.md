# 100 samples

## Audio encoder, both speaker and spoofing module
Not good, reaally, look at traintorch_testsmall_20250921-081522.txt, or this graph right here:
![39 epochs](docs/audioencoder_both_spoofing_spekervertification_100samples.png)

Experiment ran on google collab, T4 Telsa GPU
```bash
%cd /content/HalfWhipResult

!python -m test.traintorch_testsmall \
  --metadata_path /content/metadatas/metadata_small_train.csv \
  --prefix /content/vsasv_reorganized/content/vsasv_reorganized \
  --batch_size 128 \
  --epochs 64 \
  --device cuda \
  --lr 1e-3 \
  --eps 1e-8 \
  --weight_decay 0.01 \
  --n_mels 80 \
  --n_audio_ctx 1024 \
  --n_audio_state 128 \
  --n_audio_head 8 \
  --n_audio_layer 8 \
  --n_spkemb_layers 3
```
