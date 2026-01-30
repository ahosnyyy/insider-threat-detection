# 1. Paper LSTM Autoencoder @src/models/paper_lstm_autoencoder.py

## Type
Unsupervised Seq2Seq LSTM Autoencoder for session-level anomaly detection (NLP-style sequence reconstruction applied to behavioural event vectors).

## Encoder
- **Core:** Uni-Directional LSTM
  - Default: 2 layers
  - Layer 1: input_dim → enc_hidden
  - Layer 2: enc_hidden → bottleneck_dim
  - Purpose: Encode session into a fixed-size latent vector
- **Bottleneck:**
  - Last valid (non-padded) time step of the encoder output is taken as the session embedding
  - Mask-aware: uses the final position where mask=1 per sample (variable-length sequences)
  - Produces a fixed-size session embedding of dimension bottleneck_dim

## RepeatVector
- Bottleneck embedding (B, bottleneck_dim) is repeated seq_len times → (B, T, bottleneck_dim)
- Serves as the decoder input for all time steps in a single forward pass

## Decoder
- **Core:** Uni-Directional LSTM
  - Layer 1: bottleneck_dim → bottleneck_dim
  - Layer 2: bottleneck_dim → input_dim (direct reconstruction per time step)
- **Input Strategy:**
  - Single forward pass: the repeated bottleneck is fed to the decoder for all steps at once (no teacher forcing, no autoregressive decoding)
- **Output Layer:**
  - Second decoder LSTM hidden size equals input_dim, so its output is the reconstructed feature vector per time step

## Loss & Anomaly Scoring
- **Loss:** Masked Mean Squared Error (MSE) over valid (non-padded) events
- **Anomaly Score:**
  - Mean reconstruction error over valid (non-padded) events per session
  - High error ⇒ anomalous session

## Default Configuration
- **bottleneck_dim:** 16
- **enc_hidden:** 32 (or input_dim if not set)
- **bidirectional:** False (encoder and decoder are unidirectional)
