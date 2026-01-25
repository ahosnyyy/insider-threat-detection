import argparse
import logging
import pickle
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.data import FeatureExtractor, SequenceBuilder
from src.models import LSTMAutoencoder, TransformerAutoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: Path, model_type: str, feature_dim: int):
    """Load saved model."""
    if model_type == "lstm":
        model = LSTMAutoencoder(input_dim=feature_dim)
    else:
        model = TransformerAutoencoder(input_dim=feature_dim)
        
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def visualize_sequence(
    session_id: str,
    model_type: str,
    db_path: Path,
    output_dir: Path
):
    """Visualize reconstruction of a specific session."""
    models_dir = Path("models")
    extractor_path = models_dir / "feature_extractor.pkl"
    model_path = models_dir / f"{model_type}_autoencoder_best.pt"
    
    if not extractor_path.exists() or not model_path.exists():
        logger.error("Model or FeatureExtractor not found. Run training first.")
        return

    # 1. Load Extractor
    with open(extractor_path, "rb") as f:
        extractor = pickle.load(f)
        
    # 2. Get Data for User
    con = duckdb.connect(str(db_path))
    
    # identifying user first
    res = con.execute(f"SELECT user_id, start_time FROM session_features WHERE session_id = '{session_id}'").fetchone()
    if not res:
        logger.error(f"Session {session_id} not found in database.")
        con.close()
        return
        
    user_id, session_time = res
    logger.info(f"Found session {session_id} for user {user_id} at {session_time}")
    
    # Get user history (all sessions up to this one, or just all sessions to simplify grouping)
    # We load all sessions for the user to ensure SequenceBuilder works as expected (context)
    df = con.execute(f"SELECT * FROM session_features WHERE user_id = '{user_id}' ORDER BY start_time").fetchdf()
    con.close()
    
    # 3. Process Data
    features = extractor.transform(df)
    
    builder = SequenceBuilder(sequence_length=100) # Assuming default
    sequences, masks, session_ids = builder.build_user_sequences(df.reset_index(drop=True), features)
    
    # Find index of target session
    try:
        idx = session_ids.index(session_id)
    except ValueError:
        logger.error("Session lost during sequence building (maybe truncated?)")
        return
        
    target_seq = sequences[idx] # (seq_len, feature_dim)
    target_mask = masks[idx]
    
    # 4. Load Model and Predict
    feature_dim = extractor.get_feature_dim()
    model = load_model(model_path, model_type, feature_dim)
    
    # Prepare batch
    seq_tensor = torch.FloatTensor(target_seq).unsqueeze(0) # (1, seq_len, dim)
    mask_tensor = torch.FloatTensor(target_mask).unsqueeze(0)
    
    with torch.no_grad():
        output = model(seq_tensor, mask_tensor)
        if isinstance(output, tuple):
            reconstructed, _ = output
        else:
            reconstructed = output
            
    reconstructed = reconstructed.squeeze(0).numpy()
    
    # 5. Visualize
    # We visualize the LAST step of the sequence (current session) or the whole sequence?
    # Usually we want to see the reconstruction of the whole sequence context vs actual.
    
    # But for anomaly detection, we care about the reconstruction error of the valid steps.
    
    # Plot Heatmap of Error
    error = (target_seq - reconstructed) ** 2
    
    # Mask out padding
    valid_len = int(target_mask.sum())
    # Take only valid steps (at the end of sequence usually)
    # SequenceBuilder pads at BEGINNING if short.
    # So valid data is at [-valid_len:]
    
    plot_seq = target_seq[-valid_len:]
    plot_recon = reconstructed[-valid_len:]
    plot_error = error[-valid_len:]
    
    feature_names = extractor.get_feature_names()
    
    plt.figure(figsize=(15, 10))
    
    # 1. Original
    plt.subplot(3, 1, 1)
    sns.heatmap(plot_seq.T, cmap="viridis", yticklabels=feature_names)
    plt.title("Original Sequence (User History -> Current Session)")
    
    # 2. Reconstructed
    plt.subplot(3, 1, 2)
    sns.heatmap(plot_recon.T, cmap="viridis", yticklabels=feature_names)
    plt.title("Reconstructed Sequence")
    
    # 3. Error
    plt.subplot(3, 1, 3)
    sns.heatmap(plot_error.T, cmap="rocket", yticklabels=feature_names)
    plt.title("Reconstruction Error (MSE)")
    
    plt.tight_layout()
    save_path = output_dir / f"sequence_{session_id}_{model_type}.png"
    plt.savefig(save_path)
    logger.info(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize sequence reconstruction")
    parser.add_argument("--session-id", type=str, required=True, help="Session ID to visualize")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/cert.duckdb"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visualize_sequence(args.session_id, args.model, args.db_path, args.output_dir)

if __name__ == "__main__":
    main()
