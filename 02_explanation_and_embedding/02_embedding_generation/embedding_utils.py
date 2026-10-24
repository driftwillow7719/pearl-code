# embedding_utils.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import json
import gc
import psutil
import os

# Optional: set HF_ENDPOINT in your shell if you need a Hugging Face mirror.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "microsoft/harrier-oss-v1-270m")
# Exact model snapshot used in the experiments (Hugging Face commit hash).
EMBEDDING_REVISION = os.environ.get("EMBEDDING_REVISION", "31de22b673913c7d658c0f03f792d77c2dcf8ebd")

def data_loading(file_path):
    """Load data from a file (full CSV or a batch)"""
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} rows from {file_path}")
    return df

def embedder_model_loading():
    """Load embedding model"""
    print(f"\n🔢 Loading embedding model: microsoft/harrier-oss-v1-270m")
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cuda:0",
        revision=EMBEDDING_REVISION,
    )
    print(f"✅ Model loaded! Dimension: {model.get_sentence_embedding_dimension()}")
    return model

def embedding_process_batch(model, df, batch_name, output_dir='embeddings_output'):
    """
    Process one batch and save immediately

    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # product_description is for test set
    # texts = df['narration'].fillna('').tolist()
    # n_nan = df['product_description'].isna().sum()
    # n_empty = (df['product_description'].fillna('').str.strip() == '').sum()  # catches NaN + empty-string cases
    n_nan = df['item_description'].isna().sum()
    n_empty = (df['item_description'].fillna('').str.strip() == '').sum()  # catches NaN + empty-string cases
    print(f"NaN: {n_nan} / {len(df)} ({n_nan/len(df):.1%})")
    print(f"NaN or blank: {n_empty} / {len(df)} ({n_empty/len(df):.1%})")
    # texts = df['product_description'].fillna('').tolist()
    texts = df['item_description'].fillna('').tolist()
    print(f"\n🔢 Generating embeddings for {batch_name} ({len(texts)} texts)...")
    
    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=16,
            device='cuda'
        )
        
        # Save files
        emb_path = f"{output_dir}/{batch_name}_embeddings.npy"
        df_path = f"{output_dir}/{batch_name}_data.pkl"
        
        np.save(emb_path, embeddings)
        df.to_pickle(df_path)
        
        # Save metadata
        metadata = {
            'batch_name': batch_name,
            'num_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'dtype': str(embeddings.dtype),
            'normalized': True
        }
        
        with open(f"{output_dir}/{batch_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ {batch_name} saved! Shape: {embeddings.shape}")
        
        # Cleanup
        del embeddings
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed on {batch_name}: {e}")
        return False