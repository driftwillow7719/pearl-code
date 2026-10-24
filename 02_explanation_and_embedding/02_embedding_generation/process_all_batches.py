# process_all_batches.py
import pandas as pd
from pathlib import Path
import time
import sys
from embedding_utils import embedder_model_loading, embedding_process_batch
import numpy as np
def process_all_batches(batch_dir='batches', output_dir='embeddings_output'):
    """
    Process all batches with auto-resume capability
    """
    # Setup
    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all batch files
    batch_files = sorted(batch_dir.glob("batch_*.csv"))
    
    if not batch_files:
        print(f"❌ No batch files found in {batch_dir}/")
        print("Run split_data_first.py first!")
        return
    
    print(f"Found {len(batch_files)} batches to process")
    
    # Check which batches are already done
    completed_batches = []
    for emb_file in output_dir.glob("batch_*_embeddings.npy"):
        batch_name = emb_file.stem.replace('_embeddings', '')
        completed_batches.append(batch_name)
    
    print(f"Already completed: {len(completed_batches)} batches")
    
    # Load model once (reuse for all batches)
    print("\nLoading model...")
    model = embedder_model_loading()
    
    # Process each batch
    success_count = 0
    failed_batches = []
    
    for i, batch_file in enumerate(batch_files, 1):
        batch_name = batch_file.stem  # e.g., 'batch_0001'
        
        # Skip if already done
        if batch_name in completed_batches:
            print(f"\n⏭️  {i}/{len(batch_files)}: {batch_name} already completed, skipping...")
            success_count += 1
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(batch_files)}: {batch_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load batch data
        df = pd.read_csv(batch_file)
        
        # Process batch
        success = embedding_process_batch(
            model=model,
            df=df,
            batch_name=batch_name,
            output_dir=str(output_dir)
        )
        
        elapsed = time.time() - start_time
        
        if success:
            success_count += 1
            print(f"✅ {batch_name} completed in {elapsed/60:.2f} minutes")
        else:
            failed_batches.append(batch_name)
            print(f"❌ {batch_name} FAILED after {elapsed/60:.2f} minutes")
            
            # Optional: stop on first failure
            # break
        
        # Small pause between batches
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Success: {success_count}/{len(batch_files)}")
    
    if failed_batches:
        print(f"❌ Failed: {len(failed_batches)} batches")
        print(f"Failed batches: {failed_batches}")
    
    return success_count, failed_batches

def merge_all_batches(batch_dir='batches', output_dir='embeddings_output'):
    """
    Merge all processed batches into final files
    """
    print("\n📦 Merging all batches...")
    
    output_dir = Path(output_dir)
    
    # Load all embeddings
    emb_files = sorted(output_dir.glob("batch_*_embeddings.npy"))
    if not emb_files:
        print("❌ No embedding files found!")
        return None, None
    
    all_embeddings = []
    all_data = []
    
    for emb_file in emb_files:
        batch_name = emb_file.stem.replace('_embeddings', '')
        
        # Load embeddings
        embeddings = np.load(emb_file)
        all_embeddings.append(embeddings)
        
        # Load corresponding data
        data_file = output_dir / f"{batch_name}_data.pkl"
        if data_file.exists():
            df = pd.read_pickle(data_file)
            all_data.append(df)
        else:
            print(f"⚠️  Warning: No data file for {batch_name}")
    
    # Combine
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
        final_data = pd.concat(all_data, ignore_index=True) if all_data else None
        
        # Save final
        np.save("final_embeddings.npy", final_embeddings)
        if final_data is not None:
            final_data.to_pickle("final_data.pkl")
            final_data.to_csv("final_data.csv", index=False)
        
        print(f"✅ Final embeddings shape: {final_embeddings.shape}")
        if final_data is not None:
            print(f"✅ Final data shape: {final_data.shape}")
        
        return final_embeddings, final_data
    
    return None, None

if __name__ == "__main__":
    # Step 1: Process all batches
    success, failed = process_all_batches(batch_dir='260819_train_embed_batches', output_dir='260819_train_embed')
    
    # Step 2: Merge results (optional)
    if success > 0 and not failed:
        merge_all_batches(batch_dir='260819_train_embed_batches', output_dir='260819_train_embed')
    else:
        print("\n⚠️  Some batches failed. Run again to retry failed batches.")