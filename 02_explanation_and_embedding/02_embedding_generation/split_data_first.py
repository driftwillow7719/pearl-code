# split_data_first.py
import pandas as pd
from pathlib import Path

def split_csv_into_batches(original_file, batch_size=1000, output_dir='batches'):
    """
    Split CSV into smaller batch files
    
    Parameters:
    - original_file: path to original CSV
    - batch_size: number of rows per batch
    - output_dir: directory to save batches
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {original_file}...")
    # df = pd.read_excel(original_file)
    df = pd.read_csv(original_file)
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    print(f"Total rows: {total_rows}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}\n")
    
    batch_files = []
    
    for i in range(0, total_rows, batch_size):
        batch_num = i // batch_size + 1
        batch_df = df.iloc[i:i+batch_size]
        
        batch_file = f"{output_dir}/batch_{batch_num:04d}.csv"
        batch_df.to_csv(batch_file, index=False)
        batch_files.append(batch_file)
        
        print(f"✅ Batch {batch_num:04d}: {len(batch_df)} rows -> {batch_file}")
    
    print(f"\n🎉 Split complete! {len(batch_files)} batches created")
    print(f"📁 Batches saved in: {output_dir}/")
    
    return batch_files

# Jalankan split
if __name__ == "__main__":
    split_csv_into_batches(
        original_file="dataset/260819/260819_train_with_narration.xlsx",
        batch_size=1000,  # ← Atur sesuai kebutuhan
        output_dir='260819_train_embed_batches'
    )