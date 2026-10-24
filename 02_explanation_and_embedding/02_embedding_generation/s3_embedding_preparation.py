import argparse

from split_data_first import split_csv_into_batches
from process_all_batches import process_all_batches, merge_all_batches


def main():
    parser = argparse.ArgumentParser(description="Split a CSV into batches and embed each batch.")
    parser.add_argument("--input-csv", required=True, help="CSV with the text column to embed")
    parser.add_argument("--batch-dir", required=True, help="where the CSV batches are written")
    parser.add_argument("--embed-dir", required=True, help="where the per-batch embeddings and merged output go")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    # split data into batches
    split_csv_into_batches(
        original_file=args.input_csv,
        batch_size=args.batch_size,
        output_dir=args.batch_dir)

    # Step 1: produce embeddings batch by batch
    success, failed = process_all_batches(batch_dir=args.batch_dir, output_dir=args.embed_dir)

    # Step 2: merge results
    if success > 0 and not failed:
        merge_all_batches(batch_dir=args.batch_dir, output_dir=args.embed_dir)
    else:
        print("\n⚠️  Some batches failed. Run again to retry failed batches.")


if __name__ == "__main__":
    main()
