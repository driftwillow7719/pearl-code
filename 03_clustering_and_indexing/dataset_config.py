"""
Per-dataset settings shared by every script (s5, s6, s7, s8, s16, ...).
Edit paths and cutoffs here, in one place only.
"""

DATASETS = {
    "douban": dict(
        high=15, kind="pickle",
        data="260828_train_embed/final_data.pkl",
        embed="260828_train_embed/final_embeddings.npy",
        out_dir="260828_group_profiles",
    ),
    "baby": dict(
        high=3, kind="csv",
        data="260915_baby_train_embed/final_data.csv",
        embed="260915_baby_train_embed/final_embeddings.npy",
        out_dir="261002_baby_group_profiles",
    ),
    "home_and_kitchen": dict(
        high=6, kind="csv",
        data="260921_home_and_kitchen_synthetic_training_data.csv",
        embed="260921_home_and_kitchen_train_embed/final_embeddings.npy",   # check this name
        out_dir="260921_home_and_kitchen_group_profiles",                   # check this name
    ),
}