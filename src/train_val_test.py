import os
import pandas as pd


def train_val_test(input_dir, save_dir, split_ratio=(0.8, 0.18)):
    """Splits dataset into train, validation and test with specified split_ratio.
    and Returns split datasets saved in `.csv`

    Parameters:
        input_dir (str): Input path of the dataset
        save_dir (str): Directory to save output datasets
        split_ratio (tuple(float)): Proportion to split dataset into the train, validation and test dataset
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(input_dir) if input_dir.endswith(".csv") else pd.read_excel(input_dir)
    filename = os.path.basename(input_dir).split(".")[0]

    data_size = len(df)
    train_size = round(split_ratio[0] * data_size)
    val_size = round(split_ratio[1] * data_size)
    test_size = data_size - (train_size + val_size)

    train_df = df.sample(n=train_size, axis=0)
    val_df = df.sample(n=val_size, axis=0)
    test_df = df.sample(n=test_size, axis=0)

    # Save all dataframes
    train_df.to_csv(f"{save_dir}/{filename}_train.csv", index=False)
    val_df.to_csv(f"{save_dir}/{filename}_val.csv", index=False)
    test_df.to_csv(f"{save_dir}/{filename}_test.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="python train_val_test.py",
                                     description="Training, Validation and Testing Dataset",
                                     usage="%(prog)s [source_dir] [target_dir] [split_ratio]")
    parser.add_argument("--source_dir", "-i", type=str, help="Path to input dataset")
    parser.add_argument("--target_dir", "-o", type=str, help="Directory to saved datasets")
    parser.add_argument("--split_ratio", "-r", type=float, nargs="+", help="Ratio to split dataset")

    opt = parser.parse_args()

    SOURCE_DIR = opt.source_dir
    TARGET_DIR = opt.target_dir
    RATIO = opt.split_ratio

    train_val_test(input_dir=SOURCE_DIR, save_dir=TARGET_DIR, split_ratio=RATIO)
