import os
import pickle

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--transformer-dir")
def preprocess(input_dir: str, output_dir: str, transformer_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "train.csv"), index_col=0)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop("target", 1))
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(transformer_dir, exist_ok=True)
    data["target"].to_csv(os.path.join(output_dir, "train_target.csv"))
    np.savetxt(
        os.path.join(output_dir, "train_scaled.csv"), data_scaled, delimiter=","
    )
    with open(os.path.join(transformer_dir, "scaler.pkl"), "wb") as fout:
        pickle.dump(scaler, fout)
        

if __name__ == "__main__":
    preprocess()
