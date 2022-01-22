import numpy as np
import pandas as pd

if __name__ == '__main__':
    path = f"results/nbody/nbody_results_01.csv"
    df = pd.read_csv(path)
    model_types = df["misc.model_type"].unique()
    # Compute average errors
    processed_results = df.groupby(["misc.model_type"])[["test_ade_loss",
                                                         "test_fde_loss",
                                                         "test_vel_loss",
                                                         "val_ade_loss",
                                                         "val_fde_loss",
                                                         "val_vel_loss"]].agg(["mean", "std"])

    print(processed_results.to_latex(float_format="{:0.1f}".format))