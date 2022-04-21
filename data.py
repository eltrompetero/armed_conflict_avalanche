import pandas as pd


def conflict_data_loader(conflict_type):
    return pd.read_csv(f"../data/conflict/data_{conflict_type}.csv")
