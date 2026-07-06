import pandas as pd

def load_bible():

    df = pd.read_csv(
        "../dataset/bible.csv",
        skiprows=5
    )

    return df