import pandas as pd
import os

def load_bible():

    ruta = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "dataset",
        "bible.csv"
    )

    ruta = os.path.abspath(ruta)

    df = pd.read_csv(
        ruta,
        skiprows=5
    )

    return df