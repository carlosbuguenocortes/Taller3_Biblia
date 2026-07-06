from fastapi import FastAPI
from services.data_loader import load_bible

app = FastAPI()

@app.get("/")
def inicio():

    df = load_bible()

    return {
        "filas": len(df),
        "columnas": list(df.columns)
    }


@app.get("/dashboard")
def dashboard():

    df = load_bible()

    total_capitulos = (
        df.groupby("Book Name")["Chapter"]
        .max()
        .sum()
    )

    return {
        "total_versiculos": len(df),
        "total_libros": df["Book Name"].nunique(),
        "total_capitulos": int(total_capitulos)
    }

@app.get("/books")
def books():

    df = load_bible()

    return sorted(
        df["Book Name"].unique().tolist()
    )

@app.get("/verses-per-book")
def verses_per_book():

    df = load_bible()

    result = (
        df.groupby("Book Name")
        .size()
        .sort_values(ascending=False)
    )

    return result.to_dict()

@app.get("/book/{book_name}")
def get_book(book_name: str):

    df = load_bible()

    result = df[
        df["Book Name"] == book_name
    ]

    return result.to_dict(
        orient="records"
    )