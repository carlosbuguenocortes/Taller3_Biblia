from fastapi import FastAPI
from api.services.data_loader import load_bible
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from collections import Counter
import re
import random
from gensim.models import Word2Vec
import numpy as np

app = FastAPI()

@app.get("/")
def inicio():

    df = load_bible()

    return {
        "filas": len(df),
        "columnas": list(df.columns)
    }

def get_testament(book):

    old_testament = [
        "Génesis","Éxodo","Levítico","Números","Deuteronomio",
        "Josué","Jueces","Rut","1 Samuel","2 Samuel",
        "1 Reyes","2 Reyes","1 Crónicas","2 Crónicas",
        "Esdras","Nehemías","Esther","Job","Salmos",
        "Proverbios","Ecclesiastés","Canción de canciones",
        "Isaías","Jeremías","Lamentaciones","Ezequiel",
        "Daniel","Oseas","Joel","Amós","Abdías",
        "Jonás","Miqueas","Nahum","Habacuc",
        "Sofonías","Haggeo","Zacarías","Malaquías"
    ]

    if book in old_testament:
        return "Antiguo Testamento"

    return "Nuevo Testamento"

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

@app.get("/top-books")
def top_books():

    df = load_bible()

    result = (
        df.groupby("Book Name")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    return result.to_dict()
#busqueda de versiculos por texto
@app.get("/search")
def search(text: str):

    df = load_bible()

    result = df[
        df["Text"].str.contains(
            text,
            case=False,
            na=False
        )
    ]

    return result.head(50).to_dict(
        orient="records"
    )

@app.get("/random-verse")
def random_verse():

    df = load_bible()

    verse = df.sample(1)

    return verse.to_dict(
        orient="records"
    )[0]

@app.get("/book-stats/{book_name}")
def book_stats(book_name: str):

    df = load_bible()

    book = df[df["Book Name"] == book_name]

    return {
        "versiculos": len(book),
        "capitulos": int(book["Chapter"].max())
    }

@app.get("/search-semantic")
def search_semantic(text: str):

    df = load_bible()

    verses = df["Text"].fillna("").tolist()

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        verses
    )

    query_vector = vectorizer.transform(
        [text]
    )

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarities.argsort()[-10:][::-1]

    results = []

    for idx in top_indices:

        results.append({
            "book": df.iloc[idx]["Book Name"],
            "chapter": int(df.iloc[idx]["Chapter"]),
            "verse": int(df.iloc[idx]["Verse"]),
            "text": df.iloc[idx]["Text"],
            "score": float(similarities[idx])
        })

    return results

@app.get("/pca-2d")
def pca_2d():

    df = load_bible()

    sample = df.sample(
        min(500, len(df)),
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=500
    )

    X = vectorizer.fit_transform(
        sample["Text"]
    )

    pca = PCA(
        n_components=2
    )

    coords = pca.fit_transform(
        X.toarray()
    )

    result = []

    for i, row in enumerate(sample.itertuples()):

        result.append({
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "book": row._2,
            "text": row.Text[:100]
        })

    return result

@app.get("/pca-3d")
def pca_3d():

    df = load_bible()

    sample = df.sample(
        min(500, len(df)),
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=500
    )

    X = vectorizer.fit_transform(
        sample["Text"]
    )

    pca = PCA(
        n_components=3
    )

    coords = pca.fit_transform(
        X.toarray()
    )

    result = []

    for i, row in enumerate(sample.itertuples()):

        result.append({
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "z": float(coords[i][2]),
            "book": row._2,
            "text": row.Text[:100]
        })

    return result
@app.get("/word2vec-2d")
def word2vec_2d():

    df = load_bible()

    sample = df.sample(
        min(500, len(df)),
        random_state=42
    )

    sentences = [
        str(text).lower().split()
        for text in sample["Text"]
    ]

    model = Word2Vec(
        sentences,
        vector_size=50,
        window=5,
        min_count=1
    )

    embeddings = []

    for sentence in sentences:

        vectors = [
            model.wv[word]
            for word in sentence
        ]

        embeddings.append(
            np.mean(vectors, axis=0)
        )

    embeddings = np.array(embeddings)

    pca = PCA(
        n_components=2
    )

    coords = pca.fit_transform(
        embeddings
    )

    result = []

    for i, row in enumerate(sample.itertuples()):

        result.append({
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "book": row._2,
            "text": str(row.Text)[:100]
        })

    return result

@app.get("/word2vec-3d")
def word2vec_3d():

    df = load_bible()

    sample = df.sample(
        min(500, len(df)),
        random_state=42
    )

    sentences = [
        str(text).lower().split()
        for text in sample["Text"]
    ]

    model = Word2Vec(
        sentences,
        vector_size=50,
        window=5,
        min_count=1
    )

    embeddings = []

    for sentence in sentences:

        vectors = [
            model.wv[word]
            for word in sentence
        ]

        embeddings.append(
            np.mean(vectors, axis=0)
        )

    embeddings = np.array(embeddings)

    pca = PCA(
        n_components=3
    )

    coords = pca.fit_transform(
        embeddings
    )

    result = []

    for i, row in enumerate(sample.itertuples()):

        result.append({
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "z": float(coords[i][2]),
            "book": row._2,
            "text": str(row.Text)[:100]
        })

    return result

@app.get("/avg-verse-length")
def avg_verse_length():

    df = load_bible()

    df["length"] = (
        df["Text"]
        .astype(str)
        .str.len()
    )

    result = (
        df.groupby("Book Name")["length"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    return result.to_dict()

@app.get("/top-words")
def top_words():

    df = load_bible()

    text = " ".join(
        df["Text"].astype(str)
    ).lower()

    words = re.findall(
        r"\b\w+\b",
        text
    )

    
    stopwords = {
        "de","la","el","y","que",
        "en","a","los","las",
        "del","por","con","un",
        "una","se","su","al",
        "á","no","para","porque",
        "él","tu","es","lo",
        "como","mi","sobre",
        "yo","mas","ellos"
    }


    words = [
        w for w in words
        if w not in stopwords
    ]

    counter = Counter(words)

    return dict(
        counter.most_common(20)
    )

@app.get("/filter")
def filter_data(
    testament: str = None,
    book: str = None,
    chapter: int = None
):

    df = load_bible()

    df["Testament"] = (
        df["Book Name"]
        .apply(get_testament)
    )

    if testament:
        df = df[
            df["Testament"] == testament
        ]

    if book:
        df = df[
            df["Book Name"] == book
        ]

    if chapter:
        df = df[
            df["Chapter"] == chapter
        ]

    return df.head(100).to_dict(
        orient="records"
    )

@app.get("/generate-unigram")
def generate_unigram(
    length: int = 20
):

    df = load_bible()

    words = " ".join(
        df["Text"]
    ).split()

    generated = []

    for _ in range(length):
        generated.append(
            random.choice(words)
        )

    return {
        "text": " ".join(generated)
    }

@app.get("/generate-bigram")
def generate_bigram(
    start: str,
    length: int = 20
):

    import random

    df = load_bible()

    words = " ".join(
        df["Text"].astype(str)
    ).split()

    pairs = {}

    for i in range(len(words) - 1):

        current = words[i]
        next_word = words[i + 1]

        if current not in pairs:
            pairs[current] = []

        pairs[current].append(next_word)

    generated = [start]

    current = start

    for _ in range(length - 1):

        if current not in pairs:
            break

        current = random.choice(
            pairs[current]
        )

        generated.append(current)

    return {
        "text": " ".join(generated)
    }
    
@app.get("/generate-trigram")
def generate_trigram(
    start: str,
    length: int = 20
):

    import random

    df = load_bible()

    words = " ".join(
        df["Text"].astype(str)
    ).split()

    if len(words) < 3:
        return {
            "text": "Corpus demasiado pequeño"
        }

    triples = {}

    for i in range(len(words) - 2):

        key = (
            words[i],
            words[i + 1]
        )

        next_word = words[i + 2]

        if key not in triples:
            triples[key] = []

        triples[key].append(
            next_word
        )

    # buscar alguna pareja que comience con la palabra inicial
    start_pair = None

    for key in triples:

        if key[0].lower() == start.lower():
            start_pair = key
            break

    if start_pair is None:
        return {
            "text": f"No encontré la palabra inicial '{start}'"
        }

    generated = [
        start_pair[0],
        start_pair[1]
    ]

    current_pair = start_pair

    for _ in range(length - 2):

        if current_pair not in triples:
            break

        next_word = random.choice(
            triples[current_pair]
        )

        generated.append(
            next_word
        )

        current_pair = (
            current_pair[1],
            next_word
        )

    return {
        "text": " ".join(generated)
    }




@app.get("/test-word2vec")
def test_word2vec():

    sentences = [
        ["dios", "ama"],
        ["jesus", "salva"]
    ]

    model = Word2Vec(
        sentences,
        vector_size=10,
        min_count=1
    )

    return {
        "ok": True,
        "vector_size": len(model.wv["dios"])
    }