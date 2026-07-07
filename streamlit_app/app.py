import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from gensim.models import Word2Vec


st.title(" Dashboard Biblia")

# 1) Dashboard principal
st.header("Dashboard Principal")
data = requests.get(
    "http://127.0.0.1:8000/dashboard"
).json()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Versículos",
        data["total_versiculos"]
    )

with col2:
    st.metric(
        "Libros",
        data["total_libros"]
    )

with col3:
    st.metric(
        "Capítulos",
        data["total_capitulos"]
    )


#filtros 
st.header("Filtros")

books = requests.get(
    "http://127.0.0.1:8000/books"
).json()

selected_book = st.selectbox(
    "Selecciona un libro",
    books,
    key="book_selector"
)
testament = st.selectbox(
    "Testamento",
    [
        "",
        "Antiguo Testamento",
        "Nuevo Testamento"
    ]
)
chapter = st.number_input(
    "Capítulo",
    min_value=1,
    value=1
)
filtered = requests.get(
    "http://127.0.0.1:8000/filter",
    params={
        "testament": testament,
        "book": selected_book,
        "chapter": chapter
    }
).json()

st.write(
    f"Resultados del filtro: {len(filtered)}"
)
st.divider()

#==========
# Top libros
#==========
books = requests.get(
    "http://127.0.0.1:8000/top-books"
).json()

df_books = pd.DataFrame(
    books.items(),
    columns=["Libro", "Versiculos"]
)

fig = px.bar(
    df_books,
    x="Libro",
    y="Versiculos",
    title="Top 10 Libros con Más Versículos"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

#Top palabras
st.header("Top 20 Palabras")

words = requests.get(
    "http://127.0.0.1:8000/top-words"
).json()

df_words = pd.DataFrame(
    words.items(),
    columns=["Palabra", "Frecuencia"]
)

fig = px.bar(
    df_words,
    x="Palabra",
    y="Frecuencia",
    title="Palabras Más Frecuentes"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

#longitud promedio de versiculos
#========
st.header(
    "Longitud Promedio de Versículos"
)

avg = requests.get(
    "http://127.0.0.1:8000/avg-verse-length"
).json()

df_avg = pd.DataFrame(
    avg.items(),
    columns=[
        "Libro",
        "Longitud"
    ]
)

fig = px.bar(
    df_avg.head(15),
    x="Libro",
    y="Longitud",
    title="Top 15 Libros por Longitud Promedio"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

#nube de palabras
st.header(f"Nube de Palabras - {selected_book}")
results = requests.get(
    f"http://127.0.0.1:8000/book/{selected_book}"
).json()

texto = " ".join(
    [r["Text"] for r in results]
)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(texto)

fig, ax = plt.subplots()

ax.imshow(wordcloud)

ax.axis("off")

st.pyplot(fig)

stats = requests.get(
    f"http://127.0.0.1:8000/book-stats/{selected_book}"
).json()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Capítulos",
        stats["capitulos"]
    )

with col2:
    st.metric(
        "Versículos",
        stats["versiculos"]
    )

st.divider()

#explorador de libros
st.header("Explorador de Libros")
book_data = requests.get(
    f"http://127.0.0.1:8000/book/{selected_book}"
).json()

st.subheader(f"Versículos de {selected_book}")
st.write(
    f" Total de versículos: {len(book_data)}"
)

cantidad = st.slider(
    "Cantidad de versículos",
    5,
    50,
    10
)

for verse in book_data[:cantidad]:
    st.info(
        f"{verse['Chapter']}:{verse['Verse']} - {verse['Text']}"
)
    
st.divider()    
#buscador de palabras
st.header("Buscadores")
st.subheader("Búsqueda por Palabra")
query = st.text_input(
    "Ingrese una palabra"
)

if query:

    results = requests.get(
        f"http://127.0.0.1:8000/search?text={query}"
    ).json()

    st.write(
        f"Resultados encontrados: {len(results)}"
    )

    for verse in results:

        st.write(
            f" {verse['Book Name']} "
            f"{verse['Chapter']}:{verse['Verse']}"
        )

        st.write(
            verse["Text"]
        )

        st.divider()

#nueva sección de buscador semántico
st.subheader("Buscador Semántico TF-IDF")
semantic_query = st.text_input(
    "Consulta semántica"
)
if semantic_query:

    semantic_results = requests.get(
        f"http://127.0.0.1:8000/search-semantic?text={semantic_query}"
    ).json()

    for result in semantic_results:

        st.success(
            f"{result['book']} "
            f"{result['chapter']}:{result['verse']}"
        )

        st.write(
            result["text"]
        )

        st.write(
            f"Similitud: {result['score']:.4f}"
        )

        st.divider()

#=========
#pca 2d
st.divider()
st.header("Visualizador PCA")
st.subheader("PCA 2D")
pca_data = requests.get(
    "http://127.0.0.1:8000/pca-2d"
).json()

df_pca = pd.DataFrame(
    pca_data
)

fig = px.scatter(
    df_pca,
    x="x",
    y="y",
    color="book",
    hover_data=["text"],
    title="PCA 2D de Versículos"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

#pca 3d
st.subheader("PCA 3D")
pca3d = requests.get(
    "http://127.0.0.1:8000/pca-3d"
).json()

df_pca3d = pd.DataFrame(
    pca3d
)

fig3d = px.scatter_3d(
    df_pca3d,
    x="x",
    y="y",
    z="z",
    color="book",
    hover_data=["text"],
    title="PCA 3D de Versículos"
)

st.plotly_chart(
    fig3d,
    use_container_width=True
)

st.divider()

st.header(" Word2Vec")
st.subheader(" Word2Vec 2D")

w2v_data = requests.get(
    "http://127.0.0.1:8000/word2vec-2d"
).json()

df_w2v = pd.DataFrame(
    w2v_data
)

fig_w2v = px.scatter(
    df_w2v,
    x="x",
    y="y",
    color="book",
    hover_data=["text"],
    title="Word2Vec 2D"
)

st.plotly_chart(
    fig_w2v,
    use_container_width=True
)

st.subheader(" Word2Vec 3D")

w2v_data_3d = requests.get(
    "http://127.0.0.1:8000/word2vec-3d"
).json()

df_w2v3d = pd.DataFrame(
    w2v_data_3d
)

fig_w2v3d = px.scatter_3d(
    df_w2v3d,
    x="x",
    y="y",
    z="z",
    color="book",
    hover_data=["text"],
    title="Word2Vec 3D"
)

st.plotly_chart(
    fig_w2v3d,
    use_container_width=True
)

st.divider()
#generador de versículos
st.header("Generador de Versículos")
model = st.selectbox(
    "Modelo",
    [
        "unigram",
        "bigram",
        "trigram"
    ],
    key="model_selector"
)

start_word = st.text_input(
    "Palabra inicial",
    value="Dios",
    key="start_word"
)

max_length = st.slider(
    "Largo máximo",
    5,
    50,
    20,
    key="length_slider"
)

if st.button(
    "Generar",
    key="generate_button"
):

    if model == "unigram":

        result = requests.get(
            f"http://127.0.0.1:8000/generate-unigram?length={max_length}"
        ).json()

    elif model == "bigram":

        result = requests.get(
            f"http://127.0.0.1:8000/generate-bigram?start={start_word}&length={max_length}"
        ).json()

    else:

        result = requests.get(
            f"http://127.0.0.1:8000/generate-trigram?start={start_word}&length={max_length}"
        ).json()

    st.success(
        result["text"]
    )


st.divider()
#versículo aleatorio
st.header("Versículo Aleatorio")

if st.button("Generar Versículo"):

    verse = requests.get(
        "http://127.0.0.1:8000/random-verse"
    ).json()

    st.success(
        f"{verse['Book Name']} "
        f"{verse['Chapter']}:{verse['Verse']}"
    )

    st.write(
        verse["Text"]
    )

