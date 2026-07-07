#  Taller 3 - Análisis Interactivo del Corpus Bíblico

## Curso

Programación Científica
## Integrantes
- Carlos Bugueño Cortés
- Constantino Bekios
## Descripción

Este proyecto corresponde al Taller 3 del curso Programación Científica.

Se desarrolló una arquitectura cliente-servidor para el análisis y visualización interactiva del corpus bíblico utilizando FastAPI como backend y Streamlit como frontend.

La aplicación permite explorar el texto bíblico mediante técnicas de Procesamiento de Lenguaje Natural (NLP), incluyendo:

- Estadísticas descriptivas del corpus.
- Búsqueda semántica basada en TF-IDF.
- Visualización mediante PCA.
- Representación semántica mediante Word2Vec.
- Generación automática de texto mediante modelos N-Gram.

El procesamiento de datos se realiza completamente en la API, mientras que Streamlit actúa únicamente como interfaz de visualización e interacción. Esto cumple con los requerimientos establecidos en el laboratorio.

---

# Tecnologías Utilizadas

- Python 3.12
- FastAPI
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Gensim
- Plotly
- WordCloud

---

# Estructura del Proyecto

```text
Taller3_Biblia
│
├── api
│   ├── services
│   └── main.py
│
├── dataset
│   └── bible.csv
│
├── streamlit_app
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

# Requisitos

- Python 3.12
- pip
- Git

---

# Instalación

## 1. Clonar repositorio

```bash
git clone https://github.com/carlosbuguenocortes/Taller3_Biblia.git
```

## 2. Entrar al proyecto

```bash
cd Taller3_Biblia
```

## 3. Crear entorno virtual

```bash
python -m venv venv
```

## 4. Activar entorno virtual

### Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux / Mac

```bash
source venv/bin/activate
```

## 5. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecución

## Levantar la API

```bash
uvicorn api.main:app --reload
```

Disponible en:

```text
http://127.0.0.1:8000
```

Documentación Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Levantar Streamlit

En una segunda terminal:

```bash
streamlit run streamlit_app/app.py
```

Disponible en:

```text
http://localhost:8501
```

---

# Funcionalidades

## Dashboard Principal

Permite visualizar:

- Cantidad de versículos.
- Cantidad de capítulos.
- Cantidad de libros.
- Top palabras más frecuentes.
- Longitud promedio de versículos.
- Nube de palabras.
- Estadísticas por libro.

Incluye filtros de:

- Testamento.
- Libro.
- Capítulo.

---

## Buscador Semántico

Implementado mediante TF-IDF y similitud coseno.

El usuario ingresa una consulta textual y la API devuelve los versículos más similares semánticamente.

---

## PCA

Visualización interactiva de versículos utilizando reducción de dimensionalidad basada en TF-IDF.

### PCA 2D

Endpoint:

```text
/pca-2d
```

Visualiza los versículos proyectados en dos componentes principales.

### PCA 3D

Endpoint:

```text
/pca-3d
```

Visualiza los versículos proyectados en tres componentes principales.

---

## Word2Vec

Representación vectorial semántica utilizando embeddings generados mediante Word2Vec y posteriormente reducidos mediante PCA.

### Word2Vec 2D

Endpoint:

```text
/word2vec-2d
```

Entrega coordenadas bidimensionales para visualizar relaciones semánticas entre versículos.

### Word2Vec 3D

Endpoint:

```text
/word2vec-3d
```

Entrega coordenadas tridimensionales para explorar agrupaciones semánticas de forma interactiva.

---

## Generador de Versículos

Permite generar texto utilizando modelos N-Gram.

Modelos disponibles:

- Unigram
- Bigram
- Trigram

Parámetros:

- Modelo
- Palabra inicial
- Largo máximo

La generación es realizada completamente por la API.

---

# Endpoints Principales

| Endpoint | Descripción |
|-----------|-----------|
| /stats | Estadísticas generales |
| /search | Buscador semántico TF-IDF |
| /pca-2d | PCA en 2 dimensiones |
| /pca-3d | PCA en 3 dimensiones |
| /word2vec-2d | Word2Vec proyectado en 2D |
| /word2vec-3d | Word2Vec proyectado en 3D |
| /generate | Generación de texto N-Gram |

---

# Objetivos Cumplidos

 Dashboard interactivo

 API REST para análisis textual

 Búsqueda semántica con TF-IDF

 PCA 2D

 PCA 3D

 Word2Vec 2D

 Word2Vec 3D

 Generación de texto mediante N-Gram

 Arquitectura Cliente-Servidor

 Visualizaciones interactivas en Streamlit

---

# Conclusiones

El proyecto permitió aplicar diversas técnicas de Procesamiento de Lenguaje Natural sobre el corpus bíblico, incorporando recuperación de información, visualización de datos, reducción de dimensionalidad, representación semántica mediante embeddings y generación automática de texto.

La separación entre FastAPI y Streamlit permitió implementar una arquitectura cliente-servidor clara y escalable, donde la API concentra todo el procesamiento y la aplicación Streamlit se encarga exclusivamente de la interacción con el usuario.
