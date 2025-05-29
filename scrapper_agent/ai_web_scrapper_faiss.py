import requests # Importamos requests para hacer peticiones HTTP
from bs4 import BeautifulSoup # Importamos BeautifulSoup para parsear el HTML
import faiss # Importamos FAISS para manejar la indexación y búsqueda de vectores con IA
import numpy as np # Importamos numpy para manejo de arrays numéricos
import streamlit as st # Importamos Streamlit para crear la interfaz web
from langchain_ollama import OllamaLLM # Esto es para cargar el modelo de Ollama
from langchain_huggingface import HuggingFaceEmbeddings # Importamos HuggingFaceEmbeddings para manejar los embeddings de texto
from langchain_core.prompts import PromptTemplate # Esto para formatear las preguntas o prompts que le pasamos al modelo
from langchain_community.vectorstores import FAISS # Importamos FAISS para manejar la base de datos de vectores
from langchain.text_splitter import CharacterTextSplitter # Importamos CharacterTextSplitter para dividir el texto en fragmentos manejables
from langchain.schema import Document # Importamos Document para manejar los documentos de texto

# Cargar el modelo de Ollama
llm = OllamaLLM(model="mistral")
# Cargar los embeddings de HuggingFace con el modelo  sentence-transformers/all-MiniLM-L6-v2
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Crear un índice FAISS para almacenar los embeddings
index = faiss.IndexFlatL2(384)  # Crear un índice FAISS para almacenar los embeddings

vector_store ={}

# Función para hacer scraping de una página web
def scrape_website(url):
    try:
        st.write(f"🔍 Accediendo a la URL: {url}")
        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=header)
        if response.status_code != 200:
            st.error(f"⚠️ Fallo al acceder a la URL: {response.status_code}")
            return None
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')  # Extraer todos los párrafos
        # Extraer el texto de la página
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:5000]  # Limitar a los primeros 2000 caracteres para evitar problemas de longitud
    except requests.RequestException as e:
        st.error(f"❌ Error al acceder a la URL: {e}")
        return None
# Función para almacenar la información en FAISS
def store_in_faiss(text, url):
    global index, vector_store
    st.write(f"💾 Almacenando el contenido de {url} en FAISS...")
    # Dividir el texto en fragmentos manejables
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(text)
    vectors = embeddings.embed_documents(texts)  # Obtener los embeddings de los fragmentos de texto
    vectors = np.array(vectors, dtype=np.float32)  # Convertir a un array de numpy de tipo float32
    # Añadir los embeddings al índice FAISS
    index.add(vectors)
    # Almacenar los textos y sus URLs en el vector_store 
    vector_store[len(vector_store)] = (url, texts)
    return f"✅ Contenido de {url} almacenado en FAISS."

# Función para recuperar información de FAISS y responder a preguntas
def retrieve_and_answer(query):
    global index, vector_store
    st.write("🔍 Buscando información relevante...")
    # Obtener el embedding de la consulta
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)  # Obtener el embedding de la consulta

    # Buscar los k vecinos más cercanos en el índice FAISS
    k = 2  # Número de vecinos a recuperar
    distances, indices = index.search(query_vector, k)
    context = ""
    for idx in indices[0]:
        if idx in vector_store:
            context += " ".join(vector_store[idx][1]) + "\n\n"  # Concatenar los textos de los fragmentos
    if not context:
        return "No se encontró información relevante en la base de datos."
    response = llm.invoke(f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:")
    return response

st.title("🌐 Web Scraper IA")
st.write("Introduce una URL para extraer y almacenar su contenido para futuras consultas:")
url = st.text_input(": URL de la página web:")
if url:
    content = scrape_website(url)
    if "⚠️ Fallo" in content or "❌ Error" in content:
        st.error(content)
    else:
        st.subheader("📄 Contenido extraído:")
        st.write(content)
        store_message = store_in_faiss(content, url)
        st.success(store_message)

#Cuando el usuario quiere hacer una pregunta
st.subheader("❓ Pregunta:")
query = st.text_input("Pregunta sobre el contenido almacenado:")
if query:
    response = retrieve_and_answer(query)
    st.subheader("💬 Respuesta:")
    st.write(response)
