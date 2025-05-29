import streamlit as st # Importamos Streamlit para crear la interfaz web
import numpy as np # Importamos numpy para manejo de arrays numéricos
import PyPDF2 # Importamos PyPDF2 para manejar archivos PDF
import faiss # Importamos FAISS para manejar la indexación y búsqueda de vectores con IA

# Importaciones relacionadas con torch después de streamlit
from langchain_ollama import OllamaLLM # Esto es para cargar el modelo de Ollama
from langchain_huggingface import HuggingFaceEmbeddings # Importamos HuggingFaceEmbeddings para manejar los embeddings de texto
from langchain_community.vectorstores import FAISS # Importamos FAISS para manejar la base de datos de vectores
from langchain.text_splitter import CharacterTextSplitter # Importamos CharacterTextSplitter para dividir el texto en fragmentos manejables
from langchain.schema import Document # Importamos Document para manejar los documentos de texto

# ... resto del código ...
# Cargar el modelo de Ollama
llm = OllamaLLM(model="mistral")
# Cargar los embeddings de HuggingFace con el modelo  sentence-transformers/all-MiniLM-L6-v2
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Crear un índice FAISS para almacenar los embeddings
index = faiss.IndexFlatL2(384)  # Crear un índice FAISS para almacenar los embeddings MINILM-L6-v2
vector_store ={}
summary_text = ""  # Variable para almacenar el texto resumido
# Función para leer un archivo PDF y extraer su texto
def read_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error al leer el PDF: {e}")
        return None
# Función para almacenar la información en FAISS
def store_in_faiss(text, doc_id):
    global index, vector_store
    st.write(f"💾 Almacenando el contenido del documento {doc_id} en FAISS...")
    # Dividir el texto en fragmentos manejables
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(text)
    vectors = embeddings.embed_documents(texts)  # Obtener los embeddings de los fragmentos de texto
    vectors = np.array(vectors, dtype=np.float32)  # Convertir a un array de numpy de tipo float32 para la consistencia con FAISS
    # Añadir los embeddings al índice FAISS
    index.add(vectors)
    # Almacenar los textos y sus IDs en el vector_store
    vector_store[len(vector_store)] = (doc_id, texts)
    return f"✅ Contenido del documento {doc_id} almacenado en FAISS."


# Funcion para resumir el contenido de la página web
def summarize_content(content):
    global summary_text
    st.write("📝 Resumiendo el contenido...")
    if not content:
        return "No se pudo extraer contenido de la página."
    prompt = f"Resume el siguiente contenido:\n\n{content[:3000]}" # Limitar a los primeros 3000  caracteres para evitar problemas de longitud
    response = llm.invoke(prompt)
    summary_text = response
    return response

# Funcion para descargar el resumen
def download_summary():
    global summary_text
    if summary_text:
        st.download_button(
            label="Descargar Resumen",
            data=summary_text,
            file_name="resumen.txt",
            mime="text/plain"
        )
    else:
        st.warning("No hay resumen disponible para descargar.")

def retrieve_and_answer(query):
    global index, vector_store
    st.write("🔍 Buscando información relevante...")
    # Obtener el embedding de la consulta
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)  # Obtener el embedding de la consulta
    # Buscar los k vecinos más cercanos en FAISS
    k = 2  # Número de resultados a recuperar
    distances, indices = index.search(query_vector, k)
    print(f"Indices encontrados: {indices}, Distancias: {distances}")
    context = ""
    for idx in indices[0]:
        if idx in vector_store:
            texts = vector_store[idx][1]  # Obtener los textos asociados al índice
            context += " ".join(texts) + "\n\n"
    if not context:
        return "No se encontró información relevante en la base de datos."
    response = llm.invoke(f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:")
    return response

st.title("IA Document Reader")  # Título de la aplicación
st.write("Sube un archivo PDF y haz preguntas sobre su contenido y proporciona un resumen.")  # Descripción de la aplicación
# Subida de archivo PDF
uploaded_file = st.file_uploader("Selecciona un archivo PDF", type="pdf")  # Subida de archivo PDF
if uploaded_file is not None:
    st.write("📄 Archivo PDF subido correctamente.")
    # Leer el archivo PDF y extraer su texto
    pdf_text = read_pdf(uploaded_file)
    summary_text = summarize_content(pdf_text)  # Resumir el contenido del PDF
    if pdf_text:
        st.write("Texto extraído del PDF:")
        st.text_area("Texto del PDF", value=pdf_text, height=300)  # Mostrar el texto extraído
        # Almacenar el texto en FAISS
        doc_id = uploaded_file.name  # Usar el nombre del archivo como ID del documento
        result = store_in_faiss(pdf_text, doc_id)
        st.success(result)  # Mostrar mensaje de éxito
        st.subheader("📝 Resumen del contenido:")
        st.write(summary_text)
        download_summary()
        # Botón para descargar el resumen
# Pregunta del usuario
st.subheader("❓ Pregunta:")
query = st.text_input("Pregunta sobre el contenido del PDF:")  # Entrada de texto para la pregunta
if query:
    response = retrieve_and_answer(query)  # Recuperar la respuesta basada en la pregunta
    st.subheader("💬 Respuesta:")  # Subtítulo para la respuesta
    st.write(response)  # Mostrar la respuesta al usuario
