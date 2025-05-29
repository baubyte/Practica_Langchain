import requests # Importamos requests para hacer peticiones HTTP
from bs4 import BeautifulSoup # Importamos BeautifulSoup para parsear el HTML
import streamlit as st # Importamos Streamlit para crear la interfaz web
from langchain_community.chat_message_histories import ChatMessageHistory # Esto es para manejar el historial de mensajes de chat
from langchain_core.prompts import PromptTemplate # Esto para formatear las preguntas o prompts que le pasamos al modelo
from langchain_ollama import OllamaLLM # Esto es para cargar el modelo de Ollama

# Cargar el modelo de Ollama
llm = OllamaLLM(model="mistral")

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
        return text[:2000]  # Limitar a los primeros 2000 caracteres para evitar problemas de longitud
    except requests.RequestException as e:
        st.error(f"❌ Error al acceder a la URL: {e}")
        return None
    
# Funcion para resumir el contenido de la página web
def summarize_content(content):
    st.write("📝 Resumiendo el contenido...")
    if not content:
        return "No se pudo extraer contenido de la página."
    prompt = f"Resume el siguiente contenido:\n\n{content[:1000]}" # Limitar a los primeros 1000 caracteres para evitar problemas de longitud
    response = llm.invoke(prompt)
    return response

st.title("🌐 Web Scraper IA")
st.write("Introduce una URL para extraer y resumir su contenido:")
url = st.text_input(": URL de la página web:")
if url:
    content = scrape_website(url)
    if "⚠️ Fallo" in content or "❌ Error" in content:
        st.error(content)
    else:
        st.subheader("📄 Contenido extraído:")
        st.write(content)
        summary = summarize_content(content)
        st.subheader("📝 Resumen del contenido:")
        st.write(summary)
