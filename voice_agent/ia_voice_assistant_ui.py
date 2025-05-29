import streamlit as st # Importamos Streamlit para crear la interfaz web
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory # Esto es para manejar el historial de mensajes de chat
from langchain_core.prompts import PromptTemplate # Esto para formatear las preguntas o prompts que le pasamos al modelo
from langchain_ollama import OllamaLLM # Esto es para cargar el modelo de Ollama

# Cargar el modelo de Ollama
llm = OllamaLLM(model="mistral")
# Inicializar el historial de mensajes de chat si no existe en la sesión de Streamlit
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
    
# Inicializar el motor de texto a voz
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Ajustar la velocidad de habla
# Inicializar el reconocedor de voz
recognizer = sr.Recognizer()

# Función para escuchar y reconocer voz
def listen():
    with sr.Microphone() as source:
        st.write("🎤 Escuchando...")
        recognizer.adjust_for_ambient_noise(source) # Ajustar el ruido ambiente
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language='es-ES')
            st.write(f"🔈 Has dicho: {text}")
            return text.lower()
        except sr.UnknownValueError:
            st.write("🤖 No he entendido lo que has dicho.")
            return None
        except sr.RequestError as e:
            st.write(f"⚠  Error al conectar con el servicio de reconocimiento de voz: {e}")
            return None

# Definir una plantilla de prompt para formatear las preguntas
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Conversación previa: {chat_history}\nUsuario: {question}\nIA:"
)

# Función para procesar las respuestas de la IA
def run_chain(question):
    # Actualizar el historial de chat con la pregunta del usuario
    chat_history_text = "\n".join([
        f"{'Usuario' if msg.type == 'human' else 'IA'}: {msg.content}"
        for msg in st.session_state.chat_history.messages
    ])
    # Formatear el prompt con el historial de chat y la pregunta
    prompt = prompt_template.format(chat_history=chat_history_text, question=question)
    # Invocar el modelo LLM con el prompt formateado
    response = llm.invoke(prompt)
    # Agregar la pregunta del usuario y la respuesta de la IA al historial de chat
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    return response

# Streamlit UI
st.title("🤖 Asistente de Voz IA")
st.write("Pregúntame cualquier cosa, y haré todo lo posible para ayudarte.")
# Botón para escuchar la pregunta del usuario
if st.button("🎤 Escuchar Pregunta"):
    user_input = listen()
    if user_input:
        response = run_chain(user_input)
        st.write(f"💬 Tú: {user_input}")
        st.write(f"🤖 IA: {response}")
        # Hablar la respuesta de la IA
        engine.say(response)
        engine.runAndWait()
# Mostrar el historial de chat
st.subheader("🔂 Historial de Chat")
for msg in st.session_state.chat_history.messages:
    st.write(f"{'Usuario' if msg.type == 'human' else 'IA'}: {msg.content}")
# Botón para reiniciar el historial de chat
if st.button("🔄 Reiniciar Historial de Chat"):
    st.session_state.chat_history = ChatMessageHistory()
    st.write("Historial de chat reiniciado.")