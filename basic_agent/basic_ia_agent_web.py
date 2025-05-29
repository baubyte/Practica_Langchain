import streamlit as st # Importamos Streamlit para crear la interfaz web
from langchain_community.chat_message_histories import ChatMessageHistory # Esto es para manejar el historial de mensajes de chat
from langchain_core.prompts import PromptTemplate # Esto para formatear las preguntas o prompts que le pasamos al modelo
from langchain_ollama import OllamaLLM # Esto es para cargar el modelo de Ollama

# Cargar el modelo de Ollama
llm = OllamaLLM(model="mistral")
# Inicializar el historial de mensajes de chat si no existe en la sesión de Streamlit
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Definir una plantilla de prompt para formatear las preguntas
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Conversación previa: {chat_history}\nUsuario: {question}\nIA:"
)
# Función para ejecutar el agente IA con Historial de chat
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

# Configuración de la interfaz de Streamlit
st.title("🤖 Agente IA Básico con Historial de Chat")
st.write("Pregúntame cualquier cosa, y haré todo lo posible para ayudarte.")

user_input = st.text_input("💬 Tú Pregunta:")
if user_input:
    response = run_chain(user_input)
    st.write(f"💬 Tú: {user_input}")
    st.write(f"🤖 IA: {response}")
    
# Mostrar el historial de chat
st.subheader("🔂 Historial de Chat")
if st.session_state.chat_history.messages:
    for msg in st.session_state.chat_history.messages:
        if msg.type == 'human':
            st.write(f"😜 **Usuario:** {msg.content}")
        else:
            st.write(f"🤖 **IA:** {msg.content}")