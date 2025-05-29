from langchain_community.chat_message_histories import ChatMessageHistory # Esto es para manejar el historial de mensajes de chat
from langchain_core.prompts import PromptTemplate # Esto para formatear las preguntas o prompts que le pasamos al modelo
from langchain_ollama import OllamaLLM # Esto es para cargar el modelo de Ollama

# Cargar el modelo de Ollama
llm = OllamaLLM(model="mistral")
# Inicializar el historial de mensajes de chat
chat_history = ChatMessageHistory()
# Definir una plantilla de prompt para formatear las preguntas
prompt_template = PromptTemplate(
    input_variables=["chat_history","question"],
    template="Conversación Anterior: {chat_history}\nUsuario: {question}\nIA:"
)
# Función para ejecutar el agente IA con Historial de chat
def run_chain(question):
    # Actualizar el historial de chat con la pregunta del usuario
    chat_history_text = "\n".join([
        f"{'Usuario' if msg.type == 'human' else 'IA'}: {msg.content}"
        for msg in chat_history.messages
    ])

    # Formatear el prompt con el historial de chat y la pregunta
    prompt = prompt_template.format(chat_history=chat_history_text, question=question)
    # Invocar el modelo LLM con el prompt formateado
    response = llm.invoke(prompt)
    # Agregar la pregunta del usuario y la respuesta de la IA al historial de chat
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    return response

print("\nBienvenido al Agente IA Básico con Historial de Chat!\n")
print("Pregúntame cualquier cosa, y haré todo lo posible para ayudarte.\n")
print("Escribe 'exit' para salir del chat.\n")
while True:
    user_input = input("Tú Pregunta: ")
    if user_input.lower() == "exit":
        print("¡Adiós!")
        break
    response = run_chain(user_input)
    print("IA:", response)