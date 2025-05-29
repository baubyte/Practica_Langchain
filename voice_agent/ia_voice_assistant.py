import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory # Esto es para manejar el historial de mensajes de chat
from langchain_core.prompts import PromptTemplate # Esto para formatear las preguntas o prompts que le pasamos al modelo
from langchain_ollama import OllamaLLM # Esto es para cargar el modelo de Ollama


# Cargar el modelo de Ollama
llm = OllamaLLM(model="mistral")
# Inicializar el historial de mensajes de chat
chat_history = ChatMessageHistory()
# Inicializar el motor de texto a voz
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Ajustar la velocidad de habla
# Inicializar el reconocedor de voz
recognizer = sr.Recognizer()

# Función para hablar
def speak(text):
    try:
        engine.say(text)
        engine.setProperty('voice', 'spanish')  # Cambiar a voz en español
        engine.runAndWait()
    except Exception as e:
        print(f"Error al hablar: {e}")
        print(f"Texto: {text}")
# Función para escuchar y reconocer voz
def listen():
    with sr.Microphone() as source:
        print("🎤 Escuchando...")
        recognizer.adjust_for_ambient_noise(source) # Ajustar el ruido ambiente
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language='es-ES')
            print(f"🔈 Has dicho: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("🤖 No he entendido lo que has dicho.")
            return None
        except sr.RequestError as e:
            print(f"⚠  Error al conectar con el servicio de reconocimiento de voz: {e}")
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

# Bucle principal de la IA
speak("¡Hola! Soy tu asistente de voz")
speak("Puedes hacerme preguntas sobre cualquier tema, y haré todo lo posible para ayudarte.")
speak("Para terminar, simplemente di 'adiós' o 'salir'.")
while True:
    user_input = listen()
    if user_input in ["adiós", "salir"]:
        speak("¡Hasta luego! Que tengas un buen día.")
        break
    if user_input:
        response = run_chain(user_input)
        print(f"🤖 IA: {response}")
        speak(response)
    else:
        speak("No he entendido tu pregunta. Por favor, inténtalo de nuevo.")