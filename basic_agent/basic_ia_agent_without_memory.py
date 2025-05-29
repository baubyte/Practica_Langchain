from langchain_ollama import OllamaLLM

# Load the Ollama model
llm = OllamaLLM(model="mistral")

print("\nBienvenido al Agente IA Básico!\n")
print("Pregúntame cualquier cosa, y haré todo lo posible para ayudarte.\n")
while True:
    user_input = input("Tú Pregunta (o 'exit' para salir): ")
    if user_input.lower() == "exit":
        print("¡Adiós!")
        break
    response = llm.invoke(user_input)
    print("IA:", response)