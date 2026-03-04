from langchain_community.llms import Ollama
llm = Ollama(model="mistral")
response = llm.invoke("Tell me a joke")
print(response)