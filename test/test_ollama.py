from langchain_community.llms import Ollama

llm = Ollama(model="mistral")

# response = llm.invoke("Tell me a joke")
# print(response)

response = llm.invoke("The first man on the summit of Mount Everest, the highest peak on Earth, was ...")
print(response)