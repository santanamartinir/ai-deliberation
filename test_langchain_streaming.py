from langchain_community.llms import Ollama

llm = Ollama(model="mistral")

for chunk in llm.stream("The first man on the summit of Mount Everest, the highest peak on Earth, was ..."):
    print(chunk, end="", flush=True)
print()