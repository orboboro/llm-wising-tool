
import os
from dotenv import find_dotenv, load_dotenv
import time as t
from huggingface_hub import InferenceClient
from rag_system import search_similar, load_embedding_matrix

def huggingface_API_calling(query, embedding_matrix, model, temperature = 0, RAG = True):

    MODEL = (model).replace(":", "-")
    TEMPERATURE = temperature

    adjunction = ""
    if RAG:
        adjunction = " Impourtant: in order to answer the questions you have to base your answer on the text you will be provided. Do not make up things."

    system_prompt = "You are an assistant who is in charge of answer the user questions." + adjunction
    conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
                },
            {
                "role" : "user",
                "content": [{"type": "text", "text": f"{query}"}]
                }
    ]

    # è necessario avere un file .env in cui aver definito la variabile API_KEY col valore della chiave che stiamo utilizzando

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    API_KEY = os.getenv("API_KEY")
    
    client = InferenceClient(api_key=API_KEY, provider = "novita")

    if RAG:

        relevant_chunks = search_similar(query, embedding_matrix, top_k_chunks = 5)
        conversation[-1]["content"][0]["text"] = f"{query}. Answer the question and be sure to base your answer on the following texts. Do not mention the fact you are using these texts. TEXTS:\n\n" + "\n\n".join(relevant_chunks)

    completion = client.chat.completions.create(
        model = MODEL,
        messages = conversation,
        max_tokens = 1000,
        temperature = TEMPERATURE,
    )

    reply = completion.choices[0].message.content

    timestamp = t.strftime("%Y-%m-%d_%H-%M-%S")
    rag_flag = ""
    knowledge_base = ""
    if RAG: 
        rag_flag = "RAG_"
        knowledge_base = "\n\nKNOWLEDGE_BASE:\n\n" + "\n\n".join(relevant_chunks)

    with open(f"replies/{rag_flag}{timestamp}.txt", "w", encoding = "utf-8") as reply_file:
        reply_file.write(f"SYSTEM PROMPT:\n{system_prompt}\n\nQUERY:\n{query}\n\nREPLY:\n{reply}{knowledge_base}")