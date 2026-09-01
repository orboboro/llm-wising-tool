from chunker import *
from API_calling import *
from rag_system import *
import os

document = "The AI Con - Emily M. Bender.txt"

if not os.listdir("data"):

    # Chunking

    with open(document, "r", encoding = "utf-8") as infile:
        text = infile.read()

    recursive_token_chunker = RecursiveTokenChunker(
        chunk_size = 350,
        chunk_overlap = 50,
        length_function = openai_token_count,
        separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
        )

    recursive_token_chunks = recursive_token_chunker.split_text(text)

    for idx, chunk in enumerate(recursive_token_chunks):
        with open(f"data/chunk_{idx + 1}.txt", "w", encoding = "utf-8") as f:
            f.write(chunk)

        print(f"CHUNK n_{idx}:\n\n{chunk}\n\n================================================================================\n\n")

    # Indexing chunks
    
    data = []
    for chunk in os.listdir("data"):
        with open(f"data/{chunk}", "r", encoding = "utf-8") as c:
            data.append(c.read())

    index_database(data)



embedding_matrix = load_embedding_matrix("embeddings.npy")
query = "What about working exploitation in AI industry?"

query = "What about working exploitation in AI industry?"
huggingface_API_calling(query = query, embedding_matrix = embedding_matrix, model = "meta-llama/Llama-3.3-70B-Instruct", RAG = True)