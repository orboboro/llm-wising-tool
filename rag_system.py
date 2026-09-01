import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import BGEM3FlagModel
import os
print("Caricamento del modello di embeddings:")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Funzione per creare e salvare gli embeddings
def index_database(data):
    embeddings = model.encode(data, batch_size=12, max_length=8192,)['dense_vecs']
    np.save("embeddings.npy", embeddings)

# Funzione per caricare la matrice di embedding già presente in locale
def load_embedding_matrix(embedding_matrix_path):
    print("Caricamento embedding matrix da file...")
    loaded_embeddings = np.load(embedding_matrix_path)
    print(f"Shape embedding matrix: {loaded_embeddings.shape}")
    return loaded_embeddings

# Funzione per risalire dall'indice del chunk al testo vero e proprio:
def idx_to_text(idx):

    with open(f"data/chunk_{idx}.txt", "r", encoding="utf-8") as chunk:
        return chunk.read()

# Funzione per calcolare la similarità tra query e chunks (restituisce l'indice dei k chunks più simili):
def search_similar(query, embedding_matrix, top_k_chunks):

    print("Calcolo embedding della query")
    query_embedding = model.encode(query, batch_size=12, max_length=8192,)['dense_vecs']
    print("Embedding della query completato")
    print("Calcolo similarità")
    similarities = cosine_similarity([query_embedding], embedding_matrix)[0]
    print("Calcolo similarità completato")
    sorted_similar_indexes = sorted(enumerate(similarities), key = lambda x : x[1], reverse = True)
    
    print(f"Query: {query}\n")

    for idx, similarity in sorted_similar_indexes:
        print(f"Idx chunk: '{idx}' --- Similarità: '{similarity: .4f}'")

    top_similar_indexes = sorted_similar_indexes[:top_k_chunks]
    top_similar_chunks = [idx_to_text(idx + 1) for idx, similarity in top_similar_indexes]

    print(top_similar_chunks)

    return top_similar_chunks

# Funzione per visualizzare gli embeddings nello spazio vettoriale
def plot_embeddings(query, data, embedding_matrix):

    query_embedding = model.encode(query, batch_size=12, max_length=8192,)['dense_vecs']
    jointed_matrix = np.vstack([query_embedding, embedding_matrix])

    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    embeddings_2d = tsne.fit_transform(jointed_matrix)

    query_2d = embeddings_2d[0]
    sentences_2d = embeddings_2d[1:]

    plt.figure(figsize=(12, 8))

    plt.scatter(
        sentences_2d[:, 0],
        sentences_2d[:, 1],
        s=100,
        label="Documenti"
    )

    plt.scatter(
        query_2d[0],
        query_2d[1],
        s=200,
        c="red",
        marker="o",
        label="Query",
        zorder=5
    )

    for i, sentence in enumerate(data):
        plt.annotate(
            sentence,
            (
                sentences_2d[i, 0],
                sentences_2d[i, 1]
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9
        )

    plt.annotate(
        f"QUERY: {query}",
        (query_2d[0], query_2d[1]),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=10,
        color="red"
    )

    plt.title("t-SNE visualization of BGE-M3 embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "embeddings_tsne.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# Applicazione
"""
if not
    data = []
    for chunk in os.listdir("data"):
        with open(f"data/{chunk}", "r", encoding = "utf-8") as c:
            data.append(c.read())

    index_database(data)
    """


#Trial:
"""
query = "Quanti pneumatici ha una bicicletta?"
data = ["Ricetta del polpettone alla milanese","è ora di pranzo","Che sport il ciclismo"]

index_database(data)

matrix = load_embedding_matrix("embeddings.npy")
plot_embeddings(query, data, matrix)
out = search_similar(query, matrix, 2)
"""
