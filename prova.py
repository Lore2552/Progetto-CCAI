import chromadb

chroma_client = chromadb.PersistentClient(path=r"./chroma_db")
collection = chroma_client.get_or_create_collection("ricette_giallozafferano")
print(f"Documenti presenti: {collection.count()}")
