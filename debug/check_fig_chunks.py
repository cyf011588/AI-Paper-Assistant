import sys; sys.path.insert(0, 'src')
import chromadb
from config import DB_DIR
client = chromadb.PersistentClient(path=str(DB_DIR))
col = client.get_collection('paper_chunks')
results = col.query(query_texts=['滞回曲线'], n_results=10)
for i, id_ in enumerate(results['ids'][0]):
    sec = results['metadatas'][0][i]['section_path'][:60]
    print(f"{i+1}. {id_[:50]}")
    print(f"   {sec}")
