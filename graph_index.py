from storage import VectorStore

def build_linear_graph(chunks, store: VectorStore):
    for i in range(len(chunks) - 1):
        cid = chunks[i]['id']
        nid = chunks[i + 1]['id']
        store.add_edge(cid, nid)
        store.add_edge(nid, cid)
