from sentence_transformers import SentenceTransformer
import os

class Embedder:
    def __init__(self, model_path=None):
        # 默认路径为你的本地模型路径
        if model_path is None:
            model_path = "/Users/heelgoed/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        self.model = SentenceTransformer(model_path)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
