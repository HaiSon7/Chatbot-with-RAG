from typing import List

import pandas as pd
from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self, model_name: str):
        # Khởi tạo các tham số
        self.model = SentenceTransformer(model_name)



    def load_data(self,csv_path):
        """Đọc dữ liệu từ file CSV."""
        self.data = pd.read_csv(csv_path)

    def get_embedding(self, text: str):
        """Lấy embedding cho một văn bản."""
        if isinstance(text, str) and text.strip():  # Kiểm tra nếu text là chuỗi và không rỗng
            embedding = self.model.encode(text)
            return embedding.tolist()
        else:
            print("Attempted to get embedding for empty or non-string text.")
            return []

    def get_list_embedding(self, docs: List[str]):
        embeddings = []
        for doc in docs:
            if isinstance(doc, str) and doc.strip():  # Kiểm tra nếu text là chuỗi và không rỗng
                embedding = self.model.encode(doc)
                embeddings.append(embedding.tolist())
            else:
                print("Attempted to get embedding for empty or non-string text.")
                return []
        return embeddings
    def add_embeddings(self):
        """Lấy embedding cho từng dòng trong cột 'title' và thêm vào DataFrame."""
        self.data['embedding'] = self.data['title'].apply(self.get_embedding)

    def save_to_csv(self, output_path: str):
        """Ghi dữ liệu vào file CSV mới."""
        self.data.to_csv(output_path, index=False)
        print("Embeddings have been added and saved to", output_path)
