from mongo import MongoDB
from embeddings.embedding import Embedding

# Khai báo các tham số
model_name = 'keepitreal/vietnamese-sbert'
csv_path = 'hoanghamobile.csv'
output_path = 'hoanghamobile_with_embeddings.csv'
mongo_uri = r"mongodb+srv://sonnguyenhai7:sVyOZuzdCZPC4DDn@cluster0.okt5f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
db_name = "products"
collection_name = "products"

# Tạo đối tượng Embedding
embedding  = Embedding(model_name)
embedding.load_data(csv_path)
embedding.add_embeddings()
embedding.save_to_csv(output_path)

# Tạo đối tượng MongoDB và đưa dữ liệu lên MongoDB
mongo = MongoDB(mongo_uri, db_name, collection_name)
mongo.upload_to_mongo(embedding.data)


