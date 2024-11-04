from pymongo import MongoClient

class MongoDB:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        self.client = MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection_name]

    def upload_to_mongo(self, data):
        """Chèn dữ liệu vào MongoDB."""
        data_records = data.to_dict(orient="records")
        try:
            self.collection.insert_many(data_records)
            print("Dữ liệu đã được đưa lên MongoDB Atlas thành công!")
        except Exception as e:
            print(f"Error occurred: {e}")
