from embeddings.embedding import Embedding
import pymongo
class RAG():
    def __init__(self,embedding,mongodbUri: str,
                 dbName: str,
                 dbCollection: str):
        self.client = pymongo.MongoClient(mongodbUri)
        self.db = self.client[dbName]
        self.collection = self.db[dbCollection]
        self.embedding_model = embedding


    def vector_search(
            self,
            user_query: str,
            limit=2):
        """
                Perform a vector search in the MongoDB collection based on the user query.

                Args:
                user_query (str): The user's query string.

                Returns:
                list: A list of matching documents.
        """

        query_embedding = self.embedding_model.get_embedding(text=user_query)

        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        vector_search_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": limit,
            }
        }
        unset_stage = {
            "$unset": "embedding"
        }
        project_stage = {
            "$project": {
                "_id": 0,
                "title": 1,
                "product_specs": 1,
                "color_options": 1,
                "current_price": 1,
                "product_promotion": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
        pipeline = [vector_search_stage, unset_stage, project_stage]

        # Execute the search
        results = self.collection.aggregate(pipeline)

        return list(results)

    def enhance_prompt(self, query,limit=2):
        get_knowledge = self.vector_search(query, limit)
        enhanced_prompt = ""
        i = 0
        for result in get_knowledge:
            if result.get('current_price'):
                i += 1
                enhanced_prompt += f"\n {i}) Tên: {result.get('title')}"

                if result.get('current_price'):
                    enhanced_prompt += f", Giá: {result.get('current_price')}"
                else:
                    # Mock up data
                    # Retrieval model pricing from the internet.
                    enhanced_prompt += f", Giá: Liên hệ để trao đổi thêm!"
                if result.get('product_specs'):
                    enhanced_prompt += f", Thông số: {result.get('product_specs')}"
                if result.get('product_promotion'):
                    enhanced_prompt += f", Ưu đãi: {result.get('product_promotion')}"

                if result.get('color_options'):
                    enhanced_prompt += f", Màu sắc: {result.get('color_options')}"

        return enhanced_prompt





