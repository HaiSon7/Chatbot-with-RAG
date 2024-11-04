import numpy as np
import joblib
class SemanticRouter():
    def __init__(self,embedding):
        self.embedding = embedding
        self.model_classification = joblib.load('semantic_router/logistic_model.pkl')

    def guide(self,query):
        query = np.array(self.embedding.get_embedding(query)).reshape(1,-1)
        #0:product
        #1:chitchat
        cls = self.model_classification.predict(query)
        return cls




