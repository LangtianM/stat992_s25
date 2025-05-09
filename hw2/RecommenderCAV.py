
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class RecommenderCAV:
    def __init__(self, processor, model):
        """
        processor: instance of MovieDataProcessor
        user_embeddings: numpy array [num_users, dim]
        item_embeddings: numpy array [num_items, dim]
        """
        self.processor = processor
        self.user_embeddings = model.final_user_embedding
        self.item_embeddings = model.final_item_embedding

    def train_cav(self, concept_genre="Action"):
        """
        Train a concept activation vector (CAV) for a given genre.
        """
        genre_indices = self.processor.genre_mapping
        genre_column = genre_indices.get(concept_genre, None)
        if genre_column is None:
            raise ValueError(f"Genre '{concept_genre}' not found in genre mapping.")

        # Get binary labels for concept genre
        labels = self.processor.item_features[:, genre_column].numpy().astype(int)

        valid_idx = np.where((labels == 0) | (labels == 1))[0]
        X_cav = self.item_embeddings[valid_idx]
        y_cav = labels[valid_idx]
        
        X_train, X_test, y_train, y_test = train_test_split(X_cav, y_cav, test_size=0.2, random_state=42)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        self.conf_matrix = conf_matrix
        self.cav = clf.coef_.flatten()
        self.concept_genre = concept_genre
        return self.cav