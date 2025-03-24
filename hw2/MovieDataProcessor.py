import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

class MovieDataProcessor:
    """
    Handles data loading and preprocessing for recommendation systems.
    
    This class is responsible for loading user, item, and interaction data,
    creating mappings between original IDs and internal indices, building
    adjacency matrices, and preparing feature tensors for the model.
    
    Parameters
    ----------
    ratings_path : str
        Path to the file containing user-item interactions/ratings data.
        Expected format is a delimiter-separated file with user, item, rating columns.
    
    movies_path : str
        Path to the file containing movie/item metadata.
        Expected format is a delimiter-separated file with item ID, title, genres columns.
    
    users_path : str
        Path to the file containing user metadata.
        Expected format is a delimiter-separated file with user ID, demographics columns.
    
    Attributes
    ----------
    user_mapping : dict
        Mapping from original user IDs to internal indices.
    
    item_mapping : dict
        Mapping from original item IDs to internal indices.
    
    genre_mapping : dict
        Mapping from genre names to internal indices.
    
    inv_genre_mapping : dict
        Inverse mapping from genre indices to names.
    
    data : pandas.DataFrame
        Processed interaction data with mapped user and item indices.
    
    movies : pandas.DataFrame
        Processed movie data with mapped item indices.
    
    users : pandas.DataFrame
        Processed user data with mapped user indices.
    
    num_users : int
        Total number of unique users.
    
    num_items : int
        Total number of unique items.
    
    user_features : torch.Tensor
        Tensor of user features (gender, age, occupation one-hot encoding).
    
    item_features : torch.Tensor
        Tensor of item features (genre multi-hot encoding).
    
    adj_matrix : torch.Tensor
        Normalized adjacency matrix for the user-item interaction graph.
    
    index_to_occ : dict
        Mapping from occupation indices to occupation names.
    """
    
    def __init__(self, ratings_path, movies_path, users_path):
        """
        Initialize the data processor.
        
        Args:
            ratings_path: Path to ratings data
            movies_path: Path to movies data
            users_path: Path to users data
        """
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.users_path = users_path
        
        # Mapping dictionaries
        self.user_mapping = None
        self.item_mapping = None
        self.genre_mapping = None
        self.inv_genre_mapping = None
        
        # Processed data
        self.data = None
        self.movies = None
        self.users = None
        self.num_users = 0
        self.num_items = 0
        
        # Feature tensors
        self.user_features = None
        self.item_features = None
        self.adj_matrix = None
        
        # Occupation mapping
        self.index_to_occ = {
            0: "other or not specified",
            1: "academic/educator",
            2: "artist",
            3: "clerical/admin",
            4: "college/grad student",
            5: "customer service",
            6: "doctor/health care",
            7: "executive/managerial",
            8: "farmer",
            9: "homemaker",
            10: "K-12 student",
            11: "lawyer",
            12: "programmer",
            13: "retired",
            14: "sales/marketing",
            15: "scientist",
            16: "self-employed",
            17: "technician/engineer",
            18: "tradesman/craftsman",
            19: "unemployed",
            20: "writer"
        }
        
    def load_data(self):
        """Load and preprocess datasets."""
        # Load raw data 
        self.data = pd.read_csv(self.ratings_path, sep="::", 
                              names=["user", "item", "rating", "timestamp"], 
                              engine="python")
        self.movies = pd.read_csv(self.movies_path, sep="::", 
                                names=["item", "title", "genres"], 
                                engine="python", encoding="ISO-8859-1")
        self.users = pd.read_csv(self.users_path, sep="::", 
                               names=["user", "gender", "age", "occupation", "zip"], 
                               engine="python")
        
        # Create ID mappings
        self.user_mapping = {old: new for new, old in enumerate(self.users["user"].unique())}
        self.item_mapping = {old: new for new, old in enumerate(self.movies["item"].unique())}
        
        # Apply mappings
        self.data["user"] = self.data["user"].map(self.user_mapping)
        self.data["item"] = self.data["item"].map(self.item_mapping)
        self.movies["item"] = self.movies["item"].map(self.item_mapping)
        self.users["user"] = self.users["user"].map(self.user_mapping)
        
        self.num_users = len(self.user_mapping)
        self.num_items = len(self.item_mapping)
        
        # Process features
        self._process_user_features()
        self._process_item_features()
        self._build_adj_matrix()
        
        return self
        
    def _process_user_features(self):
        """Process user features (gender, age, occupation)."""
        # Map gender to binary
        gender_map = {'M': 0, 'F': 1}
        self.users['gender'] = self.users['gender'].map(gender_map)
        
        # One-hot encode occupation
        occupation = pd.get_dummies(self.users['occupation'].map(self.index_to_occ))
        self.users = pd.concat([self.users, occupation], axis=1)
        self.users.drop(['occupation', 'zip'], axis=1, inplace=True)
        
        # Convert to tensor
        self.user_features = torch.FloatTensor(
            self.users.iloc[:, 1:].values.astype(np.float32))
    
    def _process_item_features(self):
        """Process item features (movie genres)."""
        # Extract unique genres
        genres_set = set('|'.join(self.movies['genres']).split('|'))
        self.genre_mapping = {genre: i for i, genre in enumerate(genres_set)}
        self.inv_genre_mapping = {i: genre for genre, i in self.genre_mapping.items()}
        
        # Encode genres as multi-hot vectors
        self.movies['genres_encoded'] = self.movies['genres'].apply(
            lambda x: [self.genre_mapping[g] for g in x.split('|')])
        
        # Create genre feature matrix
        self.item_features = torch.zeros((self.num_items, len(genres_set)))
        for i, row in self.movies.iterrows():
            for g in row['genres_encoded']:
                self.item_features[row['item'], g] = 1
    
    def _build_adj_matrix(self):
        """Build and normalize the adjacency matrix for the user-item graph."""
        rows, cols = self.data['user'].values, self.data['item'].values
        interactions = self.data['rating'].values.astype(np.float32)
        
        # Create sparse adjacency matrix
        adj = sp.coo_matrix((interactions, (rows, cols + self.num_users)), 
                           shape=(self.num_users + self.num_items, 
                                  self.num_users + self.num_items))
        
        # Make the graph undirected
        adj = adj + adj.T
        
        # Normalize the adjacency matrix
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # Convert to dense tensor
        self.adj_matrix = torch.FloatTensor(normalized_adj.toarray())
