import torch
import torch.nn as nn
import math

class LightGCN(nn.Module):
    """
    Light Graph Convolutional Network for recommendation systems.
    
    Parameters
    ----------
    num_users : int
        Total number of users in the dataset. Used to define the size of user embeddings.
    
    num_items : int
        Total number of items in the dataset. Used to define the size of item embeddings.
    
    embedding_dim : int
        Dimension of user and item embeddings. Higher dimensions can capture more complex
        patterns but require more data and computation.
    
    n_layers : int
        Number of graph convolutional layers. More layers allow information to propagate
        further in the graph but may lead to over-smoothing with too many layers.
    
    user_feat_dim : int
        Dimension of raw user features. Used to define the projection from user features
        to embedding space.
    
    item_feat_dim : int
        Dimension of raw item features. Used to define the projection from item features
        to embedding space.
    
    Attributes
    ----------
    user_embedding : nn.Embedding
        Learnable embeddings for users.
    
    item_embedding : nn.Embedding
        Learnable embeddings for items.
    
    user_feat_fc : nn.Linear
        Linear projection from user features to embedding space.
    
    item_feat_fc : nn.Linear
        Linear projection from item features to embedding space.
        
    final_user_embedding : torch.Tensor
        Final user embeddings after encoding the feature information.
        
    final_item_embedding : torch.Tensor
        Final item embeddings after encoding the feature information.
    """
    
    
    def __init__(self, num_users, num_items, embedding_dim, n_layers, 
                 user_feat_dim, item_feat_dim):
        """
        Initialize the LightGCN model.
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Dimension of embeddings
            n_layers: Number of graph convolutional layers
            user_feat_dim: Dimension of user features
            item_feat_dim: Dimension of item features
        """
        super(LightGCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Feature projections
        self.user_feat_fc = nn.Linear(user_feat_dim, embedding_dim)
        self.item_feat_fc = nn.Linear(item_feat_dim, embedding_dim)
        
        # Final embeddings with feature informaton encoded
        self.final_user_embedding = None
        self.final_item_embedding = None
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        # Modified Xavier initialization for embeddings since it dont have an "input dimension"
        a = math.sqrt(6 / (1 + self.embedding_dim))
        nn.init.uniform_(self.user_embedding.weight, -a, a)
        nn.init.uniform_(self.item_embedding.weight, -a, a)
        
        # Xavier initialization for feature projections
        nn.init.xavier_uniform_(self.user_feat_fc.weight)
        nn.init.xavier_uniform_(self.item_feat_fc.weight)

    def forward(self, adj, user_features, item_features):
        """
        Forward pass through the model.
        
        Args:
            adj: Adjacency matrix
            user_features: User feature tensor
            item_features: Item feature tensor
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
    
        # Standardize features
        user_features = (user_features - user_features.mean(dim=0)) / (user_features.std(dim=0) + 1e-8)
        item_features = (item_features - item_features.mean(dim=0)) / (item_features.std(dim=0) + 1e-8)
        
        # Project features to embedding space
        user_feat_embed = self.user_feat_fc(user_features)
        item_feat_embed = self.item_feat_fc(item_features)
        
        # Combine embeddings with features
        user_combined = torch.layer_norm(
            self.user_embedding.weight + user_feat_embed, [self.embedding_dim])
        item_combined = torch.layer_norm(
            self.item_embedding.weight + item_feat_embed, [self.embedding_dim])
        
        # Initialize embeddings for message passing
        all_embeddings = torch.cat([user_combined, item_combined], dim=0)
        embeddings_list = [all_embeddings]
        
        # Multi-layer graph convolution
        for _ in range(self.n_layers):
            all_embeddings = torch.mm(adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Average embeddings from all layers
        final_embedding = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)
        
        # Split user and item embeddings
        user_final, item_final = torch.split(
            final_embedding, [user_combined.shape[0], item_combined.shape[0]])
        
        return user_final, item_final
    
    def get_final_embeddings(self, adj, user_features, item_features):
        """Get the final user and item embeddings with feature information encoded."""
        with torch.no_grad():
            self.final_user_embedding, self.final_item_embedding = self(
                adj, user_features, item_features)

        return self.final_user_embedding, self.final_item_embedding
    
    
