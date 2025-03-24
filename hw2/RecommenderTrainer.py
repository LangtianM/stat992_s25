import  torch
import torch.optim as optim
import torch.nn as nn

class RecommenderTrainer:
    """Handles training and evaluation of recommender models."""
    
    def __init__(self, model, data_processor, learning_rate=0.01):
        """
        Initialize the trainer.
        
        Args:
            model: The recommender model
            data_processor: Data processor containing the dataset
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.data = data_processor
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self, epochs=5):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
        """
        self.model.train()
        
        for epoch in range(epochs):
            # Get embeddings
            user_embed, item_embed = self.model(
                self.data.adj_matrix, 
                self.data.user_features, 
                self.data.item_features
            )
            
            # Compute loss
            loss = 0
            for user, item, rating in zip(
                self.data.data['user'], 
                self.data.data['item'], 
                self.data.data['rating']
            ):
                user_vec = user_embed[user]
                item_vec = item_embed[item]
                score = torch.sum(user_vec * item_vec)
                loss += self.criterion(score, torch.tensor(float(rating)))
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
        self.model.get_final_embeddings()
    
    def save_model(self, path):
        """Save model weights to file."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights from file."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
