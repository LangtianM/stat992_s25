import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

class Interpreter:
    """
    Provides tools to interpret model predictions through feature attribution.
    """
    
    def __init__(self, model, data_processor):
        """
        Initialize the interpreter.
        
        Args:
            model: Trained recommender model
            data_processor: Data processor with dataset information
        """
        self.model = model
        self.data = data_processor
        
    # We need to define a function to predict the recommendation score for a given user along with the user features, so that we can attribute the prediction to the user features.
    def predict_for_attribution(self, user_feat_batch, item_feat_batch, 
                               target_user, target_item):
        """
        Prediction function for attribution methods.
        
        Args:
            user_feat_batch: Batch of user features
            item_feat_batch: Batch of item features
            target_user: User ID to make predictions for
            target_item: Item ID to make predictions for
            
        Returns:
            Tensor of prediction scores
        """
        outputs = []
        batch_size = user_feat_batch.shape[0]
        
        for i in range(batch_size):
            # Get current feature vectors
            cur_user_feat = user_feat_batch[i]
            cur_item_feat = item_feat_batch[i]
            
            # Create modified feature matrices
            user_features_modified = self.data.user_features.clone()
            item_features_modified = self.data.item_features.clone()
            user_features_modified[target_user] = cur_user_feat
            item_features_modified[target_item] = cur_item_feat
            
            # Calculate embeddings and score
            user_embed, item_embed = self.model(
                self.data.adj_matrix, 
                user_features_modified, 
                item_features_modified
            )
            score = torch.sum(user_embed[target_user] * item_embed[target_item])
            outputs.append(score)
        
        return torch.stack(outputs)
    
    def get_attribution(self, target_user, target_item):
        """
        Calculate feature attribution for a specific user-item pair.
        
        Args:
            target_user: User ID to analyze
            target_item: Item ID to analyze
            
        Returns:
            Tuple of (user_attributions, item_attributions)
        """
        # Prepare inputs
        target_user_feat = self.data.user_features[target_user].unsqueeze(0)
        target_item_feat = self.data.item_features[target_item].unsqueeze(0)
        
        # Average features as baseline
        baseline_user_feat = torch.mean(self.data.user_features, dim=0).unsqueeze(0)
        baseline_item_feat = torch.mean(self.data.item_features, dim=0).unsqueeze(0)
        
        # Wrapper function for attribution
        def wrapped_predict(user_feat, item_feat):
            return self.predict_for_attribution(
                user_feat, item_feat, target_user, target_item)
        
        # Calculate attributions using Integrated Gradients
        ig = IntegratedGradients(wrapped_predict)
        with torch.no_grad():
            attributions, _ = ig.attribute(
                inputs=(target_user_feat, target_item_feat),
                baselines=(baseline_user_feat, baseline_item_feat),
                return_convergence_delta=True
            )
        
        return attributions
    
    def find_top_recommendations(self, user_index, k=5):
        """
        Find top-k recommendations for a user.
        
        Args:
            user_index: User ID to get recommendations for
            k: Number of recommendations to return
            
        Returns:
            List of recommended item IDs
        """
        # Get embeddings
        with torch.no_grad():
            user_embed, item_embed = self.model(
                self.data.adj_matrix, 
                self.data.user_features, 
                self.data.item_features
            )
        
            # Calculate scores and get top-k
            scores = torch.matmul(user_embed[user_index], item_embed.T)
            top_scores, top_indices = torch.topk(scores, k)
        
        top_indices = top_indices.numpy()
        top_scores = top_scores.detach().numpy()
        
        # Print results
        for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
            movie = self.data.movies.iloc[idx]
            print(f"\nTop {i+1} Recommendation")
            print(f"Title: {movie['title']}")
            print(f"Genres: {movie['genres']}")
            print(f"Score: {score:.4f}")
            print(f"Movie index: {idx}")
        
        return top_indices

