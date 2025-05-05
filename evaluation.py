import torch
import os
import sys
from models.graph_sage import GraphSAGE
from models.link_predictor import LinkPredictor
from utils.graph_builder import build_graph_with_embeddings
from utils.text_embedder import TextEmbedder
import argparse

class CitationPredictor:
    def __init__(self, model_path=None):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(f"Using device: {self.device}")
        
        # Get the path to dataset_papers (one level up from task2)
        self.dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_papers')
        
        # Set default model path if none provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'link_prediction_model.pth')
        
        # Initialize embedder with default vectorizer path
        self.embedder = TextEmbedder()
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train the model first by running train.py"
            )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load precomputed embeddings and mappings
        self.existing_embeddings = checkpoint['final_embeddings'].to(self.device)
        self.node_map = checkpoint['node_map']
        self.node_map_reverse = checkpoint['node_map_reverse']
        
        # Initialize models with correct dimensions
        # Use 384 for text embedding dimension (from TextEmbedder)
        self.model = GraphSAGE(
            in_channels=1000,  # Text embedding dimension
            hidden_channels=64,
            out_channels=32,
            
        ).to(self.device)
        
        self.predictor = LinkPredictor(
            in_channels=32,
            hidden_channels=16,
            
        ).to(self.device)
        
        # Load trained model states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        
    def predict(self, paper_folder_path, k=20):
        """Predict top-k papers that should cite the given paper."""
        # Get text embedding for the new paper
        title, abstract = paper_folder_path
        new_text_embedding = self.embedder.get_embedding(title, abstract)
        
        # Move to device
        new_text_embedding = new_text_embedding.to(self.device)
            
        # Create a dummy edge index for the new node
        dummy_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Get predictions
        with torch.no_grad():
            self.model.eval()
            self.predictor.eval()
            
            # Get embedding for the new paper using model's forward pass
            new_embedding = self.model(new_text_embedding.unsqueeze(0), dummy_edge_index)
            new_embedding = new_embedding.squeeze(0)
            
            # Get predictions for new paper with all existing papers
            scores = []
            for i in range(self.existing_embeddings.size(0)):
                # Get embedding for this paper
                paper_embedding = self.existing_embeddings[i]
                # Ensure it's 1D
                if paper_embedding.dim() > 1:
                    paper_embedding = paper_embedding.squeeze(0)
                
                # For directed prediction, we want to predict if new_paper → existing_paper
                score = self.predictor(new_embedding, paper_embedding)
                scores.append((i, score.item()))
        
        # Sort and get top-k predictions
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:k]
        
        # Return paper IDs and scores
        return [(self.node_map_reverse[i], score) for i, score in top_k]

def main():
    # Get paper folder path from command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()
    paper_text = (args.test_paper_title, args.test_paper_abstract)
    
    # Initialize predictor
    predictor = CitationPredictor()
    
    # Get predictions
    predictions = predictor.predict(paper_text, k=20)
    result = []
    # Print predictions in required format
    for paper_id, score in predictions:
        result.append(paper_id)
    print('\n'.join(result))

if __name__ == "__main__":
    main()