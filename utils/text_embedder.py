import os
import warnings
import logging
import numpy as np
import torch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, max_features=1000, vectorizer_path=None):
        """Initialize the text embedder with TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features (terms) to keep
            vectorizer_path: Path to load/save the fitted vectorizer. If None, uses default path.
        """
        # Set default vectorizer path if none provided
        if vectorizer_path is None:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to task2 directory
            task2_dir = os.path.dirname(current_dir)
            vectorizer_path = os.path.join(task2_dir, 'vectorizer.pkl')
            
        self.vectorizer_path = vectorizer_path
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.embedding_dim = max_features
        self._is_fitted = False
        
        # Try to load existing vectorizer
        if os.path.exists(vectorizer_path):
            try:
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self._is_fitted = True
                #print(f"Loaded fitted vectorizer from {vectorizer_path}")
            except Exception as e:
                print(f"Error loading vectorizer: {str(e)}")
        
    def get_paper_text(self, paper_folder):
        """Extract title and abstract from a paper folder"""
        title_path = os.path.join(paper_folder, 'title.txt')
        abstract_path = os.path.join(paper_folder, 'abstract.txt')
        
        title = ""
        abstract = ""
        
        try:
            with open(title_path, 'r', encoding='utf-8', errors='ignore') as f:
                title = f.read().strip().lower()
            with open(abstract_path, 'r', encoding='utf-8', errors='ignore') as f:
                abstract = f.read().strip().lower()
        except:
            pass
            
        return title, abstract
    
    def get_embedding(self, title, abstract):
        """Get embedding for a paper's title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            PyTorch tensor of shape (embedding_dim,)
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted first. Call process_all_papers before getting embeddings.")
            
        text = f"{title} {abstract}"
        embedding = self.vectorizer.transform([text]).toarray()
        return torch.tensor(embedding[0], dtype=torch.float)
    
    def process_all_papers(self, papers_dir):
        """Process all papers and return embeddings for building citation graph.
        
        Args:
            papers_dir: Directory containing paper folders
            
        Returns:
            Dictionary mapping paper_id to PyTorch tensor embedding
        """
        # Get all paper folders
        paper_folders = [os.path.join(papers_dir, pid) for pid in os.listdir(papers_dir) 
                        if os.path.isdir(os.path.join(papers_dir, pid))]
        
        # Process each paper
        texts = []
        paper_ids = []
        for folder in paper_folders:
            try:
                title, abstract = self.get_paper_text(folder)
                if title and abstract:  # Only process if we have both
                    texts.append(f"{title} {abstract}")
                    paper_ids.append(os.path.basename(folder))
            except Exception as e:
                print(f"Error processing {folder}: {str(e)}")
                continue
        
        if not texts:
            return {}
        
        # Fit the vectorizer and get embeddings
        embeddings = self.vectorizer.fit_transform(texts).toarray()
        embeddings = torch.tensor(embeddings, dtype=torch.float)
        
        # Mark vectorizer as fitted
        self._is_fitted = True
        
        # Save the fitted vectorizer
        try:
            os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            #print(f"Saved fitted vectorizer to {self.vectorizer_path}")
        except Exception as e:
            print(f"Error saving vectorizer: {str(e)}")
        
        # Return dictionary mapping paper_id to tensor embedding
        return {pid: emb for pid, emb in zip(paper_ids, embeddings)} 