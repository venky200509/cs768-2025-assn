import os
import re
import networkx as nx
import torch
import chardet
from torch_geometric.data import Data
from .text_embedder import TextEmbedder

def normalize_title(title):
    """Basic normalization focusing on title text only."""
    title = re.sub(r"[^a-zA-Z0-9]", "", title).strip().lower() # remove non-alphanumeric
    title = re.sub(r"\s+", " ", title).strip()
    return title

def process_paper(folder_path):
    """Process a single paper's bibliography files (only one .bib or .bbl)."""
    # 1) find .bib/.bbl
    bib_files = [f for f in os.listdir(folder_path)
                 if f.endswith(('.bib', '.bbl'))]
    if not bib_files:
        return []
    bib_files.sort(key=lambda x: (not x.lower().endswith('.bib'), x.lower()))
    selected = bib_files[0]
    path = os.path.join(folder_path, selected)

    # 2) read with proper encoding
    with open(path, 'rb') as f:
        raw = f.read(10000)
        enc = chardet.detect(raw)['encoding'] or 'latin-1'
    text = open(path, 'r', encoding=enc, errors='ignore').read()

    found = []
    if selected.lower().endswith('.bib'):
        # Split into entries
        found = re.findall(r"title\s*=\s*{(.*?)}", text, re.DOTALL)
    else:
        # split on each \bibitem
        found = re.findall(r"\\bibitem\{[^}]*\}.*?\\newblock\s+(.*?)[.,]", text, re.DOTALL)

    # normalize and return
    return [normalize_title(t) for t in found]

def build_graph_with_embeddings(papers_dir):
    """Build citation graph with text embeddings"""
    # First get the text embeddings
    embedder = TextEmbedder()
    paper_embeddings = embedder.process_all_papers(papers_dir)
    
    # Build the graph
    G = nx.DiGraph()
    paper_map = {}  # normalized_title -> paper_id
    
    # First pass: Map titles to paper IDs and add nodes with embeddings
    for paper_id in os.listdir(papers_dir):
        title_path = os.path.join(papers_dir, paper_id, "title.txt")
        if os.path.exists(title_path):
            try:
                with open(title_path, 'r', errors='ignore') as f:
                    raw_title = f.read().strip()
                    norm_title = normalize_title(raw_title)
                    paper_map[norm_title] = paper_id
                    G.add_node(paper_id)
            except:
                pass
    
    # Second pass: Process citations
    for paper_id in paper_map.values():
        folder_path = os.path.join(papers_dir, paper_id)
        cited_titles = process_paper(folder_path)
        
        for t in cited_titles:
            if t in paper_map:
                G.add_edge(paper_id, paper_map[t])
    
    # Convert to PyTorch Geometric format
    # Create node features matrix
    num_nodes = len(G.nodes())
    embedding_dim = next(iter(paper_embeddings.values())).shape[0]
    x = torch.zeros((num_nodes, embedding_dim))
    
    # Create node mapping
    node_map = {node: i for i, node in enumerate(G.nodes())}
    node_map_reverse = {i: node for node, i in node_map.items()}
    
    # Fill in the features
    for node in G.nodes():
        if node in paper_embeddings:
            x[node_map[node]] = paper_embeddings[node]  # Already a tensor
    
    # Create edge index
    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_map[u], node_map[v]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    data.node_map = node_map
    data.node_map_reverse = node_map_reverse
    
    return G, data 

