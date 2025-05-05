import os
import re
import networkx as nx
import torch
import chardet
import matplotlib.pyplot as plt

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
    return G 

def analyze_graph(G):
    """Calculate and display all required metrics"""
    # Basic metrics
    num_edges = len(G.edges())
    num_nodes = len(G.nodes())
    isolated = sum(1 for n in G if G.degree(n) == 0)
    
    # Cache degree calculations
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    avg_in = sum(in_degrees.values()) / num_nodes
    avg_out = sum(out_degrees.values()) / num_nodes
    
    
    # Use weakly connected components for directed graph
    undirected = G.to_undirected()
    components = list(nx.weakly_connected_components(G))
    largest_component = max(components, key=len)
    subgraph = G.subgraph(largest_component).to_undirected()
    
    # Use approximate diameter for large graphs
    if len(largest_component) > 1000:
        diameter = nx.approximation.diameter(subgraph)
    else:
        diameter = nx.diameter(subgraph)

    # Plotting with optimized settings
    plt.figure(figsize=(12, 5))
    
    # Degree distribution with optimized bins
    plt.subplot(1, 2, 1)
    degrees = [d for _,d in G.degree()]
    max_degree = max(degrees)
    bins = min(50, max_degree)
    plt.hist(degrees, bins=bins, alpha=0.7)
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    
    # In/Out degree comparison with optimized bins
    plt.subplot(1, 2, 2)
    max_in = max(in_degrees.values())
    max_out = max(out_degrees.values())
    bins = min(30, max(max_in, max_out))
    plt.hist(in_degrees.values(), bins=bins, alpha=0.5, label='In-degree')
    plt.hist(out_degrees.values(), bins=bins, alpha=0.5, label='Out-degree')
    plt.title('In-degree vs Out-degree')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('degree_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print results
    print(f"""
    Citation Graph Analysis Report
    ==============================
    Total Papers (Nodes): {num_nodes}
    Total Citations (Edges): {num_edges}
    
    Isolated Papers: {isolated}
    Percentage Isolated: {isolated/num_nodes:.1%}
    
    Average In-degree: {avg_in:.2f}
    Average Out-degree: {avg_out:.2f}
    
    Graph Diameter: {diameter}
    Largest Component Size: {len(largest_component)}
    """)

G = build_graph_with_embeddings("./dataset_papers")
analyze_graph(G)