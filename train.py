import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from torch_geometric.utils import negative_sampling
from models.graph_sage import GraphSAGE
from models.link_predictor import LinkPredictor
from utils.graph_builder import build_graph_with_embeddings

# Optional plotting for LR finder
import matplotlib.pyplot as plt


def train(model, predictor, data, optimizer, batch_size=1024):
    model.train()
    predictor.train()
    total_loss = 0.0

    # Shuffle positive edges
    pos_edge_index = data.edge_index  # shape: [2, num_edges]
    num_edges = pos_edge_index.size(1)
    perm = torch.randperm(num_edges, device=data.edge_index.device)
    pos_edge_index = pos_edge_index[:, perm]

    batch_size = batch_size

    neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=num_edges
        )
    for start in range(0, num_edges, batch_size):
        optimizer.zero_grad()
        
        # Compute embeddings for this batch
        x = model(data.x, data.edge_index)  # data.x shape: [num_nodes, in_channels], x shape: [num_nodes, out_channels]
        
        end = min(start + batch_size, num_edges)
        batch_pos = pos_edge_index[:, start:end]  # shape: [2, batch_size]
        neg_edge_index1 = neg_edge_index[:, start:end]
        # Negative sampling
          # shape: [2, batch_size]

        # Gather src/dst embeddings
        pos_src = x[batch_pos[0]]  # shape: [batch_size, out_channels]
        pos_dst = x[batch_pos[1]]  # shape: [batch_size, out_channels]
        neg_src = x[neg_edge_index1[0]]  # shape: [batch_size, out_channels]
        neg_dst = x[neg_edge_index1[1]]  # shape: [batch_size, out_channels]

        # Vectorized prediction for positive and negative edges
        pos_pred = predictor(pos_src, pos_dst)  # shape: [batch_size, 1]
        neg_pred = predictor(neg_src, neg_dst)  # shape: [batch_size, 1]

        # Compute loss
        pos_loss = -torch.log(pos_pred + 1e-15).mean()  # scalar
        neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()  # scalar
        batch_loss = pos_loss + neg_loss  # scalar

        # Backprop and step per batch
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * (end - start)

    return total_loss / num_edges





def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_papers')
    
    # Check if graph data is already saved
    graph_data_path = os.path.join(os.path.dirname(__file__), 'graph_data.pth')
    if os.path.exists(graph_data_path):
        print("Loading pre-built graph data...")
        graph_data = torch.load(graph_data_path)
        G = graph_data['G']
        data = graph_data['data']
        print("Graph data loaded successfully!")
    else:
        print("Building graph data...")
        G, data = build_graph_with_embeddings(dataset_path)
        # Save graph data
        torch.save({
            'G': G,
            'data': data
        }, graph_data_path)
        print("Graph data saved successfully!")

    # Move data to device
    data.x, data.edge_index = data.x.to(device), data.edge_index.to(device)

    embedding_dim = data.x.size(1)

    model = GraphSAGE(
        in_channels=embedding_dim,
        hidden_channels=64,
        out_channels=32
    ).to(device)
    predictor = LinkPredictor(
        in_channels=32,
        hidden_channels=16,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.01
    )

    num_epochs, batch_size = 100, 1024
    for epoch in range(num_epochs):
        loss = train(model, predictor, data, optimizer, batch_size=batch_size)
        print(f'Epoch: {epoch}, Loss: {loss:.4f}')

    # Compute and save final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(data.x, data.edge_index)
    
    model_path = os.path.join(os.path.dirname(__file__), 'link_prediction_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'final_embeddings': final_embeddings,
        'node_map': data.node_map,
        'node_map_reverse': data.node_map_reverse
    }, model_path)

if __name__ == "__main__":
    main()
