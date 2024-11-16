from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, VGAE
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

"""
This script trains a Variational Graph Autoencoder (VGAE) on graph data from NebulaGraph, 
reconstructs node features using a custom decoder, and suggests new nodes and edges. It also includes
a function to perform cluster density analysis to identify sparse clusters in the latent space. These clusters
are then used to identify points along paths between clusters for node generation.
""" 

# Connect to NebulaGraph
config = Config()
config.max_connection_pool_size = 10
connection_pool = ConnectionPool()
assert connection_pool.init([('127.0.0.1', 9669)], config)
session = connection_pool.get_session('root', 'nebula')
session.execute('USE danswer_graph')

# Define the encoder for the VGAE model to extract latent
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, heads=4):
        super(Encoder, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.gat_mu = GATConv(hidden_dim * heads, latent_dim, heads=1, concat=False)
        self.gat_logstd = GATConv(hidden_dim * heads, latent_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        # Apply the first GAT layer with ReLU activation
        x = F.relu(self.gat1(x, edge_index))

        # Separate GAT layers for mu and logstd
        mu = self.gat_mu(x, edge_index)
        logstd = self.gat_logstd(x, edge_index)
        return mu, logstd

# Initialize VGAE model with encoder and custom decoder
class VGAEModel(VGAE):
    def __init__(self, encoder):
        super(VGAEModel, self).__init__(encoder)
    
    # Reconstruction loss function for VGAE model
    def custom_loss(self, z, pos_edge_index, neg_edge_index):
        # Decode positive and negative edges
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)

        # Apply BCE loss with sigmoid
        pos_loss = -torch.log(pos_scores + 1e-15).mean()  # Positive edges
        neg_loss = -torch.log(1 - neg_scores + 1e-15).mean()  # Negative edges

        return pos_loss + neg_loss

# VGAE model training
def train_vgae(model, data, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_edge_index = data.edge_index  # Positive edges from the data

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Encode and calculate loss
        z = model.encode(data.x, pos_edge_index)

        # Sample negative edges
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        
        # Calculate loss
        #loss = model.recon_loss(pos_edge_index, neg_edge_index)
        loss = model.custom_loss(z, pos_edge_index, neg_edge_index)
        
        # Backpropagation and optimization
        loss.backward(retain_graph=True)
        optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Function to identify sparse regions for node generation based on cluster density using DBSCAN
def analyze_cluster_density(z, eps=0.5, min_samples=3):    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(z.detach().cpu().numpy())
    labels = db.labels_
    sparse_clusters = {}
    
    # Identify sparse clusters by counting the number of points in each cluster
    for i, label in enumerate(labels):
        if label == -1:
            continue  # Skip noise points
        if label not in sparse_clusters:
            sparse_clusters[label] = []
        sparse_clusters[label].append(i)
    
    # Filter sparse clusters
    sparse_clusters = {k: v for k, v in sparse_clusters.items() if len(v) < min_samples * 2}
    return sparse_clusters

# Function to identify points along paths between clusters for node generation
def suggest_path_based_nodes(z, sparse_clusters):
    new_node_positions = []
    for cluster_id, nodes in sparse_clusters.items():
        # Define paths based on centroid proximity or connections within the cluster
        centroid = torch.mean(z[nodes], dim=0)
        new_node_positions.append(centroid)
    return new_node_positions

# Function to find neighbors for suggested new nodes, to aggregate documents for new document generation
def find_neighbors_for_new_nodes(new_node_positions, z, k=5):
    # Normalize embeddings for cosine similarity
    z_normalized = torch.nn.functional.normalize(z, p=2, dim=1)
    new_node_results = []

    for new_position in new_node_positions:
        # Normalize the new node's embedding
        new_position_normalized = torch.nn.functional.normalize(new_position.unsqueeze(0), p=2, dim=1)

        # Compute cosine similarity with all existing nodes
        similarities = torch.matmul(z_normalized, new_position_normalized.T).squeeze()
        
        # Find indices of the k-nearest neighbors
        top_k_indices = torch.topk(similarities, k=k, largest=True).indices.tolist()
        
        # Query Nebula Graph to retrieve the data for these neighbors
        neighbors_data = []
        for idx in top_k_indices:
            query = f"""
            MATCH (d:Document) WHERE id(d) == {idx} 
            RETURN d;
            """
            result = session.execute(query).rows()
            if result:
                neighbors_data.append(result[0])  # Append the first (and likely only) result

        # Add the results to the list
        new_node_results.append({
            "new_node_position": new_position.tolist(),
            "neighbors": neighbors_data
        })
    
    return new_node_results