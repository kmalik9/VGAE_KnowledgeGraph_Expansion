from graph_data_prep import *
from vgae_model import *
from doc_generator import *

"""
This script orchestrates the process of preparing data for a Variational Graph Autoencoder (VGAE), 
training the model, encoding the graph data, performing cluster density analysis, suggesting new node positions, 
and generating content for new nodes. It also includes functions to insert new nodes and edges into the NebulaGraph database.
"""

def main():
    # Step 1: Prepare Data
    print("Preparing data...")
    data = prepare_data()
    print("Data prepared.")
    
    # Step 2: Initialize VGAE Model
    print("Initializing VGAE model...")
    in_channels = data.num_features  # Assumes data.x contains the feature vectors
    hidden_dim = 64
    latent_dim = 32
    encoder = Encoder(in_channels, hidden_dim, latent_dim)
    model = VGAEModel(encoder)
    print("Model initialized.")
    
    # Step 3: Train VGAE Model
    print("Training VGAE model...")
    epochs = 20
    learning_rate = 0.001
    train_vgae(model, data, epochs=epochs, lr=learning_rate)
    print("Model training completed.")
    
    # Step 4: Encode the graph data to get embeddings
    model.eval()
    z = model.encode(data.x, data.edge_index)
    print("Encoded node embeddings obtained.")
    
    # Step 5: Perform Cluster Density Analysis
    print("Performing cluster density analysis...")
    sparse_clusters = analyze_cluster_density(z, eps=0.5, min_samples=3)
    print('Cluster density analysis completed. Sparse clusters identified:', sparse_clusters)

    # Step 6: Suggest Positions for New Nodes
    print("Suggesting new node positions...")
    new_node_positions = suggest_path_based_nodes(z, sparse_clusters)
    print("New node positions suggested: ", new_node_positions)

    # Step 7: Generate Content for New Nodes
    print("Generating content for new nodes using OpenAI API...")
    new_nodes = find_neighbors_for_new_nodes(new_node_positions, z, k=5)

    # Step 8: Output the new documents
    print("Newly generated nodes with content:")
    new_documents = []
    for doc in new_nodes:
        prompt_content = []
        for neighbor in doc['neighbors']:
            title = neighbor.values[0].get_vVal().tags[0].props[b'title'].get_sVal().decode("utf-8")
            content = neighbor.values[0].get_vVal().tags[0].props[b'content'].get_sVal().decode("utf-8")
            prompt_content.append({"title": title, "content": content})
        ai_generated_content = generate_structured_document_content(prompt_content)
        new_documents.append(ai_generated_content)
    print(new_documents)

    # Step 9: We can now insert the new documents into the graph
    #print("Inserting new documents into the graph...")
    #insert_document(new_documents, sparse_clusters)
    #print("New documents inserted into the graph.")


# run the main function
if __name__ == "__main__":
    main()