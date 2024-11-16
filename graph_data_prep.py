from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.gclient.net import Session
from torch_geometric.data import Data
import torch
from transformers import AutoTokenizer, AutoModel

"""
This script interfaces with the NebulaGraph knowledge graph to query data. The script processes the 
queried data to prepare it for use with PyTorch Geometric by creating feature vectors for document nodes 
and an edge index. The main function prepares data for a Variational Graph Autoencoder (VGAE) 
by assembling these features and edges into a PyTorch Geometric Data object. It also provides
functions to insert new nodes and edges into the NebulaGraph database.
"""

# Connect to NebulaGraph
config = Config()
config.max_connection_pool_size = 10
connection_pool = ConnectionPool()
assert connection_pool.init([('127.0.0.1', 9669)], config)
session = connection_pool.get_session('root', 'nebula')
session.execute('USE danswer_graph')

# Load pre-trained language model for text embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Helper function to embed text with a pre-trained model
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Average pooling to get a fixed-size vector

# Function to retrieve Document nodes
def get_documents():
    query = """
    MATCH (d:Document) RETURN d
    """
    result = session.execute(query)
    documents = result.rows()
    return documents

# Function to retrieve edges for each edge type
def get_edges(edge_label):
    query = f"""
    MATCH (d1:Document)-[r:{edge_label}]->(d2:Document) RETURN d1 AS src, d2 AS dst;
    """
    result = session.execute(query)
    edges = [(res.values[0].get_vVal().vid.get_iVal(), res.values[1].get_vVal().vid.get_iVal()) for res in result.rows()]
    return edges

# Document processing to create nodes with feature embeddings
def process_documents(documents):
    nodes = {}
    x_data = []
    
    for doc in documents:

        # Access the Vertex object using the get_vVal() method
        vertex = doc.values[0].get_vVal()

        # Access the properties as before
        props = vertex.tags[0].props

        doc_id = vertex.vid.get_iVal()
        title = props[b'title'].get_sVal().decode("utf-8")
        content = props[b'content'].get_sVal().decode("utf-8")
        source = props[b'source'].get_sVal().decode("utf-8")
        timestamp = props[b'doc_timestamp'].get_dtVal()
        tags = props[b'doc_tags'].get_sVal().decode("utf-8")

        # Embed title and content with a pre-trained model
        title_embedding = embed_text(title)
        content_embedding = embed_text(content)

        # One-hot encoding for source if it's categorical
        source_encoding = torch.tensor([1.0 if source == "source_A" else 0.0, 1.0 if source == "source_B" else 0.0])

        # Convert timestamp to a relative age (e.g., days since publication)
        current_year = 2024
        #publication_year = int(timestamp[:4])
        publication_year = timestamp.year  # Access the year attribute directly
        relative_time = torch.tensor([current_year - publication_year], dtype=torch.float)

        # Process tags as a tag count or convert them to embeddings if tag vocabulary is available
        #tag_count = len(json.loads(tags)) if tags else 0
        tag_count = len(tags.split(','))
        tag_count_tensor = torch.tensor([tag_count], dtype=torch.float)

        # Concatenate all features to form a single feature vector
        feature_vector = torch.cat([title_embedding, content_embedding, source_encoding, relative_time, tag_count_tensor])
        
        # Update node index and feature data
        nodes[doc_id] = len(nodes)
        x_data.append(feature_vector)

    x = torch.stack(x_data)
    return nodes, x

# Process edges to create edge_index tensor for PyTorch Geometric
def process_edges(nodes):
    edge_index_data = []
    edge_types = ["reference", "topic_association", "authored_by_same_person",
                  "update_dependency", "part_of", "created_same_day"]
    
    for edge_type in edge_types:
        edges = get_edges(edge_type)
        for src, dst in edges:
            if src in nodes and dst in nodes:
                edge_index_data.append([nodes[src], nodes[dst]])

    edge_index = torch.tensor(edge_index_data, dtype=torch.long).t().contiguous()
    return edge_index

# Insert a document node into NebulaGraph
def insert_document(doc_id, title, content, source, timestamp, tags, metadata):
    query = f"""
    INSERT VERTEX Document (id, title, content, source, timestamp, tags, metadata)
    VALUES "{doc_id}": ("{doc_id}", "{title}", "{content}", "{source}", datetime("{timestamp}"), "{tags}", "{metadata}");
    """
    session.execute(query)


# Insert an edge between two documents in NebulaGraph
def insert_edge(edge_type, src_id, dst_id):
    query = f'INSERT EDGE {edge_type} () VALUES "{src_id}"->"{dst_id}": ();'
    session.execute(query)

# Main function to prepare data for VGAE
def prepare_data():
    documents = get_documents()
    nodes, x = process_documents(documents)
    edge_index = process_edges(nodes)

    # Return a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    return data