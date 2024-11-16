# VGAE_KnowledgeGraph_Expansion

**VGAE_KnowledgeGraph_Expansion** is a Python-based repository that leverages **NebulaGraph** and **PyTorch Geometric** to identify knowledge gaps in a knowledge graph, suggest new node positions in the latent embedding space, and generate structured content for the proposed nodes using **OpenAI's language models**.

This repository is designed to enhance knowledge graphs by dynamically identifying underrepresented regions, generating new nodes, and seamlessly integrating them into the graph structure. The core functionality revolves around training a **Variational Graph Autoencoder (VGAE)** to analyze graph embeddings and performing cluster-based analysis to drive graph expansion.

## Features

- **Knowledge Gap Identification**:
  - Use VGAE latent space to identify sparse regions in the graph.
  - Perform cluster density analysis to suggest new node positions.
  
- **Content Generation**:
  - Generate structured content for new nodes using OpenAI's language models.
  - Align generated content with existing graph themes and metadata.

- **Graph Expansion**:
  - Dynamically insert new nodes and edges into NebulaGraph.
  - Use graph embeddings and cosine similarity to identify relevant neighbors.

- **Graph Learning with PyTorch Geometric**:
  - Train a VGAE model with a custom encoder using **GATConv** layers.
  - Encode nodes into a latent embedding space for downstream analysis.

## Repository Structure

### 1. **Data Preparation** (`graph_data_prep.py`)
- Connects to NebulaGraph to query document nodes and edges.
- Embeds document features using **BERT** for textual data.
- Prepares `Data` objects compatible with PyTorch Geometric.

### 2. **VGAE Model** (`vgae_model.py`)
- Implements a VGAE with a custom GATConv-based encoder.
- Trains the model to encode the graph structure and node features.
- Provides methods for cluster density analysis and node position suggestions.

### 3. **Content Generator** (`doc_generator.py`)
- Uses OpenAI's GPT models to create structured document content.
- Incorporates metadata and neighbor nodes to generate contextually relevant content.

### 4. **Main Orchestration** (`main.py`)
- Executes the pipeline:
  1. Prepares graph data.
  2. Trains the VGAE model.
  3. Analyzes cluster density.
  4. Suggests new node positions.
  5. Generates and integrates new content.

### 5. **Data Files** (`data/`)
- Contains data for the nodes and edges of the example input knowledge graph.

## Workflow

### **Generate New Nodes**
Use the main script to identify knowledge gaps, suggest new nodes, and generate content:
```bash
python main.py
```

### Example Pipeline
1. Query and preprocess document nodes and edges from NebulaGraph.
2. Train a VGAE to encode graph structure and identify sparse regions.
3. Suggest positions for new nodes using cluster analysis.
4. Generate structured content for these nodes.
5. Insert the new nodes and edges into NebulaGraph.

## Requirements

- **Python 3.8+**
- Libraries:
  - `torch`
  - `torch_geometric`
  - `transformers`
  - `nebula3-python`
  - `openai`
  - `scikit-learn`

## Future Directions

- Support for multi-modal graphs with additional edge types and node categories.
- Integration with real-time graph updates for dynamic applications.
- Exploration of advanced graph neural networks for richer embeddings.
- Prompt engineering and additional LLM strategies to refine document content generation.

Built by **Kaustav Malik** for dynamic knowledge graph enrichment and expansion.
