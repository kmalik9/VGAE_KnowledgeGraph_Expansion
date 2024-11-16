from openai import OpenAI
import json

"""
This script uses OpenAI to generate structured document content based on specified node data from the
suggested clusters. The generated content can be used for knowledge discovery to help create new document nodes.
"""

# Initialize OpenAI API key
client = OpenAI(api_key="OPENAI_API_KEY")

# Define the function schema for structured document generation
def structured_document_function():
    return {
        "name": "generate_document_content",
        "description": "Generates structured content for a new document node",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the document"
                },
                "content": {
                    "type": "string",
                    "description": "The main text content of the document"
                },
                "source": {
                    "type": "string",
                    "description": "The source or origin of the document"
                },
                "timestamp": {
                    "type": "string",
                    "description": "The creation or update timestamp of the document in ISO 8601 format"
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of relevant tags for categorizing the document"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata for the document",
                    "properties": {
                        "topic": {"type": "string"},
                        "related_topics": {"type": "array", "items": {"type": "string"}},
                        "objective": {"type": "string"}
                    }
                }
            },
            "required": ["title", "content", "source", "timestamp", "tags", "metadata"]
        }
    }

tools = [
    {
        "name": "generate_document_content",
        "description": "Generates structured content for a new document node",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the document"},
                "content": {"type": "string", "description": "The main text content of the document"},
                "source": {"type": "string", "description": "The source or origin of the document"},
                "timestamp": {"type": "string", "description": "The creation or update timestamp of the document in ISO 8601 format"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "List of relevant tags for categorizing the document"},
                "metadata": {"type": "object", "description": "Additional metadata for the document", "properties": {
                    "topic": {"type": "string"},
                    "related_topics": {"type": "array", "items": {"type": "string"}},
                    "objective": {"type": "string"}
                }}
            },
            "required": ["title", "content", "source", "timestamp", "tags", "metadata"]
        }
    }
]

def generate_structured_document_content(node_metadata):
    """
    Generates structured document content using OpenAI's function-calling capability.

    Args:
        node_metadata (dict): Metadata about the node for which the document is generated.

    Returns:
        dict: The structured content of the generated document.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a document generator that creates structured content based on provided document data."
            },
            {
                "role": "user",
                "content": (
                    f"Generate a document about {node_metadata} that is structured and contains the following information: "
                    "title, content, source, timestamp, tags, and metadata. The document should be about an insight that "
                    "aligns with the provided data, enabling new branches of thought from the existing knowledge."
                )
            }
        ],
        
        functions=tools  # Pass the tools correctly, including the "type"
    )

    # Convert the response to a dictionary
    response_dict = response.to_dict()

    # Extract the structured content from the function call response
    structured_content = response_dict["choices"][0]["message"]["function_call"]["arguments"]
    return json.loads(structured_content)