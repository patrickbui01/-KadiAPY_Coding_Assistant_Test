from langchain_huggingface import HuggingFaceEmbeddings
import torch






def get_hf_embedding_model(model_name=None):
    """Retrieve a Hugging Face embedding model using the specified model name.."""

    if model_name is None:
        # "sentence-transformers/all-mpnet-base-v2"
        model_name = "BAAI/bge-base-en-v1.5"

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return embeddings


def get_sfr_embedding_model(
    model_name="Salesforce/SFR-Embedding-Code-400M_R", device="auto"
):
    """Get jinaai embedding."""

    # device: cpu or cuda
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = model_name
    model_kwargs = {"device": device, "trust_remote_code": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )

    return embeddings