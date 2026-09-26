import json
import os
import sys
import uuid

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http.models import models

from app.config import settings
from app.ingestion.chuncking.splitter import chunk_text
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embeddings import embed_texts, get_embedding_dim

# Configure Logfire
logfire.configure(service_name="kube-rag-enterprise-ingestion-service")

# Define processed data directory as local storage
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant client
qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def save_processed_data_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in the local processed data directory."""

    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str) -> str | None:
    """Process -> chunk -> save locally -> embed -> index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract file extension
            ext = filename.lower().rsplit(".", 1)[-1]

            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("doc", "docx", "pptx"):
                from app.ingestion.loaders.office import parse_office

                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping Unsupported file name: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from: {filename} - skipping")
                return

            # 2. Chunk the text
            chunks = chunk_text(full_text)

            if not chunks:
                logfire.warning(f"No chunks extracted from: {filename} - skipping")
                return

            # 3. Save processed chunk metadata locally
            processed_data = {
                "filename": filename,
                "chunks": chunks,
                "source_type": source_type,
            }

            local_path = save_processed_data_locally(
                processed_data, source_type, filename
            )
            logfire.info(f"Saved processed data locally: {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing a& Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for vector, chunk in zip(embeddings, chunks)
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}")

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")


def process_directory(directory_path: str, source_type: str):
    """Process all files in a directory and return a list of processed file paths."""
    with logfire.span("Scanning Directory", path=directory_path, source=source_type):
        files = [
            f
            for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ]
        logfire.info(f"Found {len(files)} files in {directory_path}.")
        for filname in files:
            process_file(os.path.join(directory_path, filname), filname, source_type)


def run_universal_ingestion(
    base_dir: str, explicit_source_type: str | None = None, wipe: bool = False, wipe_source: bool = False
):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    --wipe: drop and recreate entire collection before ingestion.
    --wipe-source: delete only vectors from the source_type being ingested.

    Args:
        base_dir (str): The base directory to process.
        explicit_source_type (str | None): The source type to use for processing.
        wipe (bool): Whether to wipe entire Qdrant collection before ingesting.
        wipe_source (bool): Whether to wipe only this source_type's data.
    """

    with logfire.span("Universal Ingestion Started", base_directory=base_dir):
        if wipe and qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
            logfire.info(f"Wiped entire collection '{settings.QDRANT_COLLECTION}'")

        # Recreate collection - dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )

            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}'"
                f" with dimension {dim}, Cosine distance."
            )

        subdirectories = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirectories:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true"
                    if "true" in base_name
                    else "noisy"
                    if "nosiy" in base_name
                    else "general"
                )
            logfire.info(
                f"No sub-folders found - processing '{base_dir}' as {source_type}"
            )

            if wipe_source and qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                qdrant_client.delete(
                    collection_name=settings.QDRANT_COLLECTION,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source_type",
                                    match=models.MatchValue(value=source_type)
                                )
                            ]
                        )
                    ),
                )
                logfire.info(f"Wiped source_type '{source_type}' from collection")

            process_directory(base_dir, source_type)
        else:
            for subdir in subdirectories:
                source_type = (
                    "true"
                    if "true" in subdir.lower()
                    else "noisy"
                    if "nosiy" in subdir.lower()
                    else subdir
                )
                logfire.info(
                    f"Sub-folder found - processing '{subdir}' as {source_type}"
                )

                if wipe_source and qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete(
                        collection_name=settings.QDRANT_COLLECTION,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="source_type",
                                        match=models.MatchValue(value=source_type)
                                    )
                                ]
                            )
                        ),
                    )
                    logfire.info(f"Wiped source_type '{source_type}' from collection")

                process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    wipe_requested = "--wipe" in sys.argv
    wipe_source_requested = "--wipe-source" in sys.argv
    clean_args = [arg for arg in sys.argv if arg not in ("--wipe", "--wipe-source")]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: target directory '{target_dir}' does not exist")
        sys.exit(1)

    run_universal_ingestion(
        target_dir,
        explicit_source_type=explicit_type,
        wipe=wipe_requested,
        wipe_source=wipe_source_requested,
    )
    logfire.info("Ingestion job completed successfully")
