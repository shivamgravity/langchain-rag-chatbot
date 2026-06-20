import json
import os
import uuid
import shutil
from datetime import datetime

# Metadata file path
METADATA_FILE = "data/metadata.json"

# Generate unique document ids for every uploaded document
def generate_document_id():

    return uuid.uuid4().hex[:8]

# Function to load document configs and details
def load_metadata():

    if not os.path.exists(METADATA_FILE):
        return {}

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)

# Function to metadata of processed file
def save_metadata(metadata):

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

# Function to add a new document
def add_document(
    filename,
    pages,
    chunks,
    summary="",
    questions=None,
    document_id=None
):

    metadata = load_metadata()

    if document_id is None:
        document_id = generate_document_id()

    pdf_path = (
        f"data/documents/{document_id}.pdf"
    )

    metadata[document_id] = {
        "display_name": filename,
        "pages": pages,
        "chunks": chunks,
        "summary": summary,
        "questions": questions or [],
        "uploaded_at": (
            datetime.now()
            .strftime("%Y-%m-%d %H:%M")
        ),
        "pdf_path": f"data/documents/{document_id}.pdf"
    }

    save_metadata(metadata)

    return document_id, pdf_path

# Fetch existing documents
def get_documents():

    return load_metadata()

# Delete an existing document
def delete_document(filename):

    metadata = load_metadata()

    pdf_path = os.path.join(
        "data/documents",
        filename
    )

    chroma_path = os.path.join(
        "data/chroma",
        os.path.splitext(filename)[0]
    )

    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    metadata.pop(
        filename,
        None
    )

    save_metadata(metadata)

if __name__ == "__main__":

    pass