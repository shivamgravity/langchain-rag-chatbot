import json
import os
import shutil
from datetime import datetime

# Metadata file path
METADATA_FILE = "data/metadata.json"

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
    chunks
):

    metadata = load_metadata()

    metadata[filename] = {
        "pages": pages,
        "chunks": chunks,
        "uploaded_at": (
            datetime.now()
            .strftime("%Y-%m-%d %H:%M")
        )
    }

    save_metadata(metadata)

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

    # add_document(
    #     "test.pdf",
    #     10,
    #     25
    # )

    # print(get_documents())

    # delete_document("test.pdf")