import os
import uuid
import shutil
import json
from datetime import datetime
from src.db import get_session, Document

def generate_document_id():
    return uuid.uuid4().hex[:8]

def add_document(
    filename,
    pages,
    chunks,
    user_id,
    summary="",
    questions=None,
    document_id=None
):
    if document_id is None:
        document_id = generate_document_id()

    pdf_path = f"data/documents/{document_id}.pdf"
    
    session = get_session()
    try:
        new_doc = Document(
            id=document_id,
            user_id=user_id,
            filename=filename,
            pages=pages,
            chunks=chunks,
            summary=summary,
            questions=json.dumps(questions or []),
            pdf_path=pdf_path
        )
        session.add(new_doc)
        session.commit()
    finally:
        session.close()

    return document_id, pdf_path

# Fetch existing documents
def get_documents(user_id):
    session = get_session()
    try:
        docs = session.query(Document).filter_by(user_id=user_id).all()
        # Return format expected by the app (dictionary by ID)
        return {
            doc.id: {
                "display_name": doc.filename,
                "pages": doc.pages,
                "chunks": doc.chunks,
                "summary": doc.summary,
                "questions": json.loads(doc.questions),
                "uploaded_at": doc.uploaded_at.strftime("%Y-%m-%d %H:%M"),
                "pdf_path": doc.pdf_path
            } for doc in docs
        }
    finally:
        session.close()

# Delete an existing document
def delete_document(document_id, user_id):
    session = get_session()
    try:
        doc = session.query(Document).filter_by(id=document_id, user_id=user_id).first()
        if not doc:
            return

        if os.path.exists(doc.pdf_path):
            os.remove(doc.pdf_path)

        chroma_path = os.path.join(
            "data/chroma",
            doc.id
        )
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)

        session.delete(doc)
        session.commit()
    finally:
        session.close()

if __name__ == "__main__":
    pass