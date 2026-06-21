from src.document_manager import get_documents


def get_selected_pdf_paths(
    selected_document_ids,
    user_id
):

    documents = get_documents(user_id)

    pdf_paths = []

    for doc_id in selected_document_ids:

        filename = (
            documents[doc_id]
            ["display_name"]
        )

        pdf_paths.append(
            f"data/documents/{filename}"
        )

    return pdf_paths