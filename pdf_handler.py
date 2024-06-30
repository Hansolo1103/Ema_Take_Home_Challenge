from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llm_chain import load_vectordb, create_embeddings
from unstructured.partition.pdf import partition_pdf
import os


def get_pdf_elements(pdf_bytes, pdf_name):
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_pdf_path = os.path.join(temp_dir, pdf_name)
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    raw_pdf_elements = partition_pdf(
        filename=temp_pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir="extracted_data",
        extract_image_block_to_payload=False
    )

    os.remove(temp_pdf_path)

    return raw_pdf_elements


def categorize_pdf_elements(elements):
    headers = []
    footers = []
    titles = []
    narrative_texts = []
    texts = []
    list_items = []
    tables = []

    for element in elements:
        element_type = str(type(element))
        if "unstructured.documents.elements.Title" in element_type:
            titles.append(str(element))
        elif "unstructured.documents.elements.Header" in element_type:
            headers.append(str(element))
        elif "unstructured.documents.elements.Footer" in element_type:
            footers.append(str(element))
        elif "unstructured.documents.elements.ListItem" in element_type:
            list_items.append(str(element))
        elif "unstructured.documents.elements.NarrativeText" in element_type:
            narrative_texts.append(str(element))
        elif "unstructured.documents.elements.Text" in element_type:
            texts.append(str(element))
        elif "unstructured.documents.elements.Table" in element_type:
            tables.append(str(element))

    return {
        "headers": headers,
        "footers": footers,
        "titles": titles,
        "narrative_texts": narrative_texts,
        "texts": texts,
        "list_items": list_items,
        "tables": tables
    }


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=50,
        separators=["\n", "\n\n"]
    )
    return splitter.split_text(text)


def get_document_chunks(text_list):
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content=chunk))
    return documents


def add_docs_to_db(pdfs):
    texts = []
    headers = []
    footers = []
    tables = []

    for pdf in pdfs:
        pdf_elements = get_pdf_elements(pdf.read(), pdf.name)
        categorized_elements = categorize_pdf_elements(pdf_elements)

        texts.append("\n".join(categorized_elements["texts"]))
        headers.append("\n".join(categorized_elements["headers"]))
        footers.append("\n".join(categorized_elements["footers"]))
        tables.append("\n".join(categorized_elements["tables"]))

    text_documents = get_document_chunks(texts)
    header_documents = get_document_chunks(headers)
    footer_documents = get_document_chunks(footers)
    table_documents = get_document_chunks(tables)

    vector_db = load_vectordb(create_embeddings())

    if text_documents:
        vector_db.add_documents(text_documents, collection_name="texts")
    if header_documents:
        vector_db.add_documents(header_documents, collection_name="headers")
    if footer_documents:
        vector_db.add_documents(footer_documents, collection_name="footers")
    if table_documents:
        vector_db.add_documents(table_documents, collection_name="tables")
