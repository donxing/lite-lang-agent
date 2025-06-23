from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=300, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.create_documents([text])
    return [{"id": f"chunk_{i}", "text": chunk.page_content} for i, chunk in enumerate(chunks)]
