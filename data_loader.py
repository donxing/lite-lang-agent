from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_text(text, chunk_size=300, overlap=50):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.create_documents([text])
        result = [{"id": f"chunk_{i}", "text": chunk.page_content} for i, chunk in enumerate(chunks)]
        logger.info(f"Split text into {len(result)} chunks with chunk_size={chunk_size}, overlap={overlap}")
        return result
    except Exception as e:
        logger.error(f"Failed to split text: {e}")
        raise