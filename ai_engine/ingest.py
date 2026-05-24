import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Sử dụng HuggingFace Embeddings thay thế cho Google
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load biến môi trường từ file .env
# (DEEPSEEK_API_KEY của bạn sẽ được load tự động và nằm chờ ở đây cho luồng RAG)
load_dotenv()


# Đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "chroma_db")


def process_modality_documents(modality_name):
    """
    Xử lý tài liệu theo từng modality:
    - xray
    - mri
    """

    source_dir = os.path.join(KNOWLEDGE_DIR, modality_name)
    persist_dir = os.path.join(VECTOR_DB_DIR, modality_name)

    print(f"\n===== PROCESSING {modality_name.upper()} =====")

    # Kiểm tra thư mục tồn tại
    if not os.path.exists(source_dir):
        print(f"Folder not found: {source_dir}")
        return

    # Kiểm tra có file PDF không
    pdf_files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
    if len(pdf_files) == 0:
        print(f"No PDF files inside {source_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files")


    # =========================
    # STEP 1 — LOAD PDF
    # =========================

    loader = PyPDFDirectoryLoader(source_dir)
    raw_documents = loader.load()

    print(f"Loaded {len(raw_documents)} pages")


    # =========================
    # STEP 2 — SPLIT TEXT
    # =========================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_documents(raw_documents)

    print(f"Created {len(chunks)} chunks")


    # =========================
    # STEP 3 — EMBEDDING MODEL (Chạy Local - Miễn phí & Không Rate Limit)
    # =========================

    # all-MiniLM-L6-v2 là model cực nhẹ, băm vector tốt cho tiếng Anh và khá ổn với tiếng Việt
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # =========================
    # STEP 4 — SAVE TO CHROMADB
    # =========================

    print("Creating vector database...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"SUCCESS: Vector DB saved at {persist_dir}")


if __name__ == "__main__":
    process_modality_documents("xray")
    process_modality_documents("mri")