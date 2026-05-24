import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load biến môi trường chứa DEEPSEEK_API_KEY
load_dotenv()

# Đường dẫn gốc tới thư mục chứa Vector DB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "chroma_db")


class MedicalRAGAssistant:
    def __init__(self, modality_name="xray"):
        """
        Khởi tạo Trợ lý AI dựa trên Modality (xray hoặc mri)
        """
        self.modality = modality_name
        persist_dir = os.path.join(VECTOR_DB_DIR, modality_name)

        # 1. Khởi tạo lại mô hình nhúng (Cùng mô hình lúc nạp dữ liệu ở Phase 1)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. Kết nối lại ChromaDB đã lưu
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"⚠️ Không tìm thấy VectorDB cho {modality_name}. Hãy chạy ingest.py trước.")

        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

        # Khởi tạo bộ máy tìm kiếm (Lấy 3 đoạn văn bản liên quan nhất)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})

        # 3. Khởi tạo LLM (Sử dụng DeepSeek chuẩn mới)
        self.llm = ChatOpenAI(
            model="deepseek-v4-pro",  # Sử dụng model cao cấp của DeepSeek
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",  # URL chuẩn mới
            max_tokens=1000,
            temperature=0.2  # Temperature thấp = Tính chính xác cao, hạn chế bịa đặt (hallucination)
        )

        # 4. Prompt Engineering - Dùng ChatPromptTemplate tối ưu cho mô hình Chat
        system_prompt = (
            "Bạn là một Trợ lý Bác sĩ Chẩn đoán Hình ảnh cấp cao thuộc nền tảng MedVision AI." 
            "Nhiệm vụ của bạn là phân tích các câu hỏi lâm sàng và đưa ra các khuyến nghị chuyên môn dựa trên bằng chứng.\n\n "
    
            "HƯỚNG DẪN TỐI QUAN TRỌNG (CRITICAL INSTRUCTIONS):\n"
            "1. LUÔN LUÔN trả lời bằng Tiếng Việt chuyên ngành Y khoa, sử dụng đúng thuật ngữ lâm sàng.\n"
            "2. Dựa TUYỆT ĐỐI vào bối cảnh (context) tài liệu y khoa được cung cấp bên dưới.\n"
            "3. Nếu bối cảnh không chứa thông tin để trả lời, HÃY NÓI: 'Tôi không tìm thấy thông tin phù hợp trong phác đồ y khoa hiện tại.' TUYỆT ĐỐI KHÔNG tự bịa đặt hay suy diễn kiến thức nằm ngoài tài liệu.\n"
            "4. Trình bày câu trả lời rõ ràng, rành mạch bằng các gạch đầu dòng (bullet points).\n\n"
            "BỐI CẢNH LÂM SÀNG ĐƯỢC TRÍCH XUẤT (CONTEXT):\n"
            "{context}"
        )

        # Tách bạch vai trò System và Human
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 5. Xây dựng Chuỗi RAG (RAG Chain)
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, document_chain)

    def ask(self, question: str) -> str:
        """
        Gửi câu hỏi và nhận câu trả lời
        """
        print(f"\n🔍 [MedVision RAG] Đang tìm kiếm tài liệu cho câu hỏi: '{question}'...")

        try:
            # Chạy toàn bộ luồng RAG
            response = self.rag_chain.invoke({"input": question})
            return response["answer"]
        except Exception as e:
            return f"❌ System Error: {str(e)}"


# === TEST PHASE 2 ===
if __name__ == "__main__":
    print("🏥 MEDVISION AI - CLINICAL ASSISTANT TEST TERMINAL")
    print("-------------------------------------------------")

    try:
        # Giả lập Bác sĩ đang xem một ca X-quang
        xray_assistant = MedicalRAGAssistant(modality_name="xray")

        while True:
            user_question = input("\n👨‍⚕️ Bác sĩ (Gõ 'exit' để thoát): ")
            if user_question.lower() == 'exit':
                break

            print("🤖 AI Assistant đang suy nghĩ...\n")
            answer = xray_assistant.ask(user_question)

            print("\n================== ĐÁP ÁN TỪ HỆ THỐNG ==================")
            print(answer)
            print("========================================================\n")

    except FileNotFoundError as e:
        print(e)