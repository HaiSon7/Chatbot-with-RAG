import streamlit as st
from pathlib import Path
from typing import List, Dict
from rag.rag import RAG
from reflection.reflection import Reflection
from semantic_router.router import SemanticRouter
from embeddings.embedding import Embedding
from langchain_ollama.chat_models import ChatOllama


# Constants
class Config:
    PAGE_TITLE = "Hoàng Hà Mobile - Tư vấn điện thoại"
    PAGE_ICON = "📱"
    MODEL_NAME = "gemma2:2b"
    MODEL_TEMP = 0.5
    EMBEDDING_MODEL = 'keepitreal/vietnamese-sbert'
    DB_NAME = "products"
    DB_COLLECTION = "products"

    SYSTEM_MESSAGE = {
        "role": "system",
        "content": """Bạn tên là Hoàng Hà, bạn là nhân viên bán hàng điện thoại của 
        Hoàng Hà Mobile. Trả lời dựa trên thông tin được cung cấp. 
        Không trả lời những vấn đề khác"""
    }


class UIComponents:
    @staticmethod
    def load_css():
        css = """
        .main { padding: 0rem 1rem; }
        .stTextInput { padding: 0.5rem 0; }
        .stTextInput>div>div>input {
            padding: 0.5rem 1rem;
            border-radius: 20px;
        }
        .stChatMessage {
            background-color: #f0f2f6;
            border-radius: 15px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .custom-header {
            background-color: #1d3557;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem;
        }
        """
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    @staticmethod
    def render_sidebar():
        with st.sidebar:
            st.image("images/vn-11134233-7r98o-m09xp0xm5b716b.jpg", use_column_width=True)
            st.title("Thông tin")
            st.write("""
            👋 Chào mừng bạn đến với hệ thống tư vấn của Hoàng Hà Mobile!

            💡 Bạn có thể hỏi về:
            - Thông tin sản phẩm
            - So sánh các model
            - Tư vấn chọn điện thoại
            - Thông số kỹ thuật
            """)

            with st.expander("📞 Liên hệ"):
                st.write("""
                - Hotline: 1900.xxxx
                - Email: support@hoanghamobile.com
                - Website: https://hoanghamobile.com
                """)

            if st.button("🗑️ Xóa lịch sử chat"):
                st.session_state.messages = [Config.SYSTEM_MESSAGE]
                st.experimental_rerun()

            st.markdown("""
            <div style='text-align: center; padding: 1rem; color: #666;'>
                © 2024 Hoàng Hà Mobile. All rights reserved.
            </div>
            """, unsafe_allow_html=True)


class ChatBot:
    def __init__(self):
        self.model = ChatOllama(
            model=Config.MODEL_NAME,
            temperature=Config.MODEL_TEMP,
            stream=True
        )
        self.embedding = Embedding(Config.EMBEDDING_MODEL)
        self.rag = RAG(
            self.embedding,
            st.secrets['mongo_uri'],
            Config.DB_NAME,
            Config.DB_COLLECTION
        )
        self.semantic_router = SemanticRouter(self.embedding)
        self.reflection = Reflection(self.model)

    def format_chat_history(self, messages: List[Dict]) -> List[Dict]:
        """Convert Streamlit message format to reflection format"""
        return [
            {msg['role']: msg['content']}
            for msg in messages
            if msg['role'] != 'system'
        ]

    def process_query(self, query: str, history: List[Dict]) -> str:
        try:
            enhanced_query = self.reflection(history, query)
            cls = self.semantic_router.guide(enhanced_query)

            if cls == 0:  # Product query
                enhanced_info = self.rag.enhance_prompt(enhanced_query, 2)
                prompt = f"""
                Context: {enhanced_query}
                Thông tin sản phẩm: {enhanced_info}
                Yêu cầu: Hãy trả lời câu hỏi của người dùng dựa trên context và 
                thông tin sản phẩm được cung cấp. Nếu không có sản phẩm cụ thể được yêu cầu, 
                hãy gợi ý các sản phẩm phù hợp dựa trên nhu cầu được thể hiện trong context.
                """
                print(prompt)
            else:  # Chitchat
                prompt = f"""
                Context: {enhanced_query}
                Yêu cầu: Hãy trả lời một cách thân thiện và tự nhiên, 
                phù hợp với vai trò nhân viên tư vấn.
                """

            return prompt.replace('<br>', '\n')
        except Exception as e:
            print(f"Error in query processing: {e}")
            return query

    def stream_response(self, messages: List[Dict]) -> str:
        try:
            full_response = ""
            holder = st.empty()

            for response in self.model.stream(messages):
                full_response += (response.content or '')
                holder.markdown(full_response + " ▌")
            holder.markdown(full_response)

            return full_response
        except Exception as e:
            st.error("😔 Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.")
            st.exception(e)
            return ""


def main():
    # Page config
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="wide"
    )

    # Load CSS
    UIComponents.load_css()

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [Config.SYSTEM_MESSAGE]

    # Initialize chatbot
    chatbot = ChatBot()

    # Render sidebar
    UIComponents.render_sidebar()

    # Main chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        if message['role'] != 'system':
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    # Chat input
    if query := st.chat_input("💭 Hãy đặt câu hỏi về sản phẩm..."):
        # Display user message
        with st.chat_message('user'):
            st.markdown(query)
        st.session_state.messages.append({'role': 'user', 'content': query})

        # Generate and display response
        with st.chat_message('assistant'):
            chat_history = chatbot.format_chat_history(st.session_state.messages)
            processed_query = chatbot.process_query(query, chat_history)

            messages = st.session_state.messages.copy()
            messages.append({"role": "user", "content": processed_query})

            response = chatbot.stream_response(messages)
            if response:
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response
                })

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()