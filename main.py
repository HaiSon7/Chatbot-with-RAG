import streamlit as st
from rag.rag import RAG
from reflection.reflection import Reflection
from semantic_router.router import SemanticRouter
from embeddings.embedding import Embedding
# Chat with an intelligent assistant in your terminal
from openai import OpenAI


# Point to the local server
client = OpenAI(base_url=st.secrets['base_url'], api_key=st.secrets["OPENAI_API_KEY"])
reflection = Reflection(llm=client)
mongo_uri = st.secrets['mongo_uri']
db_name = "products"
collection_name = "products"
history = [
    {"role": "system",
     "content": "Hãy trả lời như là một nhân viên bán hàng điện thoại dựa trên các mẫu được cung cấp. Trả lời chi tiết vào chiếc điện thoại mà khách hàng hỏi và trả lời KHÔNG BIẾT những vấn đề khác. "},
    {"role": "user",
     "content": "Tôi muốn mua điện thoại. Hãy tư vấn cho tôi"},
]
embedding = Embedding('keepitreal/vietnamese-sbert')
rag = RAG(embedding,mongo_uri,db_name,collection_name)
semantic_router = SemanticRouter(embedding)


def process_query(query_user):
    prompt = query_user.lower()
    cls = semantic_router.guide(query_user)
    # 0:product
    # 1:chitchat
    if cls == 0:
        prompt = reflection.__call__(st.session_state.messages)
        prompt = f'Tập trung vào câu hỏi của người dùng : {prompt}. Nếu không có điện thoại khách hàng yêu cầu hãy gợi ý một sản phẩm khác có trong mẫu. Trả lời dựa theo các thông tin sau : {rag.enhance_prompt(prompt, 2)}'
        prompt = prompt.replace('<br>', '\n')
    return prompt

if 'model' not in st.session_state:
    st.session_state['model'] = 'duyntnet/Vistral-7B-Chat-DPO-imatrix-GGUF'
# Khởi tạo trạng thái tin nhắn nếu chưa có
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
            'role': 'assistant',
            'content': 'Xin chào. Tôi có thể giúp gì cho bạn ?'
        })
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Nhập tin nhắn từ người dùng
if query_user := st.chat_input("Tôi có thể giúp gì cho bạn ? "):
    with st.chat_message('user'):
        st.markdown(query_user)
    st.session_state.messages.append({
        'role': 'user',
        'content': query_user
    })

    with st.chat_message('assistant'):
        prompt = process_query(query_user)
        # Lưu tin nhắn của người dùng
        full_res =""
        holder = st.empty()

        for respone in client.chat.completions.create(
            model = st.session_state['model'],
            messages = [
                {'role': m['role'],'content':m['content']}
                for m in st.session_state.messages
            ],
            temperature= 0.3,
            stream = True,
        ):
            full_res += (respone.choices[0].delta.content or '')
            holder.markdown(full_res + "")
            holder.markdown(full_res)

    st.session_state.messages.append({
        'role': 'assistant',
        'content': full_res
    })






