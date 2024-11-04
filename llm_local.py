from rag.rag import RAG
from reflection.reflection import Reflection
from semantic_router.router import SemanticRouter
from embeddings.embedding import Embedding
# Chat with an intelligent assistant in your terminal
from openai import OpenAI


# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
mongo_uri = r"mongodb+srv://sonnguyenhai7:sVyOZuzdCZPC4DDn@cluster0.okt5f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
db_name = "products"
collection_name = "products"
history = [
    {"role": "system",
     "content": "Hãy trả lời như là một nhân viên bán hàng điện thoại dựa trên các mẫu được cung cấp. Trả lời chi tiết vào chiếc điện thoại mà khách hàng hỏi và không trả lời những vấn đề khác. "},
    {"role": "user",
     "content": "Tôi muốn mua điện thoại. Hãy tư vấn cho tôi"},
]
embedding = Embedding('keepitreal/vietnamese-sbert')
rag = RAG(embedding,mongo_uri,db_name,collection_name)

semantic_router = SemanticRouter(embedding)
while True:
    completion = client.chat.completions.create(
        model="duyntnet/Vistral-7B-Chat-DPO-imatrix-GGUF",
        messages=history,
        temperature=0.3,
        stream=True,
    )

    new_message = {"role": "assistant", "content": ""}

    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content

    history.append(new_message)

    # Uncomment to see chat history
    # import json
    # gray_color = "\033[90m"
    # reset_color = "\033[0m"
    # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
    # print(json.dumps(history, indent=2))
    # print(f"\n{'-'*55}\n{reset_color}")
    print()
    reflection = Reflection(llm=client)
    query_user = input("> ")
    print(reflection.__call__(history))
    query_user = query_user.lower()
    cls = semantic_router.guide(query_user)
    prompt = query_user

    if cls == 0:
        prompt = f'Câu hỏi của người dùng : {query_user} + Nếu không có điện thoại khách hàng yêu cầu hãy gợi ý một sản phẩm khác có trong mẫu. Trả lời dựa theo các thông tin sau : {rag.enhance_prompt(query_user,2)}'
        prompt = prompt.replace('<br>', '\n')
    print(prompt)


    history.append({"role": "user", "content": prompt})