# Chatbot Tư Vấn Điện Thoại - Hoàng Hà Mobile 📱

Chatbot tư vấn thông minh sử dụng RAG (Retrieval Augmented Generation) để tư vấn sản phẩm điện thoại, tích hợp với cơ sở dữ liệu MongoDB và mô hình ngôn ngữ Gemma.

## 📸 Demo Screenshots

### Giao diện chính
![Main Interface](images/Screenshot%202024-12-08%20112639.png)
*Giao diện chính của chatbot với sidebar thông tin và khung chat*

## Trò chuyện thông thường 
![Chitchat](images/Screenshot%202024-12-08%20112639.png)
*Chatbot có thể trò chuyện tự nhiên với người dùng*

## Tư vấn sản phẩm
![Product Consultation](images/Screenshot%202024-12-08%20112155.png)
*Chatbot đang tư vấn chi tiết về một mẫu điện thoại*

## So sánh sản phẩm
![Product Comparison](images/Screenshot%202024-12-08%20112421.png)
![Product Comparison](images/Screenshot%202024-12-08%20112523.png)
*So sánh thông số kỹ thuật giữa các mẫu điện thoại*

### Tính năng tìm kiếm
![Seacrh Product Info](images/Screenshot%202024-10-29%20204524.png)

## 🌟 Tính năng chính

- Tư vấn sản phẩm điện thoại dựa trên dữ liệu thực
- Phân loại câu hỏi thông minh (sản phẩm/trò chuyện)
- Tìm kiếm vector với MongoDB
- Giao diện người dùng thân thiện với Streamlit
- Xử lý ngữ cảnh thông minh với reflection

## 🛠 Công nghệ sử dụng

- **Python 3.8+**
- **Streamlit**: UI Framework
- **MongoDB**: Vector Database
- **LangChain**: Framework xử lý LLM
- **Gemma**: Mô hình ngôn ngữ
- **Sentence Transformers**: Mô hình embedding

## 📋 Yêu cầu hệ thống

- Python 3.8 trở lên
- MongoDB 7.0+ với Atlas Vector Search
- GPU (khuyến nghị) hoặc CPU mạnh
- Tối thiểu 8GB RAM

## 🚀 Hướng dẫn cài đặt

1. Clone repository:
```bash
git clone https://github.com/HaiSon7/Chatbot-with-RAG
```

2. Tạo môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

4. Cấu hình MongoDB:
- Tạo tài khoản MongoDB Atlas
- Tạo database và collection
- Tạo vector index cho collection
- Cập nhật connection string trong `.env`

5. Cấu hình biến môi trường:
```bash
cp .env.example .env
# Cập nhật các biến trong .env
```

## 💻 Cấu trúc dự án

```
chatbot-hoanghamobile/
├── embeddings/           # Xử lý embedding
├── rag/                  # Logic RAG
├── reflection/          # Xử lý ngữ cảnh
├── semantic_router/     # Phân loại câu hỏi
├── main.py              # Entry point
├── requirements.txt     # Dependencies
└── README.md
```

## 🎯 Sử dụng

1. Khởi động ứng dụng:
```bash
streamlit run main.py
```

2. Truy cập ứng dụng tại:
```
http://localhost:8501
```

## 📝 API Reference

### RAG Class
```python
rag = RAG(embedding, mongodbUri, dbName, dbCollection)
rag.vector_search(query, limit=2)
rag.enhance_prompt(query, limit=2)
```

### SemanticRouter Class
```python
router = SemanticRouter(embedding)
router.guide(query)  # Returns: 0 (product) or 1 (chitchat)
```

### Reflection Class
```python
reflection = Reflection(llm)
reflection.summarize_context(history, query)
reflection(history, query)
```

## 🔧 Cấu hình

Các thông số có thể điều chỉnh trong `config.py`:

```python
MODEL_NAME = "gemma2:2b"
MODEL_TEMP = 0.5
EMBEDDING_MODEL = 'keepitreal/vietnamese-sbert'
DB_NAME = "products"
DB_COLLECTION = "products"
```

## 🤝 Đóng góp

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

## 👥 Tác giả

- **Nguyễn Hải Sơn** - *Initial work* - [HaiSon7](https://github.com/HaiSon7)

## 📊 Roadmap

- [ ] Thêm support nhiều ngôn ngữ
- [ ] Cải thiện độ chính xác của vector search
- [ ] Tối ưu hóa performance
- [ ] Thêm tính năng analytics
- [ ] Tích hợp với các nguồn dữ liệu khác

## ❓ FAQ

**Q: Làm sao để thêm sản phẩm mới?**
A: Thêm dữ liệu vào MongoDB collection và chạy script cập nhật embedding.

**Q: Mô hình có hoạt động offline không?**
A: Có, nhưng cần tải về mô hình Gemma và cấu hình phù hợp.

**Q: Làm sao để thay đổi mô hình ngôn ngữ?**
A: Cập nhật `MODEL_NAME` trong config và đảm bảo mô hình tương thích với LangChain.

## 🔗 Links hữu ích

- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
