class Reflection():
    SYSTEM_PROMPTS = {
        "summarize": """Với lịch sử trò chuyện và câu hỏi mới nhất của người dùng, 
            hãy tóm tắt ý định và nhu cầu chính của người dùng thành một câu hoàn chỉnh.
            Nếu là câu hỏi về sản phẩm, hãy làm rõ sản phẩm nào đang được đề cập.
            Nếu là trò chuyện thông thường, hãy nêu rõ chủ đề đang được thảo luận.""",

        "rebuild": """Dựa vào context đã tóm tắt, hãy xây dựng một câu hỏi độc lập 
            và đầy đủ bằng tiếng Việt. Câu hỏi này phải chứa đầy đủ thông tin để có thể 
            hiểu được mà không cần tham chiếu đến lịch sử trò chuyện."""
    }

    def __init__(self, llm):
        self.llm = llm

    def _get_messages(self, system_prompt, user_content):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def _invoke_llm(self, messages):
        try:
            completion = self.llm.invoke(messages)
            return completion.content
        except Exception as e:
            print(f"Error in reflection: {e}")
            return messages[1]["content"].split(":")[-1].strip()

    def summarize_context(self, history, query, lastItemsConsidered=100):
        history = history[-lastItemsConsidered:] if len(history) >= lastItemsConsidered else history
        history_text = "\n".join(f"{k}: {v}" for entry in history for k, v in entry.items())

        messages = self._get_messages(
            self.SYSTEM_PROMPTS["summarize"],
            f"Lịch sử chat:\n{history_text}\nCâu hỏi hiện tại: {query}"
        )
        return self._invoke_llm(messages)

    def __call__(self, history, query, lastItemsConsidered=100):
        summarized_context = self.summarize_context(history, query)
        messages = self._get_messages(
            self.SYSTEM_PROMPTS["rebuild"],
            f"Context: {summarized_context}\nCâu hỏi cần xây dựng lại: {query}"
        )
        return self._invoke_llm(messages)