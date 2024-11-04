class Reflection():
    def __init__(self,llm):
        self.llm = llm



    def _concat_and_format_texts(self, data):
        concatenatedTexts = []
        for entry in data:
            for key in entry.keys():
                concatenatedTexts.append(f"{key}: {entry.get(key)} \n")
        return ''.join(concatenatedTexts)

    def __call__(self, history,  lastItemsConsidereds=100):
        if len(history) >= lastItemsConsidereds:
            history = history[len(history) - lastItemsConsidereds:]

        historyString = self._concat_and_format_texts(history)
        higherLevelSummariesPrompt = """Với lịch sử trò chuyện và câu hỏi mới nhất của người dùng tham chiếu theo ngữ cảnh đến lịch sử trò chuyện, 
                                        hãy xây dựng một câu hỏi độc lập bằng tiếng Việt có thể hiểu được mà không cần lịch sử trò chuyện. 
                                        KHÔNG trả lời câu hỏi, chỉ cần xây dựng lại câu hỏi nếu cần và nếu không thì trả lại nguyên văn. {historyString}
        """.format(historyString=historyString)

        completion = self.llm.chat.completions.create(
            model="duyntnet/Vistral-7B-Chat-DPO-imatrix-GGUF",
            messages=[
                {
                    "role": "user",
                    "content": higherLevelSummariesPrompt
                }
            ]
        )

        return completion.choices[0].message.content
