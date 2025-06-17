from openai import OpenAI
import os
def gpt_response(model, query):
    key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=key)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"""Bạn là một trợ lý trả lời câu hỏi lịch sử dựa trên kiến thức của bạn 
                 """},
            {"role": "user", "content": f"""câu hỏi: {query}. 

                câu trả lời:"""},
        ]
    )
    response_content = completion.choices[0].message.content
    return response_content.strip()

while True:
    input_query = input("Enter your query (or type 'exit' to quit): ")
    if input_query.lower() == 'exit':
        break
    response = gpt_response("gpt-4o", input_query)
    print(f"Generated Response:\n{response}\n")