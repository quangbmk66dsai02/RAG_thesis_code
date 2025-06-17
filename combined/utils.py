import os
import numpy as np
import faiss
import argparse
import pandas as pd
import json
import torch
# Function to load and parse JSON file
def load_questions_and_data(json_file_path):
    """
    Loads a JSON file and structures its content by pairing each question with its corresponding data.

    Parameters:
    - json_file_path (str): The path to the JSON file.

    Returns:
    - list of tuples: A list where each tuple contains a question and its corresponding data.
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    question_data_pairs = []
    for entry in json_data:
        data_entry = entry.get('paragraph_id')
        questions = entry.get('questions', [])
        for question in questions:
            question_data_pairs.append((question, data_entry))

    return question_data_pairs


def load_faiss_index_and_mapping(faiss_index_path, mapping_csv_path):
    """
    Loads the FAISS index and the mapping DataFrame.

    Parameters:
    - faiss_index_path (str): Path to the saved FAISS index.
    - mapping_csv_path (str): Path to the saved mapping CSV.

    Returns:
    - faiss.Index: The loaded FAISS index.
    - pd.DataFrame: The loaded mapping DataFrame.
    """
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors.")

    # Load mapping DataFrame
    mapping_df = pd.read_csv(mapping_csv_path, encoding='utf-8-sig')
    print(f"Loaded mapping DataFrame with {len(mapping_df)} entries.")
    return index, mapping_df


def search_faiss(query, index, mapping_df, fine_tuned_model,k =5):
    """
    Performs a similarity search for the given query using FAISS.

    Parameters:
    - query (str): The search query string.
    - index (faiss.Index): The FAISS index.
    - mapping_df (pd.DataFrame): The mapping DataFrame.
    - k (int): Number of nearest neighbors to retrieve.

    Returns:
    - List[Dict]: A list of search results with metadata.
    """
    # Generate embedding for the query
    query_embedding = fine_tuned_model.encode(query)

    # Perform the search
    distances, indices = index.search(np.array([query_embedding], dtype='float32'), k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < len(mapping_df):
            result = {
                'rank': rank,
                'answer_chunk': mapping_df.iloc[idx]['text_chunk'],
                'original_index': mapping_df.iloc[idx]['file_name'],
                'distance': distances[0][rank-1],
                'chunk_id': idx
            }
            results.append(result)
        else:
            # Handle out-of-bounds indices
            print(f"Index {idx} is out of bounds for the mapping DataFrame.")

    return results

def format_prompt(instruction, context):
    if context.strip():
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            "### Response:"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:"
        )

def generate_response(model, tokenizer, instruction, context, device="cuda", max_length=1512):
    prompt = format_prompt(instruction, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length + prompt_len,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][prompt_len:]  # Skip the prompt
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

from openai import OpenAI
def gpt_response(model, query, answer_content):
    key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=key)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"""Bạn là một trợ lý trả lời câu hỏi chỉ dựa trên các tài liệu được cung cấp. Không sử dụng bất kỳ kiến thức hoặc thông tin bên ngoài nào không có trong dữ liệu được cung cấp. Nếu không tìm thấy câu trả lời trong tài liệu, hãy trả lời 'Không có thông tin trong cơ sở dữ liệu.' Đồng thời, bao gồm liên kết và vị trí của bằng chứng văn bản trong câu trả lời.
                Bạn có thể sử dụng nhiều nguồn để trả lời câu hỏi, nhưng hãy đảm bảo rằng mỗi nguồn đều được trích dẫn rõ ràng trong câu trả lời của bạn. Nếu không có thông tin nào liên quan đến câu hỏi, hãy trả lời 'Không tìm được thông tin liên quan'.
                Example:
                câu hỏi: Đại tướng Võ Nguyên Giáp sinh ra ở đâu?
                nội dung cung cấp: 0. Content Võ Nguyên Giáp sinh ra ở Quảng Bình. Location: abc.txt, Link: xyz
                câu trả lời: Đại tướng Võ Nguyên Giáp sinh ra ở Quảng Bình. Location: abc.txt, Link: xyz
                câu hỏi: Vị vua nào được dân gian mô tả với "gương mặt sắt đen sì"?
                nội dung cung cấp: 
                0. Mai Hắc Đế là vị vua anh minh. Location abc.txt, Link xyz, 
                1. Nhà Trần được thành lập năm ... Location abc.txt, Link xyz.
                câu trả lời: Không tìm được thông tin liên quan """},
            {"role": "user", "content": f"""câu hỏi: {query}. 
                nội dung cung cấp: {answer_content}
                câu trả lời:"""},
        ]
    )
    response_content = completion.choices[0].message.content
    return response_content.strip()

from openai import OpenAI
system_prompt = """Bạn là 1 trợ lý hữu ích, khi nhận được 1 câu hỏi hay phân tách nó ra thành các câu hỏi thành phần dạng fact-based.
Example:
Câu phức tạp: 
Các cuộc khởi nghĩa nông dân dưới triều đại phong kiến từ thế kỉ 10 đến thế kỉ 19?
Câu hỏi thành phần:
Đặc điểm của các cuộc khởi nghĩa nông dân trong thế kỉ 10 - 14?
Các cuộc nổi dậy của nông dân diễn ra như thế nào trong thế kỉ 15 - 16?
Những cuộc khởi nghĩa nào tiêu biểu trong thế kỉ 17 - 18?
Thế kỉ 19 chứng kiến những cuộc khởi nghĩa nông dân nào dưới triều Nguyễn?

Câu hỏi phức tạp: So sánh các hình thức nghệ thuật trong văn hóa Đông Sơn và văn hóa Sa Huỳnh.
Câu hỏi thành phần:
Hình thức nghệ thuật nào đặc trưng cho văn hóa Đông Sơn?
Các hình thức nghệ thuật nào tiêu biểu trong văn hóa Sa Huỳnh?

Câu hỏi phức tạp: Đại tướng Võ Nguyên Giáp sinh ra ở đâu?
Câu hỏi thành phần:
Đại tướng Võ Nguyên Giáp sinh ra ở đâu? 

Câu hỏi phức tạp: Vì sao nhà Nguyễn thất bại trước thực dân Pháp?
Câu hỏi thành phần:
Chính sách đối ngoại của nhà Nguyễn như thế nào trước khi Pháp xâm lược?
Nội bộ triều đình Nguyễn có những yếu tố nào gây suy yếu?
Vai trò của quân sự và công nghệ trong thất bại của nhà Nguyễn là gì?"""


def gpt_decompose(model, query):
    key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Câu hỏi phức tạp: {query}. 
Câu hỏi thành phần:"""},
        ]
    )
    response_content = completion.choices[0].message.content
    return response_content.strip()