import os
import numpy as np
import faiss
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils import load_questions_and_data, load_faiss_index_and_mapping, search_faiss


def main():
    # Argument parsing for flexibility
    parser = argparse.ArgumentParser(description="FAISS Search with BGEM3")
    parser.add_argument('--query', type=str, default="Ai là người đã cho đúc những đồng tiền thưởng đầu tiên trong triều Nguyễn?", help='Search query string')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--faiss_index', type=str, default='bgem3-components/database_building/chunked_faiss_index.bin', help='Path to FAISS index file')
    parser.add_argument('--mapping_csv', type=str, default='bgem3-components/database_building/chunked_text_data.csv', help='Path to mapping CSV file')
    args = parser.parse_args()

    file_path = 'independent_10k_test.json'  # Replace with your JSON file path
    question_data_pairs = load_questions_and_data(file_path)

# Accessing the structured data:
    print(len(question_data_pairs))
    # Load the FAISS index and mapping
    print("Loading FAISS index and mapping DataFrame...")
    index, mapping_df = load_faiss_index_and_mapping(args.faiss_index, args.mapping_csv)
    fine_tuned_model = SentenceTransformer('bgem3-components/fine-tuned-sentence-transformer')
    # fine_tuned_model = SentenceTransformer('BAAI/bge-m3')
    print("THIS IS INDEX", index)
    # Perform the search
    num_correct = 0
    cnt = 0
    pair_for_testing = question_data_pairs[:]
    for question, data in tqdm(pair_for_testing):
        cnt += 1
        query = question
        label = data
        k = args.k
        check = False
        search_results = search_faiss(query, index, mapping_df, fine_tuned_model, k=10)
        for result in search_results:
            if result['chunk_id'] == label:
                check = True
                break
        if check:
            num_correct += 1
        
    print("PERCENTAGE from", len(pair_for_testing), num_correct/len(pair_for_testing))
        # # Display the results
        # for res in search_results:
        #     print(f"--- Rank {res['rank']} ---")
        #     print(f"Answer Chunk: {res['answer_chunk']}")
        #     print(f"Original QA Pair Index: {res['original_index']}")
        #     print(f"Distance: {res['distance']:.4f}")
        #     print("-------------------------\n")

if __name__ == "__main__":
    main()
