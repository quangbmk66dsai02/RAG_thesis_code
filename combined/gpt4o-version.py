
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils import *


def main():
    # Argument parsing for flexibility
    parser = argparse.ArgumentParser(description="FAISS Search with BGEM3")
    parser.add_argument('--query', type=str, default="Ai là người đã cho đúc những đồng tiền thưởng đầu tiên trong triều Nguyễn?", help='Search query string')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--faiss_index', type=str, default='bgem3-components/database_building/chunked_faiss_index.bin', help='Path to FAISS index file')
    parser.add_argument('--mapping_csv', type=str, default='bgem3-components/database_building/chunked_text_data.csv', help='Path to mapping CSV file')
    args = parser.parse_args()

    article_links_path = 'bgem3-components/database_building/article_links.csv'  # Path to the CSV file containing article links
    df = pd.read_csv(article_links_path)
    file_to_link = dict(zip(df['Filename'], df['URL']))


    # Load the FAISS index and mapping
    print("Loading FAISS index and mapping DataFrame...")
    index, mapping_df = load_faiss_index_and_mapping(args.faiss_index, args.mapping_csv)
    fine_tuned_model = SentenceTransformer('bgem3-components/fine-tuned-sentence-transformer')
    file_path = "bgem3-components/database_building/article_links.csv"
    df_links = pd.read_csv(file_path)
    file_to_link = dict(zip(df_links['Filename'], df_links['URL']))

    model = "gpt-4o"
    while True:
        # Processing retrieved results
        input_query = input("Enter your query (or type 'exit' to quit): ")
        if input_query.lower() == 'exit':
            break
        # Perform the search
        search_results = search_faiss(input_query, index, mapping_df, fine_tuned_model, k=10)
        reranked_results = search_results

        # Add metadata to reranked results
        for res in reranked_results:
            text_chunk = res['answer_chunk']
            for _, data_rec in mapping_df.iterrows():
                if text_chunk == data_rec['text_chunk']:
                    res['original_text'] = data_rec['file_name']

        answer_content = ""
        for id, res in enumerate(reranked_results):
            filename = res['original_text']
            original_link = file_to_link.get(filename, "Link not found")

            entry = f"{id}. Content: {res['answer_chunk']}\n   Location: {filename}\n   Link: {original_link}\n\n"
            answer_content += entry
        print(answer_content)

        
        # Processing the generation part
        response = gpt_response(model, input_query, answer_content)
        print(f"Generated Response:\n{response}\n")
if __name__ == "__main__":
    main()
