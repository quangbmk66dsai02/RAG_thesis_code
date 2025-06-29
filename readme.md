# Instructions for Running Code from "Retrieval-Augmented Generation System for Question-Answering Vietnamese History"

This guide provides instructions on how to set up the environment and run the code from the thesis "Retrieval-Augmented Generation System for Question-Answering Vietnamese History." All necessary code, data, and models are provided via Google Drive links. In any conflict case, please use the code from "All codes" link provided below. If you have trouble installing the code, please contact: quang.bm214925@sis.hust.edu.vn

---

## Google Drive Links
* **All Codes:** [https://drive.google.com/drive/folders/1JWLuyV2p-6ZI5JvBlB3vj3WjxyDX_J4f?usp=drive_link](https://drive.google.com/drive/folders/1JWLuyV2p-6ZI5JvBlB3vj3WjxyDX_J4f?usp=drive_link)
* **All Data and Models:** [https://drive.google.com/drive/folders/1OBDcXtOMnlYIwZXZIGlpHR8fg1bHPyUd?usp=drive_link](https://drive.google.com/drive/folders/1OBDcXtOMnlYIwZXZIGlpHR8fg1bHPyUd?usp=drive_link)

---

## 1. Environment Setup

It's recommended to use **Python 3.9** with **Conda** support.

1.  **Create a new Conda environment:**
    ```bash
    conda create -n finetune python=3.9 -y
    ```
2.  **Activate the environment:**
    ```bash
    conda activate finetune
    ```
3.  **Install necessary packages:**
    ```bash
    pip install transformers pandas torch huggingface datasets peft tiktoken blobfile protobuf sentencepiece sentence_transformers faiss-cpu
    ```

### API Keys

* **OpenAI API Key:** Set your OpenAI API key as an environment variable named `OPENAI_API_KEY` within the `finetune` Conda environment (e.g., `set OPENAI_API_KEY=your_key_here` on Windows or `export OPENAI_API_KEY=your_key_here` on Linux/macOS).
* **Hugging Face Token:** A Hugging Face token is required to access the Llama 3.2 3B model. Ensure you are logged in via the Hugging Face CLI or have your token configured for access.

---

## 2. BGEM3 Retriever Module

Navigate to the `chatbot_bgem3_data` folder (downloaded from the "All Codes" link).

### For Re-finetuning

1.  From the "Data and Models" link, go to the `BGEM3` subfolder and download `full_30k_training_data.json` and `independent_10k_test.json`.
2.  Place `full_30k_training_data.json` directly into the `chatbot_bgem3_data` folder.
3.  Run `train_para.py` to fine-tune the model. This typically takes around 10 hours on an A4000 GPU.

### For Testing

1.  From the "Data and Models" link, go to the `BGEM3` subfolder and download the `database_building` folder.
2.  Place the `database_building` folder directly into the `chatbot_bgem3_data` folder.
3.  Place `independent_10k_test.json` (downloaded previously) into the `experiment_test` subfolder within `chatbot_bgem3_data`.
4.  Run `testing_bgem3_para.py` located in the `experiment_test` subfolder.

---

## 3. PhoBERT Retriever Module

Navigate to the `chatbot_phobert` folder (downloaded from the "All Codes" link). The steps are similar to the BGEM3 module.

### For Re-finetuning

1.  From the "Data and Models" link, go to the `phobert` subfolder and download `full_30k_training_data.json` and `independent_10k_test.json`.
2.  Place `full_30k_training_data.json` directly into the `chatbot_phobert` folder.
3.  Run `train_phobert.py` to fine-tune the model.

### For Testing

1.  From the "Data and Models" link, go to the `phobert` subfolder and download the `database_building` folder.
2.  Place the `database_building` folder directly into the `chatbot_phobert` folder.
3.  Place `independent_10k_test.json` (downloaded previously) into the `experiment_test` subfolder within `chatbot_phobert`.
4.  Run `testing_phobert.py` located in the `experiment_test` subfolder.

---

### Alternative: Using Pre-trained Retrievers for Testing

If you prefer not to re-fine-tune the BGEM3 and PhoBERT models, you can download the pre-trained versions:

* **Fine-tuned PhoBERT:** From the "Data and Models" link, navigate to `models/phobert` and download `fine-tuned-phobert-sentence-transformer`. Place this folder in your `chatbot_phobert` directory.
* **Fine-tuned BGEM3:** From the "Data and Models" link, navigate to `models/BGEM3` and download `fine-tuned-sentence-transformer`. Place this folder in your `chatbot_bgem3_data` directory.

After placing the pre-trained models, you can proceed with running the respective testing files in their `experiment_test` subfolders. Note that you still need to to have `database_building` subfolder as mentioned in the testing section.

---

## 4. Fine-tuning and Testing Llama 3.2 3B

Navigate to the `finetune_llama` folder (downloaded from the "All Codes" link).

1.  Download the `data` and `experiment_test` folders from the "All Codes" link into your `finetune_llama` directory.

### To Fine-tune

1.  Run `llama_finetune.py`.
2.  Then, run `llama_finetune_additional.py`.
    *(Other files in these folders are for data preparation and additional tokenizer/model testing and are not strictly necessary for basic fine-tuning.)*

### To Test (No fine-tuning needed).
Note that since this folder does not contain codes to run BGE-M3, each test data consists of both the question and returned passages performed previously by BGE-M3. In other words, testing in this folder is offline, users choose an index from the list of questions with their corresponding passages to test Llama 3.2 3B only, not the whole framework.

1.  From the "Data and Models" link, navigate to `models/Llama` and download the `lora-adapter` and `lora-adapter2` folders.
2.  Place these two folders directly into your `finetune_llama` folder.
3.  Run `finetuned_test_additional.py` for model testing. 
4.  Check the `experiment_test` subfolder for comparisons with GPT-4o results. If you need to re-run to get GPT's answers, do it in `gpt-4o-test.py`, if you need compare GPT's answers and Llama 3.2's answers, please run `APItest_llama_gpt.py`, once you get the results, please run `statistic_reader.py` to read the results from json file. Note that there are already 2 json files there, one is for the result in the thesis report, the other, conducted recently with some adjustments, is for the result in the slide.
---

## 5. Combined Main RAG Framework

Navigate to the `combined` folder (downloaded from the "All Codes" link).

1.  Create a subfolder named `bgem3-components` inside the `combined` folder.
2.  Download the `database_building` and `fine-tuned-sentence-transformer` subfolders (as mentioned in Section 2 and 3) and place both into `bgem3-components`.
3.  Download `lora-adapter` and `lora-adapter2` (as mentioned in Section 4 for Llama) and place them in a subfolder named `llama-components` within the `combined` folder.

Now you can run the main RAG scripts:

* **Simple RAG Approach (BGE-M3 as Retriever and Llama 3.2 3B):** Run `main.py`.
* **Decomposition Version (BGE-M3 as Retriever, GPT-4o for Decomposition, GPT-4o for Answering):** Run `gpt_decompose.py`. This uses GPT-4o to decompose complex queries and answer them.
* **Decomposition Version (BGE-M3 as Retriever, GPT-4o for Decomposition, Llama 3.2 3B for Answering):** Run `llama_decompose.py`. This uses GPT-4o for decomposition and Llama 3.2 3B for answering complex queries.

Alternatively, UI versions with Gradio for a more friendly UI are provided in `additional_UI` subfolder in code folders, download the scripts corresponding to what you want to test. After you download all scripts, put them directly into `combined` folder, e.g. `combined\app_gpt_decompose.py`, and run them normally.
---
