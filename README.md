# Question_answering

🔹 Task A — Fine-tuning Llama-3

For this task, I fine-tuned the model unsloth/llama-3-8b-bnb-4bit
 on the PersianQA dataset using the LoRA technique to reduce resource consumption. The evaluation was carried out using the standard QA metrics Exact Match (EM) and F1-Score. While the fine-tuned model was able to produce reasonable answers, the overall performance was not particularly strong, with scores of EM: 27.63 and F1: 39.87. These results suggest that additional training strategies or larger datasets may be necessary to significantly improve performance.

 🔹 Task B — RAG Implementation

In this task, I implemented a Retrieval-Augmented Generation (RAG) pipeline to answer questions from a normalized Drug PDF. I experimented with two chunking strategies:

Word-based: chunk_word_based(pages, chunk_size=200, overlap=50)

Sentence-based: chunk_sentence_based(pages, max_words_per_chunk=200, sentence_overlap=1)

For the embedding model, I used paraphrase-multilingual-MiniLM-L12-v2 from SentenceTransformers.

To retrieve relevant passages, I tested two classic retrieval algorithms:

TF-IDF (Term Frequency–Inverse Document Frequency), which scores documents based on how important a word is relative to the entire corpus.

BM25 is a probabilistic retrieval function that improves upon TF-IDF by normalizing term frequency and adjusting for document length.

The fine-tuned model was evaluated using both retrieval methods. All detailed metrics, including top_k, EM, hit@k, precision@k, recall@k, MRR, and MAP@k, are available in the file combined_retrieval_results.
The most notable results were:

TF-IDF achieved an Exact Match (EM) of 79%

BM25 achieved an Exact Match (EM) of 75%

These findings suggest that TF-IDF was slightly more effective in this setup, although both methods provided solid retrieval performance for the QA pipeline.

🔹 Task C — Semantic Similarity

For this part, I reused the saved dataset (loaded from Google Drive) and evaluated semantic similarity across multiple setups:

Baseline embedding model

Fine-tuned model

TF-IDF

BM25

For each method, I compared both word-based chunking and sentence-based chunking strategies. The evaluation was performed using Cosine Similarity and MRR, with additional metrics like EM and Hit@k. All detailed results are provided in the file semantic_similarity_mrr available in the repository.

Key findings include:

The highest Exact Match (EM) was achieved by TF-IDF with sentence-based chunking, reaching 0.7975.

Sentence-based chunking consistently outperformed word-based chunking across all methods.

The highest Hit@10 scores were obtained by TF-IDF and BM25, both at 0.9873.

These results suggest that the chunking strategy plays a critical role in retrieval quality, with sentence-level segmentation proving to be more effective overall.
🔹 Task D — Alternative Embedding Model

In this task, I experimented with another embedding model: distiluse-base-multilingual-cased-v2
. The model was fine-tuned on the project dataset and evaluated on its retrieval performance. Metrics included Cosine Similarity, Mean Reciprocal Rank (MRR), and several ranking-based evaluation scores.

The results (detailed in the provided notebook, which can be rerun to reproduce outcomes) showed:

Accuracy@1: 0.4667

Accuracy@3: 0.7333

Accuracy@5: 0.8000

Accuracy@10: 0.9000

Cosine MRR@10: 0.6131

Cosine NDCG@10: 0.6826

Cosine MAP@100: 0.6177

These results indicate that the model performed reasonably well in retrieval tasks, with strong gains at higher top-k thresholds (e.g., 90% accuracy at k=10). However, precision at lower ranks (k=1, k=3) suggests there is still room for improvement compared to other methods.


🔹 Task E — Multiple Embedding Models & Vector Search

For this task, I experimented with two embedding models:

intfloat/multilingual-e5-base

sentence-transformers/distiluse-base-multilingual-cased-v1

I reused the chunked dataset (from the Drug PDF, saved on Google Drive) and applied FAISS as the vector database for efficient similarity search.

The multilingual-e5-base model achieved the best results, with:

Hit@10 = 0.9367

In comparison, the distiluse-base-multilingual-cased-v1 model scored significantly lower, around 67% Hit@10.
All results can be reproduced by rerunning the provided notebooks.

These experiments show that the e5 model consistently provided stronger retrieval performance, especially in top-k evaluation, making it the most effective embedding model in this project.

🔹 Task F — User Interface (QA Demo)

To make the system interactive, I developed a Streamlit-based UI and deployed it using Ngrok in the Colab environment. The interface allows users to input questions and view both the retrieved top-k passages (with metadata) and the final generated answer produced via Gemini-Flash 2.5.

⚙️ Requirements

To run the UI properly, the following API keys and tokens are required:

Google API Key (for Gemini integration)

Ngrok Auth Token (for tunneling the Streamlit app)

Hugging Face Token (same as in previous notebooks)

🔑 Usage Notes

If you do not want to retrain or regenerate the results, you can simply run the Final Result (UI) cells in the notebook.

The outputs are preloaded from my Google Drive via a shareable link, so users can directly interact with the UI.

In this case, only the API keys above need to be substituted.

If you want to experiment with training and retrieval again, you can re-run the earlier notebook sections.

📸 Demo

