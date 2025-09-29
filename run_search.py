import json
import os
import numpy
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pyserini.search.lucene import LuceneSearcher

# --- IMPORTANT: SET YOUR JAVA_HOME PATH HERE ---
# Pyserini requires Java 11 or later.
# macOS example: '/Users/your_username/Library/Java/JavaVirtualMachines/openjdk-21/Contents/Home'
# Linux example: '/usr/lib/jvm/java-21-openjdk-amd64'
# Windows example: 'C:\\Program Files\\Java\\jdk-21'
# --------------------------------------------------
os.environ['JAVA_HOME'] = '/opt/homebrew/Cellar/openjdk@21/21.0.8/libexec/openjdk.jdk/Contents/Home'


class DocumentRetriever:
    """
    A class to retrieve and re-rank documents using a hybrid BM25 + BERT approach.
    """
    def __init__(self, index_path, tokenizer, model):
        """
        Initializes the retriever with the Lucene index, tokenizer, and model.
        """
        print("Initializing LuceneSearcher...")
        self.searcher = LuceneSearcher(index_path)
        # Innovation 1: Enable RM3 relevance feedback for query expansion
        self.searcher.set_rm3(fb_terms=5, fb_docs=10, original_query_weight=0.7)
        self.tokenizer = tokenizer
        self.model = model
        print("DocumentRetriever initialized successfully.")

    def _extract_bert_embeddings(self, text):
        """
        Extracts BERT embeddings for a given piece of text.
        """
        text_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)])
        
        # Split into chunks if text is too long for the model
        n_chunks = int(numpy.ceil(float(text_ids.size(1)) / 510))
        states = []
        
        for i in range(n_chunks):
            start = i * 510
            end = (i + 1) * 510
            
            # Construct chunk with [CLS] and [SEP] tokens
            chunk_ids = torch.cat([
                text_ids[0, 0].unsqueeze(0), 
                text_ids[0, start+1:end+1]
            ])
            if text_ids.size(1) > end + 1: # Check if there is a token to be a SEP
                 chunk_ids = torch.cat([chunk_ids, text_ids[0, -1].unsqueeze(0)])
            
            with torch.no_grad():
                # Get model output (last hidden state)
                state = self.model(chunk_ids.unsqueeze(0))[0]
                # Remove [CLS] and [SEP] token representations
                state = state[:, 1:-1, :]
            states.append(state)

        # Concatenate states from all chunks
        state = torch.cat(states, axis=1)
        return state[0]

    def _cross_match(self, query_state, doc_state):
        """
        Calculates the similarity matrix between query and document embeddings.
        """
        # Normalize embeddings to unit vectors
        query_state = query_state / torch.sqrt((query_state ** 2).sum(1, keepdims=True))
        doc_state = doc_state / torch.sqrt((doc_state ** 2).sum(1, keepdims=True))
        
        # Calculate cosine similarity
        sim_matrix = (query_state.unsqueeze(1) * doc_state.unsqueeze(0)).sum(-1)
        return sim_matrix

    def search(self, query, k=10):
        """
        Performs a search for a given query.
        1. Retrieves top-k documents using Pyserini with RM3.
        2. Re-ranks the documents using ClinicalBERT embeddings.
        3. Prints the top 5 final results.
        """
        print(f"\nPerforming initial search for query: '{query}' with k={k}...")
        hits = self.searcher.search(query, k)
        
        if not hits:
            print("No documents found by the initial search.")
            return

        print(f"Found {len(hits)} initial documents. Re-ranking with ClinicalBERT...")
        
        # Step 1: Get embeddings for the query (once)
        query_state = self._extract_bert_embeddings(query)
        
        doc_contents = []
        paragraph_states = []
        
        # Step 2: Get embeddings for all retrieved documents
        for hit in hits:
            doc_json = json.loads(hit.lucene_document.get('raw'))
            # FIX: The 'contents' field is a string, not a dictionary.
            # Access it directly to fix the TypeError.
            content = doc_json['contents']
            doc_contents.append(content)
            
            doc_state = self._extract_bert_embeddings(content)
            paragraph_states.append(doc_state)

        # Step 3: Calculate similarity scores for all documents
        relevance_scores = []
        for doc_state in paragraph_states:
            sim_matrix = self._cross_match(query_state, doc_state)
            # The score for a document is the maximum similarity value found
            relevance_scores.append(torch.max(sim_matrix).item())

        # Step 4: Combine scores with documents and sort
        ranked_results = sorted(zip(relevance_scores, doc_contents), key=lambda x: x[0], reverse=True)

        # Step 5: Print the top 5 results
        print(f"\n--- Top 5 re-ranked results for query: '{query}' ---")
        for i, (score, content) in enumerate(ranked_results[:5]):
            print(f"\n--- Result {i+1} (Relevance Score: {score:.4f}) ---")
            print(content)
        print("\n" + "="*50)


def main():
    """
    Main function to run the medical search engine.
    """
    # Load the pre-trained ClinicalBERT model and tokenizer
    print("Loading ClinicalBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model = AutoModelForMaskedLM.from_pretrained("medicalai/ClinicalBERT")
    print("Model and tokenizer loaded.")

    # Initialize the document retriever
    index_path = 'indexes/sample_collection_jsonl'
    retriever = DocumentRetriever(index_path, tokenizer, model)
    
    # Define a query and run the search
    query = "What is the treatment for a headache with nausea?"
    retriever.search(query, k=10)


if __name__ == '__main__':
    main()

