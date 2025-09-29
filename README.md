# Enhancing Medical Information Retrieval using ClinicalBERT and Query Expansion

This project implements an advanced, two-stage medical search engine designed to retrieve highly relevant clinical documents. It enhances a standard retrieve-and-rerank pipeline by integrating a domain-specific language model (ClinicalBERT) and a powerful query expansion technique (RM3) to achieve a deep, contextual understanding of medical text.

This repository contains the final, runnable Python script for the system described in our paper.

## Key Innovations

This system improves upon a baseline Pyserini+SciBERT pipeline through three core innovations:

1. **Domain-Specific Re-ranking:** We replaced the general scientific BERT model with **ClinicalBERT**, a language model specifically pre-trained on millions of clinical notes. This allows for a much more nuanced understanding of medical terminology, symptoms, and diagnoses.
    
2. **Query Expansion:** We implemented **RM3 Relevance Feedback**, a technique that automatically expands the user's initial query with related terms found in top-retrieved documents. This improves the recall of the initial search phase, ensuring no relevant documents are missed.
    
3. **New Dataset Validation:** The entire pipeline was developed and validated on a collection of anonymized medical clinical notes, proving its effectiveness on real-world, specialized data.
    

## System Architecture

The system operates on a two-stage "retrieve-and-rerank" architecture:

1. **Stage 1: Document Retrieval (Pyserini + RM3)**
    
    - A user's query is first expanded using RM3 to be more comprehensive.
        
    - Pyserini's `LuceneSearcher` uses this expanded query to efficiently retrieve an initial list of the top-K potentially relevant documents from a pre-built Lucene index.
        
2. **Stage 2: Re-ranking (ClinicalBERT)**
    
    - The initial list of documents is then passed to ClinicalBERT.
        
    - The model generates dense vector embeddings for the query and for each document.
        
    - By calculating the cosine similarity between the query and document embeddings, the system re-ranks the documents based on their semantic relevance, producing a final, highly accurate list.
        

## Setup and Installation

Follow these steps to set up the project on your local machine.

### 1. Prerequisites

- **Python 3.8+**
    
- **Java Development Kit (JDK) 11+** (Pyserini is built on Lucene, which requires Java)
    

### 2. Clone the Repository

```
git clone https://github.com/darc12345/Information-Retrieval-Assignment.git
cd Information-Retrieval-Assignment
```

### 3. Set Up a Virtual Environment (Recommended)

```
# Create the environment
python3 -m venv .venv

# Activate the environment
# On macOS and Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 4. Install Dependencies

Install all the required Python libraries using pip:

```
pip install torch transformers pyserini pandas tqdm
```

## How to Run

### 1. Configure the `JAVA_HOME` Path

This is the most important configuration step. The script needs to know where your JDK is installed.

- **Open the `run_search.py` file.**
    
- **Locate the `JAVA_HOME` configuration line:**
    
    ```
    # --- IMPORTANT: SET YOUR JAVA_HOME PATH HERE ---
    os.environ['JAVA_HOME'] = 'path/to/your/jdk' 
    ```
    
- **Replace the placeholder path** with the correct path for your system. To find your path:
    
    - **macOS:** Run `/usr/libexec/java_home` in the terminal.
        
    - **Linux:** Run `readlink -f $(which java)` and navigate to the root JDK directory.
        
    - **Windows:** The path is typically in `C:\Program Files\Java\` or `C:\Program Files\OpenJDK\`.
        

### 2. Run the Search Script

Execute the script from your terminal. The query is hardcoded in the script for demonstration purposes.

```
python run_search.py
```

### 3. Expected Output

The script will first load the models, then perform the search and re-ranking. The final output will be printed to the console, showing the top 5 most relevant documents along with their relevance scores.

```
Loading ClinicalBERT model and tokenizer...
Model and tokenizer loaded.
Initializing LuceneSearcher...
DocumentRetriever initialized successfully.

Performing initial search for query: 'What is the treatment for a headache with nausea?' with k=10...
Found 10 initial documents. Re-ranking with ClinicalBERT...

--- Top 5 re-ranked results for query: 'What is the treatment for a headache with nausea?' ---

--- Result 1 (Relevance Score: 0.9825) ---
Patient Information:
Age: 32 years old  
Gender: Female  
Chief Complaint: Headache for more than 3 days  
...
```

## Project Structure

```
.
├── documents/
│   └── reformatted_data.jsonl   # The processed documents used for indexing
├── indexes/
│   └── sample_collection_jsonl/ # The pre-built Pyserini Lucene index
├── reformat_data.py             # Script to convert raw data to the required format
└── run_search.py                # The main script to run the search engine
```