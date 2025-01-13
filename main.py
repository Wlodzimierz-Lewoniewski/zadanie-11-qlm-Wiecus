import math
from collections import Counter

def preprocess_text(text):
    """Normalizes text to lowercase and tokenizes it by spaces."""
    return text.lower().replace('.', '').split()

def calculate_smoothed_probability(word, doc_freq, doc_size, corpus_freq, corpus_size, alpha):
    prob_in_doc = doc_freq[word] / doc_size if doc_size > 0 else 0
    prob_in_corpus = corpus_freq[word] / corpus_size if corpus_size > 0 else 0
    return alpha * prob_in_doc + (1 - alpha) * prob_in_corpus

def compute_log_score(doc_tokens, query_tokens, corpus_freq, corpus_size, alpha):
    doc_freq = Counter(doc_tokens)
    doc_size = len(doc_tokens)
    
    log_score = sum(
        math.log(calculate_smoothed_probability(word, doc_freq, doc_size, corpus_freq, corpus_size, alpha))
        for word in query_tokens
        if calculate_smoothed_probability(word, doc_freq, doc_size, corpus_freq, corpus_size, alpha) > 0
    )
    return log_score

def rank_documents(docs, query, alpha=0.5):
    query_tokens = preprocess_text(query)
    tokenized_docs = list(map(preprocess_text, docs))
    
    corpus_tokens = [token for doc in tokenized_docs for token in doc]
    corpus_freq = Counter(corpus_tokens)
    corpus_size = len(corpus_tokens)
    
    scored_docs = [
        (idx, compute_log_score(doc, query_tokens, corpus_freq, corpus_size, alpha))
        for idx, doc in enumerate(tokenized_docs)
    ]
    
    ranked_docs = sorted(scored_docs, key=lambda item: (-item[1], item[0]))
    return [idx for idx, _ in ranked_docs]

if __name__ == "__main__":
    doc_count = int(input("Enter number of documents: "))
    documents = [input("Document {}: ".format(i + 1)).strip() for i in range(doc_count)]
    search_term = input("Enter search query: ").strip()
    
    result_indices = rank_documents(documents, search_term)
    print(result_indices)
