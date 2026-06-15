# German Medical Linguistic Analysis - Final Working Version
import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("German Medical Linguistic Analysis")
print("="*50)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.7)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
df = pd.read_csv('german_doctor_reviews_stratified_sample.csv')
df['comment'] = df['comment'].astype(str)
df = df[df['comment'].str.len() > 30].reset_index(drop=True)

max_reviews = 8000
if len(df) > max_reviews:
    df = df.sample(n=max_reviews, random_state=42).reset_index(drop=True)

print(f"Working with {len(df)} reviews")

# Sentence segmentation
def segment_german_sentences(text, min_tokens=5, max_tokens=50):
    sentences = []
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'([.!?])([A-ZÄÖÜ])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    sentences_raw = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])', text)
    
    for sent in sentences_raw:
        sent = sent.strip()
        token_count = len(sent.split())
        if (min_tokens <= token_count <= max_tokens and
            re.search(r'[a-zA-ZäöüÄÖÜß]', sent)):
            sentences.append(sent)
    
    return sentences

# Process sentences
print("Segmenting sentences...")
sentence_data = []
for doc_id, row in tqdm(df.iterrows(), total=len(df)):
    sentences = segment_german_sentences(row['comment'])
    for sent_id, sentence in enumerate(sentences):
        sentence_data.append({
            'doc_id': doc_id,
            'sent_id': sent_id,
            'sentence': sentence,
            'token_count': len(sentence.split()),
            'char_count': len(sentence),
            'rating': row.get('rating', None)
        })

sentences_df = pd.DataFrame(sentence_data)
print(f"Generated {len(sentences_df):,} sentences")

# Sample for memory optimization
max_sentences = 12000
if len(sentences_df) > max_sentences:
    sentences_df = sentences_df.sample(n=max_sentences, random_state=42).reset_index(drop=True)

print(f"Final sample: {len(sentences_df)} sentences")

# Try different German models in order of preference
model_used = None
embeddings = None

# Option 1: GottBERT with slow tokenizer
try:
    print("Trying GottBERT with slow tokenizer...")
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained('uklfr/gottbert-base', use_fast=False)
    model = AutoModel.from_pretrained('uklfr/gottbert-base').to(device)
    model.eval()
    
    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return summed / counts
    
    all_embeddings = []
    batch_size = 16
    sentences_list = sentences_df['sentence'].tolist()
    
    for i in tqdm(range(0, len(sentences_list), batch_size), desc="GottBERT encoding"):
        batch = sentences_list[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
        
        all_embeddings.append(batch_embeddings.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    model_used = "GottBERT"
    print("SUCCESS: GottBERT with slow tokenizer")
    
except Exception as e:
    print(f"GottBERT failed: {str(e)[:100]}")

# Option 2: German BERT base if GottBERT fails
if embeddings is None:
    try:
        print("Trying German BERT base...")
        from sentence_transformers import SentenceTransformer
        
        sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = sentence_model.encode(
            sentences_df['sentence'].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        model_used = "Multilingual_MiniLM"
        print("SUCCESS: Multilingual MiniLM")
        
    except Exception as e:
        print(f"Multilingual model failed: {e}")

if embeddings is not None:
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Model used: {model_used}")
    
    # Save results
    sentences_df.to_csv('data/sentences_data.csv', index=False)
    np.save('data/german_embeddings.npy', embeddings)
    
    # Save analysis info
    with open('data/embedding_info.txt', 'w', encoding='utf-8') as f:
        f.write(f"German Medical Reviews Analysis\n")
        f.write(f"==============================\n")
        f.write(f"Model used: {model_used}\n")
        f.write(f"Total reviews: {len(df)}\n")
        f.write(f"Total sentences: {len(sentences_df)}\n")
        f.write(f"Embedding dimensions: {embeddings.shape[1]}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Average sentence length: {sentences_df['token_count'].mean():.1f} tokens\n")
        f.write(f"Rating distribution:\n")
        for rating, count in sentences_df['rating'].value_counts().sort_index().items():
            f.write(f"  Rating {rating}: {count} sentences\n")
    
    print("\nEmbeddings completed successfully!")
    print("Saved: data/sentences_data.csv")
    print("Saved: data/german_embeddings.npy")
    print("Saved: data/embedding_info.txt")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")
        
    print("\nReady for topic modeling...")
    
else:
    print("ERROR: All embedding models failed!")
    print("Please check your internet connection and model availability")