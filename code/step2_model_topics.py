# BERTopic Modeling with German Embeddings
import pandas as pd
import numpy as np
import torch
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

print("Starting BERTopic Modeling with German Embeddings")
print("="*60)

# Load German sentence embeddings
sentences_df = pd.read_csv('data/sentences_data.csv')
embeddings = np.load('data/german_embeddings.npy')

print(f"Loaded {len(sentences_df)} sentences")
print(f"Embeddings shape: {embeddings.shape}")

# German stopwords for topic modeling
german_stopwords = [
    'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich',
    'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als',
    'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass', 'sie', 'nach',
    'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie', 'einem', 'über',
    'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder', 'aber', 'vor', 'zur',
    'bis', 'mehr', 'durch', 'man', 'sein', 'wurde', 'sei', 'beim', 'ihre',
    'sehr', 'gut', 'ich', 'mich', 'mir', 'mein', 'meine', 'bin', 'wir', 'uns',
    'arzt', 'ärztin', 'praxis', 'patient', 'patientin', 'termin'
]

# Configure BERTopic components
vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    stop_words=german_stopwords,
    strip_accents='unicode'
)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=15,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# Create and fit BERTopic model
print("\nFitting BERTopic model...")
topic_model = BERTopic(
    language='german',
    vectorizer_model=vectorizer,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True
)

docs = sentences_df['sentence'].tolist()
topics, probabilities = topic_model.fit_transform(docs, embeddings)

# Add results to dataframe
sentences_df['topic'] = topics
sentences_df['topic_prob'] = [max(p) if p is not None else 0.0 for p in probabilities]

# Get topic info
topic_info = topic_model.get_topic_info()
valid_topics = topic_info[topic_info['Topic'] != -1]

print(f"\nBERTopic Results:")
print(f"Valid topics discovered: {len(valid_topics)}")
print(f"Sentences assigned to topics: {sum(1 for t in topics if t != -1)}")
print(f"Outlier sentences: {sum(1 for t in topics if t == -1)}")
print(f"Coverage: {(sum(1 for t in topics if t != -1) / len(topics) * 100):.1f}%")

# Display top topics
print(f"\nTop 10 Topics:")
for _, topic in valid_topics.head(10).iterrows():
    words = [word for word, _ in topic_model.get_topic(topic['Topic'])[:5]]
    print(f"Topic {topic['Topic']:2d}: {topic['Count']:4d} sentences - {', '.join(words)}")

# Save topic results
topic_info.to_csv('data/topic_info.csv', index=False)
sentences_df.to_csv('data/sentences_with_topics.csv', index=False)

# Create topic visualizations
print("\nCreating topic visualizations...")

# 1. Topic size distribution
plt.figure(figsize=(12, 6))
valid_counts = valid_topics['Count'].values
plt.hist(valid_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Topic Sizes', fontsize=14, fontweight='bold')
plt.xlabel('Number of Sentences per Topic')
plt.ylabel('Number of Topics')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/topic_size_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Coverage by rating
if 'rating' in sentences_df.columns:
    coverage_by_rating = sentences_df.groupby('rating').agg({
        'topic': lambda x: sum(1 for t in x if t != -1) / len(x) * 100
    }).round(1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(coverage_by_rating.index, coverage_by_rating['topic'], 
                   color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
    plt.title('Topic Model Coverage by Rating', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Coverage (%)')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/coverage_by_rating.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Top topics word clouds
print("Generating word clouds for top topics...")
for i, (_, topic_row) in enumerate(valid_topics.head(6).iterrows()):
    topic_id = topic_row['Topic']
    topic_words = dict(topic_model.get_topic(topic_id)[:20])
    
    if topic_words:
        plt.figure(figsize=(10, 6))
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=20
        ).generate_from_frequencies(topic_words)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_id}: {topic_row["Count"]} sentences', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'figures/wordcloud_topic_{topic_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

# 4. Interactive topic visualization (save as HTML)
try:
    fig = topic_model.visualize_topics()
    if fig:
        fig.write_html('figures/interactive_topics.html')
        print("Saved interactive topic visualization")
except:
    print("Interactive visualization not available")

# 5. Topic hierarchy (if enough topics)
try:
    if len(valid_topics) > 5:
        hierarchy_fig = topic_model.visualize_hierarchy()
        if hierarchy_fig:
            hierarchy_fig.write_html('figures/topic_hierarchy.html')
            print("Saved topic hierarchy visualization")
except:
    print("Hierarchy visualization not available")

# Save representative sentences for each topic
print("\nExtracting representative sentences...")
representatives = []
for topic_id in valid_topics['Topic'].head(20):
    topic_sentences = sentences_df[sentences_df['topic'] == topic_id].nlargest(3, 'topic_prob')
    for i, (_, sent) in enumerate(topic_sentences.iterrows()):
        representatives.append({
            'topic_id': topic_id,
            'rank': i + 1,
            'sentence': sent['sentence'],
            'probability': sent['topic_prob'],
            'rating': sent.get('rating', None),
            'doc_id': sent['doc_id']
        })

rep_df = pd.DataFrame(representatives)
rep_df.to_csv('data/topic_representatives.csv', index=False)

# Summary statistics
summary = {
    'total_sentences': len(sentences_df),
    'valid_topics': len(valid_topics),
    'coverage_percent': sum(1 for t in topics if t != -1) / len(topics) * 100,
    'avg_topic_size': valid_topics['Count'].mean(),
    'largest_topic_size': valid_topics['Count'].max(),
    'model_backend': 'GottBERT',
    'embedding_dim': embeddings.shape[1]
}

with open('data/topic_modeling_summary.txt', 'w', encoding='utf-8') as f:
    f.write("German Medical Reviews - BERTopic Analysis Summary\n")
    f.write("="*60 + "\n\n")
    for key, value in summary.items():
        if isinstance(value, float):
            f.write(f"{key}: {value:.2f}\n")
        else:
            f.write(f"{key}: {value}\n")
    
    f.write(f"\nTop 10 Topics:\n")
    f.write("-" * 30 + "\n")
    for _, topic in valid_topics.head(10).iterrows():
        words = [word for word, _ in topic_model.get_topic(topic['Topic'])[:5]]
        f.write(f"Topic {topic['Topic']:2d}: {topic['Count']:4d} sentences - {', '.join(words)}\n")

print("\nBERTopic modeling completed successfully!")
print("Saved files:")
print("  - data/topic_info.csv")
print("  - data/sentences_with_topics.csv") 
print("  - data/topic_representatives.csv")
print("  - figures/topic_size_distribution.png")
print("  - figures/coverage_by_rating.png")
print("  - figures/wordcloud_topic_*.png")
print("  - figures/interactive_topics.html")
print("  - data/topic_modeling_summary.txt")

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU memory cleared")