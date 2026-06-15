# Comprehensive German Linguistic Feature Analysis
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("Starting Comprehensive German Linguistic Feature Analysis")
print("="*70)

# Load sentences with topics
sentences_df = pd.read_csv('data/sentences_with_topics.csv')
print(f"Analyzing {len(sentences_df)} German sentences")

def extract_german_linguistic_features(text):
    """Extract comprehensive German linguistic features"""
    features = {}
    text_lower = text.lower()
    tokens = text.split()
    
    # Basic metrics
    features['token_count'] = len(tokens)
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(token) for token in tokens]) if tokens else 0
    
    # 1. German Modal Particles (Abtönungspartikeln)
    modal_particles = {
        'intensifying': ['doch', 'ja', 'schon', 'durchaus', 'wohl', 'aber'],
        'mitigating': ['halt', 'eben', 'eigentlich', 'mal'],
        'discourse': ['denn', 'etwa', 'überhaupt', 'bloß', 'nur']
    }
    
    for category, particles in modal_particles.items():
        count = sum(1 for token in tokens if token.lower() in particles)
        features[f'modal_{category}'] = count
    
    features['modal_particles_total'] = sum(
        features[f'modal_{cat}'] for cat in modal_particles.keys()
    )
    
    # 2. Epistemic Markers and Hedging
    epistemic_verbs = ['scheinen', 'glauben', 'meinen', 'vermuten', 'denken', 
                      'annehmen', 'schätzen', 'hoffen', 'erwarten']
    epistemic_adverbs = ['wahrscheinlich', 'möglicherweise', 'vermutlich', 
                        'eventuell', 'vielleicht', 'anscheinend']
    
    features['epistemic_verbs'] = sum(
        1 for token in tokens if any(token.lower().startswith(verb) for verb in epistemic_verbs)
    )
    features['epistemic_adverbs'] = sum(
        1 for token in tokens if token.lower() in epistemic_adverbs
    )
    features['hedging_total'] = features['epistemic_verbs'] + features['epistemic_adverbs']
    
    # 3. Evaluative Language
    positive_eval = ['gut', 'sehr gut', 'ausgezeichnet', 'toll', 'super', 
                    'wunderbar', 'perfekt', 'fantastisch', 'hervorragend',
                    'zufrieden', 'kompetent', 'freundlich', 'nett']
    negative_eval = ['schlecht', 'schrecklich', 'furchtbar', 'katastrophal', 
                    'unzufrieden', 'enttäuscht', 'mangelhaft', 'ungenügend',
                    'unfreundlich', 'inkompetent', 'arrogant']
    
    features['positive_evaluation'] = sum(
        1 for expr in positive_eval if expr in text_lower
    )
    features['negative_evaluation'] = sum(
        1 for expr in negative_eval if expr in text_lower
    )
    
    # 4. Negation Patterns
    negation_words = ['nicht', 'kein', 'keine', 'keinen', 'keinem', 'keines', 
                     'nie', 'niemals', 'nimmer', 'nirgends', 'weder']
    features['negation_count'] = sum(
        1 for token in tokens if token.lower() in negation_words
    )
    
    # 6. Connectives and Discourse Structure
    connectives = {
        'causal': ['weil', 'da', 'denn', 'deshalb', 'deswegen', 'daher', 'darum'],
        'contrast': ['aber', 'jedoch', 'trotzdem', 'dennoch', 'obwohl', 'obgleich'],
        'temporal': ['dann', 'danach', 'anschließend', 'später', 'vorher', 'zunächst'],
        'additive': ['außerdem', 'zusätzlich', 'auch', 'ebenfalls', 'ferner', 'und']
    }
    
    for conn_type, words in connectives.items():
        features[f'connective_{conn_type}'] = sum(
            1 for token in tokens if token.lower() in words
        )
    
    features['connectives_total'] = sum(
        features[f'connective_{conn_type}'] for conn_type in connectives.keys()
    )
    
    # 7. Medical Terminology
    medical_patterns = {
        'diagnosis': ['diagnose', 'krankheit', 'leiden', 'störung', 'syndrom', 'befund'],
        'treatment': ['behandlung', 'therapie', 'medikament', 'operation', 'eingriff', 'heilung'],
        'symptoms': ['schmerz', 'schmerzen', 'beschwerden', 'symptom', 'problem', 'schwierigkeiten'],
        'professionals': ['arzt', 'ärztin', 'doktor', 'professor', 'schwester', 'pfleger']
    }
    
    for category, terms in medical_patterns.items():
        count = sum(1 for term in terms if term in text_lower)
        features[f'medical_{category}'] = count
    
    # 8. Temporal Expressions
    temporal_words = ['heute', 'gestern', 'morgen', 'woche', 'monat', 'jahr', 
                     'sofort', 'gleich', 'bald', 'später', 'früher']
    features['temporal_expressions'] = sum(
        1 for token in tokens if token.lower() in temporal_words
    )
    
    # 9. Intensifiers
    intensifiers = ['sehr', 'extrem', 'besonders', 'ziemlich', 'recht', 
                   'äußerst', 'überaus', 'höchst', 'ungemein']
    features['intensifiers'] = sum(
        1 for token in tokens if token.lower() in intensifiers
    )
    
    # 10. Question markers
    features['has_question'] = 1 if '?' in text else 0
    
    return features

# Extract features for all sentences
print("Extracting German linguistic features...")
feature_data = []

for _, row in tqdm(sentences_df.iterrows(), total=len(sentences_df), desc="Features"):
    features = extract_german_linguistic_features(row['sentence'])
    features.update({
        'doc_id': row['doc_id'],
        'sent_id': row['sent_id'],
        'topic': row['topic'],
        'topic_prob': row['topic_prob'],
        'rating': row.get('rating', None)
    })
    feature_data.append(features)

features_df = pd.DataFrame(feature_data)
print(f"Extracted {len(features_df.columns) - 5} linguistic features")

# Analyze features by topic
valid_topics = features_df[features_df['topic'] != -1]
topic_analysis = valid_topics.groupby('topic').agg({
    'modal_particles_total': 'mean',
    'hedging_total': 'mean',
    'positive_evaluation': 'mean',
    'negative_evaluation': 'mean',
    'connectives_total': 'mean',
    'intensifiers': 'mean',
    'token_count': 'mean'
}).round(3)

# Save detailed results
features_df.to_csv('data/linguistic_features.csv', index=False)
topic_analysis.to_csv('data/topic_linguistic_analysis.csv')

# Create linguistic visualizations
print("Creating linguistic feature visualizations...")

# 1. Feature distribution by rating
if 'rating' in features_df.columns:
    plt.figure(figsize=(15, 10))
    
    feature_cols = ['modal_particles_total', 'hedging_total', 'positive_evaluation',
                   'negative_evaluation', 'connectives_total']
    
    for i, feature in enumerate(feature_cols):
        plt.subplot(2, 3, i+1)
        rating_means = features_df.groupby('rating')[feature].mean()
        bars = plt.bar(rating_means.index, rating_means.values, 
                      color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
        plt.title(f'{feature.replace("_", " ").title()}')
        plt.xlabel('Rating')
        plt.ylabel('Average Count')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures/linguistic_features_by_rating.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Modal particles vs evaluation
plt.figure(figsize=(10, 6))
scatter_data = features_df[features_df['topic'] != -1]
colors = ['red' if x > 0 else 'blue' for x in scatter_data['negative_evaluation'] - scatter_data['positive_evaluation']]

plt.scatter(scatter_data['modal_particles_total'], 
           scatter_data['positive_evaluation'] + scatter_data['negative_evaluation'],
           c=colors, alpha=0.6, s=20)
plt.xlabel('Modal Particles Count')
plt.ylabel('Total Evaluative Expressions')
plt.title('Modal Particles vs Evaluative Language')
plt.legend(['Negative tendency', 'Positive tendency'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/modal_particles_vs_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Topic linguistic profiles (top 10 topics)
top_topics = topic_analysis.head(10)
features_to_plot = ['modal_particles_total', 'hedging_total', 'positive_evaluation',
                   'negative_evaluation']

plt.figure(figsize=(12, 8))
x_pos = np.arange(len(top_topics))

for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 3, i+1)
    bars = plt.bar(x_pos, top_topics[feature], alpha=0.7)
    plt.title(f'{feature.replace("_", " ").title()}')
    plt.xticks(x_pos, [f'T{t}' for t in top_topics.index], rotation=45)
    plt.ylabel('Average')
    
    # Color bars by value
    values = top_topics[feature].values
    max_val = max(values) if max(values) > 0 else 1
    for j, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(values[j] / max_val))

plt.tight_layout()
plt.savefig('figures/topic_linguistic_profiles.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate comprehensive linguistic report
with open('data/linguistic_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("German Medical Reviews - Comprehensive Linguistic Analysis\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-"*20 + "\n")
    f.write(f"Total sentences analyzed: {len(features_df)}\n")
    f.write(f"Sentences with topics: {len(valid_topics)}\n")
    f.write(f"Average sentence length: {features_df['token_count'].mean():.1f} tokens\n\n")
    
    f.write("GERMAN LINGUISTIC FEATURES SUMMARY\n")
    f.write("-"*40 + "\n")
    linguistic_features = ['modal_particles_total', 'hedging_total', 'positive_evaluation',
                          'negative_evaluation', 'connectives_total']
    
    for feature in linguistic_features:
        mean_val = features_df[feature].mean()
        std_val = features_df[feature].std()
        f.write(f"{feature}: {mean_val:.3f} ± {std_val:.3f}\n")
    
    f.write("\nTOP 10 TOPICS - LINGUISTIC CHARACTERISTICS\n")
    f.write("-"*50 + "\n")
    for topic_id, row in topic_analysis.head(10).iterrows():
        f.write(f"\nTopic {topic_id}:\n")
        f.write(f"  Modal particles: {row['modal_particles_total']:.3f}\n")
        f.write(f"  Hedging: {row['hedging_total']:.3f}\n")
        f.write(f"  Positive evaluation: {row['positive_evaluation']:.3f}\n")
        f.write(f"  Negative evaluation: {row['negative_evaluation']:.3f}\n")
    
    if 'rating' in features_df.columns:
        f.write("\nLINGUISTIC FEATURES BY RATING\n")
        f.write("-"*35 + "\n")
        for rating in sorted(features_df['rating'].unique()):
            rating_data = features_df[features_df['rating'] == rating]
            f.write(f"\nRating {rating} ({len(rating_data)} sentences):\n")
            f.write(f"  Modal particles: {rating_data['modal_particles_total'].mean():.3f}\n")
            f.write(f"  Positive evaluation: {rating_data['positive_evaluation'].mean():.3f}\n")
            f.write(f"  Negative evaluation: {rating_data['negative_evaluation'].mean():.3f}\n")
            f.write(f"  Hedging: {rating_data['hedging_total'].mean():.3f}\n")

print("\nLinguistic feature analysis completed!")
print("Saved files:")
print("  - data/linguistic_features.csv")
print("  - data/topic_linguistic_analysis.csv")
print("  - figures/linguistic_features_by_rating.png")
print("  - figures/modal_particles_vs_evaluation.png")
print("  - figures/topic_linguistic_profiles.png")
print("  - data/linguistic_analysis_report.txt")