# Create German Parse Trees and Advanced Linguistic Visualizations - Fixed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Creating German Parse Trees and Advanced Linguistic Visualizations")
print("="*80)

# Install and load German spaCy model
try:
    nlp = spacy.load("de_core_news_sm")
    print("German spaCy model loaded successfully")
except OSError:
    print("Installing German spaCy model...")
    os.system("python -m spacy download de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")

# Load data
representatives = pd.read_csv('data/topic_representatives.csv')
features_df = pd.read_csv('data/linguistic_features.csv')
topic_info = pd.read_csv('data/topic_info.csv')

# Create directories
os.makedirs('figures/parse_html', exist_ok=True)

# Select sentences for parsing
top_topics = topic_info[topic_info['Topic'] != -1].head(10)
selected_sentences = []

for _, topic_row in top_topics.iterrows():
    topic_id = topic_row['Topic']
    topic_reps = representatives[representatives['topic_id'] == topic_id].head(3)
    
    for _, rep in topic_reps.iterrows():
        selected_sentences.append({
            'topic_id': topic_id,
            'rank': rep['rank'],
            'sentence': rep['sentence'],
            'topic_size': topic_row['Count']
        })

print(f"Selected {len(selected_sentences)} sentences for detailed analysis")

def analyze_german_sentence(sentence, nlp_model):
    """Analyze German sentence with linguistic features"""
    doc = nlp_model(sentence)
    
    # Basic analysis
    analysis = {
        'sentence': sentence,
        'tokens': len(doc),
        'max_depth': max([len(list(token.ancestors)) for token in doc] + [0]),
        'clauses': len([t for t in doc if t.dep_ in ['ROOT', 'ccomp', 'xcomp', 'advcl']]),
        'pos_distribution': {},
        'german_features': {
            'modal_particles': [],
            'compounds': []
        }
    }
    
    # POS distribution
    pos_counts = {}
    for token in doc:
        pos = token.pos_
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    analysis['pos_distribution'] = pos_counts
    
    # German features
    german_modal_particles = ['doch', 'wohl', 'ja', 'halt', 'mal', 'eben', 'eigentlich']
    
    for token in doc:
        if token.lemma_.lower() in german_modal_particles:
            analysis['german_features']['modal_particles'].append(token.text)
        
        if len(token.text) > 10 and token.pos_ in ['NOUN', 'ADJ']:
            analysis['german_features']['compounds'].append(token.text)
    
    return analysis, doc

# Generate parse trees
print("Generating parse trees...")
parse_results = []

for i, sent_info in enumerate(tqdm(selected_sentences, desc="Analyzing")):
    sentence = sent_info['sentence']
    topic_id = sent_info['topic_id']
    rank = sent_info['rank']
    
    analysis, doc = analyze_german_sentence(sentence, nlp)
    parse_results.append(analysis)
    
    # Create dependency visualization
    try:
        dep_svg = displacy.render(doc, style="dep", jupyter=False, options={
            'compact': True,
            'bg': '#ffffff',
            'color': '#2e3440',
            'distance': 120,
            'arrow_stroke': 2
        })
    except:
        dep_svg = "<p>Parse tree visualization unavailable</p>"
    
    # Generate HTML report
    modal_particles_text = ", ".join(analysis['german_features']['modal_particles']) if analysis['german_features']['modal_particles'] else "None detected"
    compounds_text = ", ".join(analysis['german_features']['compounds']) if analysis['german_features']['compounds'] else "None detected"
    
    pos_table_rows = ""
    for pos, count in sorted(analysis['pos_distribution'].items(), key=lambda x: x[1], reverse=True):
        pos_description = spacy.explain(pos) if spacy.explain(pos) else "N/A"
        pos_table_rows += f"<tr><td><strong>{pos}</strong></td><td>{count}</td><td>{pos_description}</td></tr>"
    
    html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="utf-8">
    <title>German Parse Tree - Topic {topic_id}, Sentence {rank}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background: #f8f9fa; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; margin-bottom: 30px; }}
        .sentence-box {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 5px solid #5e81ac; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .analysis-section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 5px solid #d08770; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .parse-tree {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 6px 12px rgba(0,0,0,0.15); margin: 25px 0; overflow-x: auto; }}
        .feature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .feature-card {{ background: #ebf2ff; padding: 15px; border-radius: 6px; }}
        .highlight {{ color: #d08770; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>German Medical Discourse - Parse Tree Analysis</h1>
        <h2>Topic {topic_id} - Representative Sentence #{rank}</h2>
        <p><strong>Topic Size:</strong> {sent_info['topic_size']} sentences | <strong>Syntactic Depth:</strong> {analysis['max_depth']} levels</p>
    </div>
    
    <div class="sentence-box">
        <h3>Analyzed Sentence:</h3>
        <em>"{sentence}"</em>
    </div>
    
    <div class="parse-tree">
        <h3>Dependency Parse Tree</h3>
        {dep_svg}
    </div>
    
    <div class="analysis-section">
        <h3>Syntactic Analysis</h3>
        <div class="feature-grid">
            <div class="feature-card">
                <strong>Token Count:</strong><br><span class="highlight">{analysis['tokens']}</span>
            </div>
            <div class="feature-card">
                <strong>Maximum Depth:</strong><br><span class="highlight">{analysis['max_depth']}</span> levels
            </div>
            <div class="feature-card">
                <strong>Clause Count:</strong><br><span class="highlight">{analysis['clauses']}</span> clauses
            </div>
            <div class="feature-card">
                <strong>Complexity:</strong><br><span class="highlight">{'High' if analysis['max_depth'] > 3 else 'Moderate'}</span>
            </div>
        </div>
    </div>
    
    <div class="analysis-section">
        <h3>German-Specific Features</h3>
        <p><strong>Modal Particles:</strong> {modal_particles_text}</p>
        <p><strong>Compound Words:</strong> {compounds_text}</p>
    </div>
    
    <div class="analysis-section">
        <h3>Part-of-Speech Distribution</h3>
        <table>
            <tr><th>POS Tag</th><th>Count</th><th>Description</th></tr>
            {pos_table_rows}
        </table>
    </div>
    
    <div class="analysis-section">
        <h3>Linguistic Research Notes</h3>
        <ul>
            <li><strong>Sentence Complexity:</strong> {'Complex structure' if analysis['max_depth'] > 3 else 'Simple structure'} with {analysis['max_depth']}-level dependencies</li>
            <li><strong>German Characteristics:</strong> {len(analysis['german_features']['modal_particles'])} modal particles detected</li>
            <li><strong>Medical Discourse:</strong> {'Specialized terminology' if len(analysis['german_features']['compounds']) > 0 else 'General vocabulary'}</li>
        </ul>
    </div>
</body>
</html>"""
    
    filename = f"figures/parse_html/topic_{topic_id:02d}_sentence_{rank}_analysis.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

# Create index
index_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="utf-8">
    <title>German Parse Tree Analysis Index</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background: #f8f9fa; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 6px 12px rgba(0,0,0,0.15); }}
        th {{ background-color: #5e81ac; color: white; padding: 15px; text-align: left; }}
        td {{ padding: 15px; border-bottom: 1px solid #eceff4; }}
        tr:hover {{ background-color: #f1f6ff; }}
        a {{ color: #5e81ac; text-decoration: none; font-weight: bold; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>German Medical Discourse - Parse Tree Analysis</h1>
        <p>Detailed linguistic analysis of {len(selected_sentences)} representative sentences</p>
    </div>
    
    <table>
        <thead>
            <tr><th>Topic</th><th>Rank</th><th>Tokens</th><th>Depth</th><th>Sentence Preview</th><th>Analysis</th></tr>
        </thead>
        <tbody>"""

for i, sent_info in enumerate(selected_sentences):
    analysis = parse_results[i]
    preview = sent_info['sentence'][:60] + "..." if len(sent_info['sentence']) > 60 else sent_info['sentence']
    filename = f"topic_{sent_info['topic_id']:02d}_sentence_{sent_info['rank']}_analysis.html"
    
    index_content += f"""
        <tr>
            <td>Topic {sent_info['topic_id']}</td>
            <td>{sent_info['rank']}</td>
            <td>{analysis['tokens']}</td>
            <td>{analysis['max_depth']}</td>
            <td>{preview}</td>
            <td><a href="{filename}">View Analysis</a></td>
        </tr>"""

index_content += """
        </tbody>
    </table>
</body>
</html>"""

with open('figures/parse_html/index.html', 'w', encoding='utf-8') as f:
    f.write(index_content)

# Save analysis data
analysis_data = pd.DataFrame(parse_results)
analysis_data.to_csv('data/parse_tree_analysis.csv', index=False)

print(f"\nParse tree analysis completed!")
print(f"Created {len(selected_sentences)} detailed parse tree analyses")
print(f"Generated comprehensive index: figures/parse_html/index.html")
print(f"Saved analysis data: data/parse_tree_analysis.csv")
print(f"Modal particles found: {sum(len(r['german_features']['modal_particles']) for r in parse_results)}")
print(f"Compound words found: {sum(len(r['german_features']['compounds']) for r in parse_results)}")