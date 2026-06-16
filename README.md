# Topic Modelling in German Medical Language — Code & Artefacts

Master's thesis · M.A. Data and Discourse · TU Darmstadt · Anass Soulala

This repository contains the code and selected artefacts used for the thesis. The workflow segments German doctor reviews into sentences, creates contextual embeddings, applies BERTopic, extracts rule-based linguistic features, and uses selected dependency parses for qualitative close reading. The aim is to analyse patient stance in language, not to measure clinical quality.

## Repository structure

| Folder | Content |
|--------|---------|
| `code/` | Numbered analysis scripts in run order |
| `data/` | Result tables used to check thesis outputs |
| `figures/` | Figures generated for the thesis |

## Code pipeline

Run the scripts from the repository root in numerical order.

| Script | Purpose | Main outputs |
|--------|---------|--------------|
| `code/step1_embed_sentences.py` | Loads the working dataset, segments reviews into sentences, and creates GottBERT embeddings | `data/sentences_data.csv`, `data/german_embeddings.npy` |
| `code/step2_model_topics.py` | Runs BERTopic with UMAP, HDBSCAN, and c-TF-IDF | topic assignments and topic tables |
| `code/step3_extract_features.py` | Counts linguistic features per sentence | feature tables and topic profiles |
| `code/step4_dependency_parses.py` | Creates dependency parses for representative sentences | parse tables and HTML files |
| `code/step5_make_chart_figures.py` | Generates the main quantitative figures | Figures 4-1 to 4-4 |
| `code/step6_make_parse_tree_figures.py` | Generates simplified dependency-parse figures | qualitative parse figures |

## Reproducing the results

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
```

Then place the working dataset file in the repository root as:

```text
german_doctor_reviews_stratified_sample.csv
```

and run the scripts in numerical order.

The included result tables and figures allow the thesis outputs to be checked without re-running the whole pipeline. The submitted thesis reports 12,000 analysed sentences, 42 valid non-outlier topics, and 35.35% topic coverage.

## Dataset note

The source dataset used in the thesis is:

> German language reviews of doctors by patients (2021), Michael C.  
> https://data.world/mc51/german-language-reviews-of-doctors-by-patients

The original raw dataset and the full stratified corpus are not included in this public repository. This repository contains code, generated figures, result tables, and selected derived sentence-level artefacts used to assess the thesis results. The full corpus and the stratified corpus can be provided separately to the examiners if required.

## Methodological note

The linguistic features are rule-based lexicon counts. They are descriptive approximations and should not be treated as complete linguistic annotation. The topic model is a discovery layer. The linguistic argument of the thesis depends on combining these quantitative outputs with qualitative close reading.
