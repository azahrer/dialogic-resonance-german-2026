# Introduction

This repository contains supplementary material for the research article published in Cognitive Linguistics.

**Reference:** Zahrer, Alexander & Nele Brathe. 2026. Pairing Dialogic Syntax with Machine Learning: A Corpus-Based Study of Dialogic Resonance and Stance Alignment in German Conversation. *Cognitive Linguistics*. DOI: ...

# Data

The data consists in two sets of annotated data: a **training set data** to train the classifier model and a **test data set** to check the models accuracy. For explanations of the encoded values, see the research article cited above. The data is based on the GRASS corpus of spoken Austrian German (Schuppler, Hagmüller & Zahrer 2017). This corpus is not publicly available but can be obtained on request for research purposes [under this link](https://www.spsc.tugraz.at/databases-and-tools/grass-the-graz-corpus-of-read-and-spontaneous-speech.html).

# Analysis

The statistical analyses were conducted in Python and *scikit-learn*. The complete analysis code is provided in analysis.py.

# Running the analysis

Download or clone this repository, open a terminal in the repository's root directory, and execute:

```
python analysis.py
```

The script uses the dataset provided in the data/ directory to reproduce the analyses reported in the article.

# Output

The script generates the statistical results and saves two figures in the results/ directory:

* confusion_matrix.png – Confusion matrix showing the model's classification performance on the test data.
* feature_importances.png – Bar chart displaying the relative importance of the predictor variables in the random forest model.

# License

[...]

