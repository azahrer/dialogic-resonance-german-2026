"""Analysis

This is the source code for the analysis presented in the research article Zahrer & Brathe (2026).

It contains the folowoing steps:
1. data preperation (i.e. annotated linguistic data)
2. training of a random forest classifier
3. check the model's accuracy with a test data set
4. get the feature importances of the annotated categories
5. validate the models performance with a 5-fold cross-validation
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import graphviz
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import export_graphviz
from pathlib import Path

"""**Data loading and preparation**"""

orig_data = pd.read_csv("./data/training_data.csv", sep='\t')
data = pd.DataFrame()

orig_test_data = pd.read_csv("/home/GRASS/test_data.csv", sep='\t')
test_data = pd.DataFrame()

# copy indexical data (training data)
data['sa_interval'] = orig_data['SA-Intervall Nr.']
data['speaker_id'] = orig_data['Sprecher-ID']
data['textgrid_id'] = orig_data['TextGrid-ID']

# copy indexical data (test data)
test_data['sa_interval'] = orig_test_data['SA-Intervall Nr.']
test_data['speaker_id'] = orig_test_data['Sprecher-ID']
test_data['textgrid_id'] = orig_test_data['TextGrid-ID']

# Convert type of SA
label_encoder = LabelEncoder()

# 1 is approval, 0 is denial
data['approval'] = label_encoder.fit_transform(orig_data['Zustimmung'])

test_data['approval'] = label_encoder.fit_transform(orig_test_data['Zustimmung'])

# convert all boolean values (training data)
data['mapping_to_ref_utterance'] = orig_data["Mapping auf Bezugsäußerung"].map({'ja':1, 'nein':0})
data['information_replaced'] = orig_data["Information ersetzt"].map({'ja':1, 'nein':0})
data['information_added'] = orig_data["Information hinzugefügt"].map({'ja':1, 'nein':0})
data['information_paraphrase'] = orig_data["Information paraphrasiert"].map({'ja': 1, 'nein':0})
data['answ_to_polar_question'] = orig_data['Antwort auf Entscheidungsfrage'].map({'ja':1, 'nein':0})
data['answ_to_alt_question'] = orig_data['Antwort auf Alternativenfrage'].map({'ja':1, 'nein':0})

# convert all boolean values (test data)
test_data['mapping_to_ref_utterance'] = orig_test_data["Mapping auf Bezugsäußerung"].map({'ja':1, 'nein':0})
test_data['information_replaced'] = orig_test_data["Information ersetzt"].map({'ja':1, 'nein':0})
test_data['information_added'] = orig_test_data["Information hinzugefügt"].map({'ja':1, 'nein':0})
test_data['information_paraphrase'] = orig_test_data["Information paraphrasiert"].map({'ja': 1, 'nein':0})
test_data['answ_to_polar_question'] = orig_test_data['Antwort auf Entscheidungsfrage'].map({'ja':1, 'nein':0})
test_data['answ_to_alt_question'] = orig_test_data['Antwort auf Alternativenfrage'].map({'ja':1, 'nein':0})

orig_data["Informationstyp ersetzt"] = orig_data["Informationstyp ersetzt"].replace('NA', np.nan)
orig_data["Informationstyp hinzugefügt"] = orig_data["Informationstyp hinzugefügt"].replace('NA', np.nan)

# conversion for test data
orig_test_data["Informationstyp ersetzt"] = orig_test_data["Informationstyp ersetzt"].replace('NA', np.nan)
orig_test_data["Informationstyp hinzugefügt"] = orig_test_data["Informationstyp hinzugefügt"].replace('NA', np.nan)

# convert types of information management (training data)
data['type_info_replaced'] = label_encoder.fit_transform(orig_data["Informationstyp ersetzt"])
data['type_info_added'] = label_encoder.fit_transform(orig_data["Informationstyp hinzugefügt"])
data['polarity_change'] = label_encoder.fit_transform(orig_data['polarity change'])

# convert types of information management (training data)
test_data['type_info_replaced'] = label_encoder.fit_transform(orig_test_data["Informationstyp ersetzt"])
test_data['type_info_added'] = label_encoder.fit_transform(orig_test_data["Informationstyp hinzugefügt"])
test_data['polarity_change'] = label_encoder.fit_transform(orig_test_data['polarity change'])

# in this model, I gloss over the individual types of particles and simply say 'yes' od 'no'
data['particle_approval'] = (orig_data['Partikel Zustimmung'].isna()==False)
data['particle_refusal'] = (orig_data['Partikel Ablehnung'].isna()==False)

# now the particles for the test data
test_data['particle_approval'] = (orig_test_data['Partikel Zustimmung'].isna()==False)
test_data['particle_refusal'] = (orig_test_data['Partikel Ablehnung'].isna()==False)

"""**Model training and testing**"""

# split the training data to target and predictor variables
X_train = data.drop(['approval', 'sa_interval', 'speaker_id', 'textgrid_id'], axis=1)
y_train = data['approval']

# now the test data
X_test = test_data.drop(['approval', 'sa_interval', 'speaker_id', 'textgrid_id'], axis=1)
y_test = test_data['approval']

# now fit the model to the training data
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

pred_test = rf.predict(X_test)

acc_score = accuracy_score(y_test, pred_test)
print("Accuracy score: ", acc_score)

cm = confusion_matrix(y_test, pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

plt.savefig("results/confusion_matrix.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();

plt.savefig("results/feature_importances.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

"""**Cross validation**"""

# perform a 5-fold cross validation
rf = RandomForestClassifier()

# the KFold class can be used to shuffle the data
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(rf, X_train, y_train, cv=kfold, scoring='accuracy')
print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation accuracy: ", cv_scores.mean())
print("Standard deviation of scores:", cv_scores.std())
