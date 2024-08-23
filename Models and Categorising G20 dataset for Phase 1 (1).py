#!/usr/bin/env python
# coding: utf-8

# # Random Forests

# ## Task 1

# In[ ]:


import pandas as pd

EXISTtraining = pd.read_csv('EXISTtraining_preprocessed.csv')

print(EXISTtraining.head())

print(EXISTtraining.info())


# In[ ]:


import pandas as pd

EXISTtesting = pd.read_csv('EXISTtesting_preprocessed.csv')

print(EXISTtesting.head())

print(EXISTtesting.info())


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[ ]:


vectorizer = TfidfVectorizer(max_features=1000)  # Consider tuning this parameter
X_train = vectorizer.fit_transform(EXISTtraining['lemmatized_text'])
X_test = vectorizer.transform(EXISTtesting['lemmatized_text'])

y_train_task1 = EXISTtraining['task1']
y_test_task1 = EXISTtesting['task1']


# In[ ]:


rf_classifier_task1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_task1.fit(X_train, y_train_task1)

y_pred_task1 = rf_classifier_task1.predict(X_test)

print("Classification Report for Task 1:")
print(classification_report(y_test_task1, y_pred_task1))
print("Accuracy for Task 1:", accuracy_score(y_test_task1, y_pred_task1))


# ## Task 2

# In[ ]:


train_df_task2 = EXISTtraining[EXISTtraining['task1'] == 'sexist']
test_df_task2 = EXISTtesting[EXISTtesting['task1'] == 'sexist']

X_train_task2 = vectorizer.transform(train_df_task2['lemmatized_text']).toarray()
X_test_task2 = vectorizer.transform(test_df_task2['lemmatized_text']).toarray()

y_train_task2 = train_df_task2['task2']
y_test_task2 = test_df_task2['task2']

rf_classifier_task2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier_task2.fit(X_train_task2, y_train_task2)
# Predict on the test set for Task 2
y_pred_task2 = rf_classifier_task2.predict(X_test_task2)

print("Classification Report for Task 2:")
print(classification_report(y_test_task2, y_pred_task2))
print("Accuracy for Task 2:", accuracy_score(y_test_task2, y_pred_task2))


# # BiLSTM

# ## 1st attempt

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# ### Task 1

# In[ ]:


vocab_size = 10000  # Vocabulary size
max_length = 50     # Maximum length of the sequence
embedding_dim = 64  # Dimensionality of the embedding vector
oov_tok = "<OOV>"   # Out of vocabulary token

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(EXISTtraining['lemmatized_text'])  
word_index = tokenizer.word_index

X_train = pad_sequences(tokenizer.texts_to_sequences(EXISTtraining['lemmatized_text']), maxlen=max_length, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(EXISTtesting['lemmatized_text']), maxlen=max_length, padding='post')

y_train_task1 = np.array(EXISTtraining['task1'].map({'non-sexist': 0, 'sexist': 1}))
y_test_task1 = np.array(EXISTtesting['task1'].map({'non-sexist': 0, 'sexist': 1}))

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dense(24, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall()]
)

model.summary()

# Train the model
num_epochs = 10
history = model.fit(X_train, y_train_task1, epochs=num_epochs, validation_data=(X_test, y_test_task1), batch_size=64)

# Evaluate the model
results = model.evaluate(X_test, y_test_task1)
print("Test Loss, Test Accuracy:", results)

precision = results[2]  
recall = results[3]     
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
print(f"F1-Score: {f1_score}")


# ### Task 2

# In[ ]:


from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(EXISTtraining['task2'])
y_test_encoded = label_encoder.transform(EXISTtesting['task2'])

y_train_task2 = to_categorical(y_train_encoded)
y_test_task2 = to_categorical(y_test_encoded)


# In[ ]:


model_task2 = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dense(24, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='softmax') 
])

model_task2.compile(
    loss='categorical_crossentropy',  
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall()]
)


# In[ ]:


num_epochs = 10
history_task2 = model_task2.fit(
    X_train, y_train_task2, 
    epochs=num_epochs, 
    validation_data=(X_test, y_test_task2), 
    batch_size=64
)

results_task2 = model_task2.evaluate(X_test, y_test_task2)
print("Task 2 - Test Loss, Test Accuracy:", results_task2)


# In[ ]:


from sklearn.metrics import classification_report

y_pred_task2 = model_task2.predict(X_test)
y_pred_task2_classes = np.argmax(y_pred_task2, axis=1)
y_true_task2_classes = np.argmax(y_test_task2, axis=1)

print(classification_report(y_true_task2_classes, y_pred_task2_classes, target_names=label_encoder.classes_))


# ## 2nd attempt

# ### Task 1

# In[ ]:


## 1 Add more dropout rate
## 2 early stopping
## 3 adjust patience parameter

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=4,          
    restore_best_weights=True  
)

vocab_size = 10000  # Vocabulary size
max_length = 50     # Maximum length of the sequence
embedding_dim = 64  # Dimensionality of the embedding vector
oov_tok = "<OOV>"   # Out of vocabulary token

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(EXISTtraining['lemmatized_text'])  
word_index = tokenizer.word_index

X_train = pad_sequences(tokenizer.texts_to_sequences(EXISTtraining['lemmatized_text']), maxlen=max_length, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(EXISTtesting['lemmatized_text']), maxlen=max_length, padding='post')

y_train_task1 = np.array(EXISTtraining['task1'].map({'non-sexist': 0, 'sexist': 1}))
y_test_task1 = np.array(EXISTtesting['task1'].map({'non-sexist': 0, 'sexist': 1}))

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.6),
    Bidirectional(LSTM(32)),
    Dense(24, activation='relu'),
    Dropout(0.6),
    Dense(1, activation='sigmoid')  
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall()]
)

model.summary()

history = model.fit(
    X_train, y_train_task1,
    epochs=20,
    validation_data=(X_test, y_test_task1),
    callbacks=[early_stopping]
)

# Evaluate the model
results = model.evaluate(X_test, y_test_task1)
print("Test Loss, Test Accuracy:", results)

precision = results[2]  
recall = results[3]     
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
print(f"F1-Score: {f1_score}")


# ### Task 2

# In[ ]:


from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(EXISTtraining['task2'])
y_test_encoded = label_encoder.transform(EXISTtesting['task2'])

y_train_task2 = to_categorical(y_train_encoded)
y_test_task2 = to_categorical(y_test_encoded)


# In[ ]:


model_task2 = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dense(24, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='softmax')  
])

model_task2.compile(
    loss='categorical_crossentropy',  
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall()]
)


# In[ ]:


num_epochs = 10
history_task2 = model_task2.fit(
    X_train, y_train_task2, 
    epochs=num_epochs, 
    validation_data=(X_test, y_test_task2), 
    batch_size=64
)

# Evaluate the model
results_task2 = model_task2.evaluate(X_test, y_test_task2)
print("Task 2 - Test Loss, Test Accuracy:", results_task2)


# In[ ]:


from sklearn.metrics import classification_report

y_pred_task2 = model_task2.predict(X_test)
y_pred_task2_classes = np.argmax(y_pred_task2, axis=1)
y_true_task2_classes = np.argmax(y_test_task2, axis=1)

print(classification_report(y_true_task2_classes, y_pred_task2_classes, target_names=label_encoder.classes_))


# # Support Vector Machines

# ## Task 1

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[ ]:


EXISTtraining = EXISTtraining.dropna(subset=['lemmatized_text'])
EXISTtesting = EXISTtesting.dropna(subset=['lemmatized_text'])


# In[ ]:


vectorizer = TfidfVectorizer(max_features=1000) 
X_train = vectorizer.fit_transform(EXISTtraining['lemmatized_text'])
X_test = vectorizer.transform(EXISTtesting['lemmatized_text'])

y_train_task1 = EXISTtraining['task1']
y_test_task1 = EXISTtesting['task1']


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10, 100],  
    'gamma': [1, 0.1, 0.01, 0.001],  
    'kernel': ['linear', 'rbf']  
}

svm_classifier = SVC(class_weight='balanced')

grid_search = GridSearchCV(svm_classifier, param_grid, refit=True, verbose=2, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train_task1) 

print("Best parameters found: ", grid_search.best_params_)

best_svm = grid_search.best_estimator_

y_pred = best_svm.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test_task1, y_pred))  
print("Accuracy:", accuracy_score(y_test_task1, y_pred))  


# ## Task 2

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(
    EXISTtraining['lemmatized_text'], 
    EXISTtraining['task2'], 
    test_size=0.2, 
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

svm = SVC()
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

best_params = grid_search.best_params_
best_svm = grid_search.best_estimator_

y_pred = best_svm.predict(X_test_tfidf)

# Evaluate the model
print("Best parameters found: ", best_params)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# # Ensemble model

# ## Task 1

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

EXISTdata = pd.concat([EXISTtraining, EXISTtesting])

X = EXISTdata['lemmatized_text']
y = EXISTdata['task1'].map({'non-sexist': 0, 'sexist': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_tfidf, y_train)
best_params = grid_search.best_params_
best_params = {'C': 1, 'gamma': 1, 'kernel': 'rbf'}  # Replace with actual best_params

svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'], probability=True)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('svm', svm), 
    ('rf', rf), 
    ('lr', lr)
], voting='soft')

voting_clf.fit(X_train_tfidf, y_train)

y_pred_voting = voting_clf.predict(X_test_tfidf)

# Evaluate the model
print("Voting Classifier - Classification Report:")
print(classification_report(y_test, y_pred_voting))
print("Voting Classifier - Accuracy:", accuracy_score(y_test, y_pred_voting))


# ## Task 2

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

X = EXISTtraining['lemmatized_text']
y = EXISTtraining['task2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Post-split - X_train:", len(X_train), "y_train:", len(y_train))

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

print("Post-resampling - X_train_res:", X_train_res.shape, "y_train_res:", y_train_res.shape)

svm = SVC(probability=True)
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000)

voting_clf = VotingClassifier(estimators=[('svm', svm), ('rf', rf), ('lr', lr)], voting='soft')

voting_clf.fit(X_train_res, y_train_res)

y_pred = voting_clf.predict(X_test_tfidf)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))



# ## Task 1 with tenfold

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

EXISTdata = pd.concat([EXISTtraining, EXISTtesting])

X = EXISTdata['lemmatized_text']
y = EXISTdata['task1'].map({'non-sexist': 0, 'sexist': 1})

vectorizer = TfidfVectorizer(max_features=1000)
svm = SVC(probability=True, C=1, gamma=1, kernel='rbf')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('svm', svm),
    ('rf', rf),
    ('lr', lr)
], voting='soft')

pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('voting_clf', voting_clf)
])

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')

print("Cross-Validation Scores:", scores)
print("Average Accuracy:", scores.mean())


# ## Task 2 with tenfold

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import numpy as np

X = EXISTtraining['lemmatized_text']
y = EXISTtraining['task2']

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

svm = SVC(probability=True)
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000)

voting_clf = VotingClassifier(estimators=[
    ('svm', svm), 
    ('rf', rf), 
    ('lr', lr)
], voting='soft')

pipeline = make_pipeline(SMOTE(random_state=42), voting_clf)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X_tfidf, y, cv=kfold, scoring='accuracy')

print("Cross-Validation Scores:", scores)
print("Average Accuracy:", np.mean(scores))


# # Applying Ensemble model to G20 data

# In[ ]:


G20data = pd.read_csv('G20data_preprocessed.csv')
print(G20data.head())

print(G20data.info())


# In[ ]:


G20data = G20data.dropna(subset=['lemmatized_text'])

G20_text_tfidf = vectorizer.transform(G20data['lemmatized_text'])

G20data['predicted_sexist'] = voting_clf.predict(G20_text_tfidf)

G20data['predicted_sexist'] = G20data['predicted_sexist'].map({0: 'non-sexist', 1: 'sexist'})

print(G20data[['TweetContent', 'predicted_sexist']].head())


# In[ ]:


print(G20data[['cleaned_text', 'lemmatized_text']].head())

G20_text_tfidf = vectorizer.transform(G20data['lemmatized_text'])
print("Shape of TF-IDF matrix:", G20_text_tfidf.shape)

predictions = voting_clf.predict(G20_text_tfidf)
print("Sample predictions:", predictions[:5])

G20data['predicted_sexist'] = predictions
print("NaNs in predictions:", G20data['predicted_sexist'].isnull().sum())


# In[ ]:


G20data['predicted_sexist'] = predictions

print(G20data[['TweetContent', 'predicted_sexist']].head())


# In[ ]:


counts = G20data['predicted_sexist'].value_counts()

print(counts)


# In[ ]:


G20data.to_csv('G20_predictions.csv', index=False)


# # Comparing Topics

# In[ ]:


import pandas as pd
G20data_results = pd.read_csv('G20_predictions.csv')


# In[ ]:


import pandas as pd
import re

categories = {
    "ivanka_trump": [
        "Ivanka Trump", "Ivanka", "Trump’s daughter", "Trump daughter", "Ivanka's", "IvankaTrump", "IvankaTrumps", "Trump's Daughter:", "nepotism"
    ],
    "angela_merkel": [
        "Merkel", "Angela Merkel", "Merkel's", "Chancellor", "Angela Merkel's", "Angela's", "AngelaMerkel", "Angela", "merkels", "AngelaMerkels", "MerkelFail", "Mrrkel"
    ],
    "theresa_may": [
        "Theresa May", "Theresa", "Theresa's", "Theresas", "Theresa May's", "Theresa May's", "UK Prime Minister", "UK PM", "TheresaMay", "TheresaMays", "Mrs. May", "theresa_may", "May"
    ],
    "erna_solberg": [
        "Erna Solberg", "Erna Solberg's", "Erna's", "Solberg's", "Erna", "Solberg", "Solbergs", "Norwegian prime minister", "Norwegian PM", "ErnaSolberg", "ErnaSolbergs"
    ],
    "women": [
         "EmpoweringWomen", "ladies", "lady", "Women", "women's", "woman", "woman's", "womens", "womans", "female", "females", "female's", "girl", "girl's", "Gender", "Feminism", "feminist", "'feminist's", "feminists"
    ],
    "we_fi": [
        "Women Entrepreneurs Finance Initiative", "WeFi", "We-fi", "women’s fund"
    ]
}

categories['all_four'] = list(set(categories['ivanka_trump'] + categories['angela_merkel'] + categories['theresa_may'] + categories['erna_solberg']))

def create_regex_pattern(keywords):
    pattern_parts = []
    for keyword in keywords:
        part = r'\b' + re.escape(keyword) + r'\b'
        pattern_parts.append(part)
    return re.compile('|'.join(pattern_parts), re.IGNORECASE)

# Apply categories to each tweet
def categorize_tweet(tweet, regex_patterns):
    categories_matched = {}
    for category, pattern in regex_patterns.items():
        if pattern.search(tweet):
            categories_matched[category] = 1
        else:
            categories_matched[category] = 0
    return pd.Series(categories_matched)

regex_patterns = {category: create_regex_pattern(keywords) for category, keywords in categories.items()}

G20data_results['TweetContent'] = G20data_results['TweetContent'].astype(str)  # Ensure text is in string format
category_columns = G20data_results['TweetContent'].apply(categorize_tweet, regex_patterns=regex_patterns)

G20data_results = pd.concat([G20data_results, category_columns], axis=1)

print(G20data_results.head())


# In[ ]:


import re

def count_category_keywords(text, keywords):
    pattern = r'\b(' + '|'.join([re.escape(kw) for kw in keywords]) + r')\b'
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

category_counts = {}

for category, keywords in categories.items():
    category_counts[category] = G20data_results['TweetContent'].apply(lambda x: count_category_keywords(x, keywords)).sum()

category_counts_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])

print(category_counts_df)


# In[ ]:


G20data_results.to_csv('G20_CAT.csv', index=False)


# In[ ]:


G20dataCAT = pd.read_csv('G20_CAT.csv')


# In[ ]:


sexist_categories = ['ideological-inequality', 'misogyny-non-sexual-violence', 'objectification', 'sexual-violence', 'stereotyping-and-dominance']

print("Distribution of predictions:")
print(G20dataCAT['predicted_sexist'].value_counts())

sexist_tweet_counts = {}

for category in categories:
    filtered_data = G20dataCAT[(G20dataCAT[category] == 1) & (G20dataCAT['predicted_sexist'].isin(sexist_categories))]
    count = filtered_data.shape[0]
    sexist_tweet_counts[category] = count

sexist_tweet_counts_df = pd.DataFrame(list(sexist_tweet_counts.items()), columns=['Category', 'Sexist Tweet Count'])
print("\nSexist Tweet Counts by Category:")
print(sexist_tweet_counts_df)


# In[ ]:


total_tweet_counts = {}
for category in categories:
    total_tweet_counts[category] = G20dataCAT[G20dataCAT[category] == 1].shape[0]

category_percentages = {}
for category in categories:
    if total_tweet_counts[category] > 0:  
        percent = (sexist_tweet_counts[category] / total_tweet_counts[category]) * 100
    else:
        percent = 0  
    category_percentages[category] = percent

percentages_df = pd.DataFrame(list(category_percentages.items()), columns=['Category', 'Sexist Tweet Percentage'])
print("\nSexist Tweet Percentages by Category:")
print(percentages_df)


# In[ ]:


import pandas as pd

sexist_data = G20dataCAT[G20dataCAT['predicted_sexist'].isin(['ideological-inequality', 'sexual-violence', 'objectification', 'stereotyping-dominance', 'misogyny-non-sexual-violence'])]

melted_data = sexist_data.melt(id_vars=['predicted_sexist'], value_vars=['all_four', 'ivanka_trump', 'angela_merkel', 'theresa_may', 'erna_solberg', 'women', 'we_fi'],
                               var_name='Topic', value_name='Flag')

filtered_melted_data = melted_data[melted_data['Flag'] == 1]

pivot_table = filtered_melted_data.pivot_table(index='Topic', columns='predicted_sexist', aggfunc='size', fill_value=0)

print(pivot_table)


# In[ ]:


import pandas as pd

topic_sums = pivot_table.sum(axis=1)

pivot_percentage = pivot_table.div(topic_sums, axis=0) * 100

print(pivot_percentage)


# # Visualisation

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

categories = ['Sexual Violence', 'Ideological Inequality', 'Stereotyping and Dominance', 'Misogyny and Non-Sexual Violence', 'Objectification']
values = [78, 25, 15, 8, 4]

df_new = pd.DataFrame({
    'Category': categories,
    'Count': values
})

df_sorted_new = df_new.sort_values('Count', ascending=False)

plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

bar_plot_new = sns.barplot(
    x='Category', 
    y='Count', 
    data=df_sorted_new, 
    color='royalblue'  
)


for p in bar_plot_new.patches:
    height = p.get_height()    
    plt.text(p.get_x() + p.get_width() / 2,  
             height + 1,  
             '{:1.0f}'.format(height),  
             ha='center', va='bottom', fontsize=15)  

def wrap_labels(labels, width=15):
    return [textwrap.fill(label, width) for label in labels]

wrapped_labels = wrap_labels(df_sorted_new['Category'], width=15)
bar_plot_new.set_xticks(range(len(wrapped_labels)))  
bar_plot_new.set_xticklabels(wrapped_labels, fontsize=15, ha='center')

for label in bar_plot_new.get_xticklabels():
    label.set_rotation(0)  
    label.set_ha('center')  
    label.set_va('bottom') 
    label.set_y(-0.1)  

plt.subplots_adjust(bottom=0.25) 

plt.title('Distribution of Tweets Across Sexist Categories', fontsize=17)
plt.xlabel('Categories',fontsize=17, labelpad=20)
plt.ylabel('Number of Instances', fontsize=17)

plt.tight_layout() 
plt.show()

