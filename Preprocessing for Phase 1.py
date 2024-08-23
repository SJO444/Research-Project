#!/usr/bin/env python
# coding: utf-8

# # 1. Translating EXIST training dataset

# In[1]:


import pandas as pd

EXISTtraining = pd.read_csv('EXIST2021_training.tsv', delimiter='\t')

print(EXISTtraining.head())

print(EXISTtraining.info())


# In[ ]:


pip install googletrans


# In[2]:


from googletrans import Translator, constants


# In[3]:


from pprint import pprint


# In[4]:


translator = Translator()


# In[ ]:


pip install googletrans==3.1.0a0


# In[5]:


spanish_tweets = EXISTtraining[EXISTtraining['language'] == 'es']


# In[6]:


spanish_tweets = spanish_tweets.copy()


# In[7]:


def translate_text(text, lang='en'):
    try:
        return translator.translate(text, dest=lang).text
    except Exception as e:
        print(f"Error: {e}")
        return text


# In[8]:


import time

def safe_translate(text):
    """A safe translation function with error handling."""
    try:
        return translate_text(text, 'en')
    except Exception as e:
        print(f"Error translating text: {e}")
        return text


# In[9]:


spanish_tweets.loc[:, 'translated_text'] = spanish_tweets['text'].apply(safe_translate)


# In[10]:


print(spanish_tweets[['text', 'translated_text']].head())


# In[11]:


print("Total tweets translated:", spanish_tweets['translated_text'].notna().sum())


# In[12]:


EXISTtraining.loc[spanish_tweets.index, 'text'] = spanish_tweets['translated_text']


# In[ ]:


EXISTtraining.to_csv('EXSITtraining_translated_dataset.csv', index=False)


# # 2. Translating EXIST testing dataset

# In[13]:


EXISTtesting = pd.read_csv('EXISTtesting.tsv', delimiter='\t')

print(EXISTtesting.head())

print(EXISTtesting.info())


# In[14]:


spanish_tweets2 = EXISTtesting[EXISTtesting['language'] == 'es']


# In[15]:


spanish_tweets2 = spanish_tweets2.copy()


# In[16]:


spanish_tweets2.loc[:, 'translated_text'] = spanish_tweets2['text'].apply(safe_translate)


# In[17]:


print(spanish_tweets2[['text', 'translated_text']].head())


# In[ ]:


EXISTtesting.loc[spanish_tweets2.index, 'text'] = spanish_tweets2['translated_text']


# In[ ]:


EXISTtesting.to_csv('EXISTtesting_translated_dataset.csv', index=False)


# # 3. Filtering and Translating G20 Dataset for Phase 1

# In[ ]:


import pandas as pd
G20data = pd.read_csv('G20Hamburg.csv')
print(G20data.info())
print(G20data.describe())


# In[ ]:


# rename columns
G20data = G20data.rename(columns={
    'Tweet': 'TweetNumber',
    'Tweet.1': 'TweetContent',
})

print(G20data.columns)


# In[ ]:


# Filter out retweets
G20data = G20data[G20data['Is Retweet'] == False]


# In[ ]:


# Remove duplicates based on tweet content
G20data = G20data.drop_duplicates(subset=['TweetContent'])
# Drop rows where the tweet content is missing
G20data = G20data.dropna(subset=['TweetContent'])


# In[ ]:


print(G20data.info())
print(G20data.describe())


# ## 3.1. Filtering G20 dataset for Phase 1
# 

# In[ ]:


import re
import pandas as pd

# List of keywords and phrases related to Phase 1
keywords = [
    "Ivanka Trump", "Ivanka", "Trump’s daughter", "Trump daughter" "Theresa May", "Theresa", "Theresa's", "Theresa May's", "Theresa May's", "UK Prime Minister", "UK PM",
    "Erna Solberg", "Erna Solberg's", "Erna's", "Solberg's", "Erna", "Solberg", "Norwegian prime minister", "Norwegian PM", "Angela Merkel", "Angela", "Merkel", "Angela Merkel's", "Merkel's", "Angela's",
    "Chancellor of Germany", "Chancellor", "Women", "women's", "woman", "woman's", "female", "female's", "girl", "girl's", "Gender", "Feminism",
    "feminist", "'feminist's" "Women Entrepreneurs Finance Initiative", "WeFi", "We-fi", "women’s fund",
    "Trump's Tochter", "Trump Tochter" "Premierminister des Vereinigten Königreichs", "UK Premierminister",
    "Norwegischer Premierminister", "Bundeskanzlerin von Deutschland", "Bundeskanzlerin Deutschlands",
    "Bundeskanzler", "Frauen", "für Frauen", "damen","frau", "weiblich", "hündinnen", "weibchen", "mädchen", "mädels", "Geschlecht", "Feminismus",
    "feministin", "feministinnen", "Finance Initiative für Unternehmerinnen", "Frauen-Unternehmerinnen-Finanzinitiative",
    "Frauen Fonds", "Frauenrechte", "Unternehmerinnen", "Frauenförderung", "Gleichberechtigung", "Geschlechtergerechtigkeit"
]


# In[ ]:


def adjust_umlauts_and_specials(keyword):
    replacements = {
        'ä': '(ae|ä)',
        'ö': '(oe|ö)',
        'ü': '(ue|ü)',
        'ß': '(ss|ß)'
    }
    for german_char, replacement in replacements.items():
        keyword = re.sub(re.escape(german_char), replacement, keyword, flags=re.IGNORECASE)
    return keyword

def enhance_keyword_regex(keyword):
    keyword = adjust_umlauts_and_specials(keyword)
    
    keyword = re.escape(keyword)
    
    keyword = keyword.replace(r'\ ', r'\s*')
    
    punctuations = ["'", "-", "."]  
    for punct in punctuations:
        keyword = keyword.replace(re.escape(punct), re.escape(punct) + '?')
    
    return keyword

regex_pattern = r'\b(' + '|'.join(enhance_keyword_regex(keyword) for keyword in keywords) + r')\b'


# In[ ]:


G20data['matches'] = G20data['TweetContent'].str.contains(regex_pattern, case=False, na=False)
filtered_G20data = G20data[G20data['matches']]


# In[ ]:


pd.set_option('display.max_colwidth', None) 
pd.set_option('display.max_rows', 500)  
print(filtered_G20data.head())  


# In[ ]:


print(filtered_G20data['Tweet Language'].value_counts())

print(G20data['Tweet Language'].value_counts())


# In[ ]:


filtered_G20data.to_csv('G20_filtered_beforeT.csv', index=False)


# In[ ]:


G20_filtered_beforeT = pd.read_csv('G20_filtered_beforeT.csv')

print(G20_filtered_beforeT.head())
print(G20_filtered_beforeT.info())


# In[ ]:


# Count occurrences of each keyword
keyword_counts = {keyword: 0 for keyword in keywords}

for keyword in keywords:
    keyword_regex = enhance_keyword_regex(keyword)
    pattern = re.compile(r'\b' + keyword_regex + r'\b', re.IGNORECASE)
    keyword_counts[keyword] = filtered_G20data['TweetContent'].apply(lambda x: bool(pattern.search(str(x)))).sum()

keyword_counts_df = pd.DataFrame(keyword_counts.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)

keyword_counts_df.to_csv('keyword_counts.csv', index=False)

print(keyword_counts_df)


# ## 3.2. Translating G20 dataset for Phase 1

# In[ ]:


G20data_filtered_Ger = filtered_G20data[filtered_G20data['Tweet Language'] == 'de']


# In[ ]:


G20data_filtered_Ger = G20data_filtered_Ger.copy()


# In[ ]:


G20data_filtered_Ger.loc[:, 'translated_text'] = G20data_filtered_Ger['TweetContent'].apply(safe_translate)


# In[ ]:


print(G20data_filtered_Ger[['TweetContent', 'translated_text']].head())
print(G20data_filtered_Ger[['TweetContent', 'translated_text']].info())


# In[ ]:


G20translate['matches'].value_counts()


# ## 3.3. Merging translated tweets with English tweets

# In[ ]:


merged_dataset = G20_filtered_beforeT.merge(
    G20data_filtered_Ger[['UniqueID', 'translated_text']],
    on='UniqueID',
    how='left',
    suffixes=('', '_translated')
)

merged_dataset['TweetContent'] = merged_dataset['translated_text'].combine_first(merged_dataset['TweetContent'])


# In[ ]:


merged_dataset.drop(columns=['translated_text'], inplace=True)

merged_dataset.to_csv('G20_filtered_after_translation1.csv', index=False)


# In[ ]:


print("Original G20_filtered_beforeT data:")
print(G20_filtered_beforeT.head())


# In[ ]:


merged_dataset2 = pd.read_csv('G20_filtered_after_translation1.csv')

print(merged_dataset2.head())

print(merged_dataset2.info())


# In[ ]:


print(G20translate.columns)


# # 4. Preprocessing EXIST training and testing datasets

# In[ ]:


import pandas as pd

EXISTtraining = pd.read_csv('EXSITtraining_translated_dataset.csv')

print(EXISTtraining.head())

print(EXISTtraining.info())


# In[ ]:


EXISTtesting = pd.read_csv('EXISTtesting_translated_dataset.csv')

print(EXISTtesting.head())

print(EXISTtesting.info())


# In[ ]:


EXISTtraining = EXISTtraining.drop_duplicates(subset='text', keep='first')

print(EXISTtraining.info())


# In[ ]:


EXISTtesting = EXISTtesting.drop_duplicates(subset='text', keep='first')

print(EXISTtesting.info())


# In[ ]:


import nltk
nltk.download('wordnet')  # For lemmatisation
nltk.download('punkt')    # For tokenisation
nltk.download('stopwords')  # For stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[ ]:


stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'http\S+', 'URL', text)

    text = re.sub(r'[^a-z\s#@]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    
    text = re.sub(r'#(\w+)', r'<hashtag> \1 </hashtag>', text)

    text = re.sub(r'@(\w+)', r'<mention> \1 </mention>', text)

    return text

def lemmatize_text(text):
    tokens = text.split()

    lemmatized_tokens = []

    special_tags = {'<hashtag>', '</hashtag>', '<mention>', '</mention>'}

    for token in tokens:
        if token in special_tags:
            lemmatized_tokens.append(token)
        else:
            if token not in stop_words and token.isalpha():
                lemmatized_token = lemmatizer.lemmatize(token)
                lemmatized_tokens.append(lemmatized_token)

    lemmatized_text = ' '.join(lemmatized_tokens)
    
    return lemmatized_text


# In[ ]:


# Apply preprocessing to training data
EXISTtraining['cleaned_text'] = EXISTtraining['text'].apply(preprocess_text)
EXISTtraining['lemmatized_text'] = EXISTtraining['cleaned_text'].apply(lemmatize_text)

print(EXISTtraining[['text', 'cleaned_text', 'lemmatized_text']].head())


# In[ ]:


# Apply preprocessing to testing data
EXISTtesting['cleaned_text'] = EXISTtesting['text'].apply(preprocess_text)
EXISTtesting['lemmatized_text'] = EXISTtesting['cleaned_text'].apply(lemmatize_text)

print(EXISTtesting[['text', 'cleaned_text', 'lemmatized_text']].head())


# # 5. Preprocessing Phase 1 G20 Data

# In[ ]:


import pandas as pd

G20data = pd.read_csv('G20_filtered_after_translation1.csv')

print(G20data.head())

print(G20data.info())


# In[ ]:


# Apply preprocessing to filtered G20 data
G20data['cleaned_text'] = G20data['TweetContent'].apply(preprocess_text)
G20data['lemmatized_text'] = G20data['cleaned_text'].apply(lemmatize_text)

print(G20data[['TweetContent', 'cleaned_text', 'lemmatized_text']].head())


# In[ ]:


G20data.to_csv('G20data_preprocessed.csv', index=False)
EXISTtesting.to_csv('EXISTtesting_preprocessed.csv', index=False)

EXISTtraining.to_csv('EXISTtraining_preprocessed.csv', index=False)


# # Descriptive Statistics

# In[ ]:


import re
import pandas as pd

# Define keywords for each category
categories = {
    "all_four": [
        "Ivanka Trump", "Ivanka", "Trump’s daughter", "Trump daughter", "Ivanka's", 
        "Merkel", "Angela Merkel", "Merkel's", "Chancellor", "Angela Merkel's", "Angela's", 
        "Theresa May", "Theresa", "Theresa's", "Theresa May's", "Theresa May's", "UK Prime Minister", "UK PM", 
        "Erna Solberg", "Erna Solberg's", "Erna's", "Solberg's", "Erna", "Solberg", "Norwegian prime minister", "Norwegian PM"
    ],
    "ivanka_trump": [
        "Ivanka Trump", "Ivanka", "Trump’s daughter", "Trump daughter", "Ivanka's", "Trump's Daughter:"
    ],
    "angela_merkel": [
        "Merkel", "Angela Merkel", "Merkel's", "Chancellor", "Angela Merkel's", "Angela's"
    ],
    "theresa_may": [
        "Theresa May", "Theresa", "Theresa's", "Theresa May's", "Theresa May's", "UK Prime Minister", "UK PM", "Mrs. May", "theresa_may"
    ],
    "erna_solberg": [
        "Erna Solberg", "Erna Solberg's", "Erna's", "Solberg's", "Erna", "Solberg", "Norwegian prime minister", "Norwegian PM"
    ],
    "women": [
        "Women", "women's", "woman", "woman's", "womens", "womans", "female", "females", "female's", "girl", "girl's", "Gender", "Feminism", "feminist", "'feminist's", "feminists",
    ],
    "we_fi": [
        "Women Entrepreneurs Finance Initiative", "WeFi", "We-fi", "women’s fund"
    ]
}

def adjust_umlauts_and_specials(keyword):
    replacements = {
        'ä': '(ae|ä)',
        'ö': '(oe|ö)',
        'ü': '(ue|ü)',
        'ß': '(ss|ß)'
    }
    for german_char, replacement in replacements.items():
        keyword = re.sub(re.escape(german_char), replacement, keyword, flags=re.IGNORECASE)
    return keyword

def enhance_keyword_regex(keyword):
    keyword = adjust_umlauts_and_specials(keyword)
    
    keyword = re.escape(keyword)
    
    keyword = keyword.replace(r'\ ', r'\s*')
    
    punctuations = ["'", "-", "."]  
    for punct in punctuations:
        keyword = keyword.replace(re.escape(punct), re.escape(punct) + '?')
    
    return keyword

counts = {}
for category, keywords in categories.items():
    regex_pattern = r'\b(' + '|'.join(enhance_keyword_regex(keyword) for keyword in keywords) + r')\b'
    G20data[category + '_matches'] = G20data['TweetContent'].str.contains(regex_pattern, case=False, na=False)
    counts[category] = G20data[category + '_matches'].sum()

for category, count in counts.items():
    print(f"Number of tweets in category '{category}': {count}")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

categories = ['Angela Merkel', 'Ivanka Trump', 'Theresa May', 'Erna Solberg', 'General Discussion on Women', 'We-Fi', 'All Four Female Politicians']
values = [4700, 2557, 286, 11, 960, 30, 7440]

df_new = pd.DataFrame({
    'Category': categories,
    'Count': values
})

df_sorted_new = df_new.sort_values('Count', ascending=False)

plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

colors = ['royalblue' if cat != 'All Four Female Politicians' else 'darksalmon' for cat in df_sorted_new['Category']]
bar_plot_new = sns.barplot(
    x='Category', 
    y='Count', 
    data=df_sorted_new, 
    palette=colors
)

for p in bar_plot_new.patches:
    height = p.get_height()    
    plt.text(p.get_x() + p.get_width() / 2,  
             height + 50, 
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

plt.title('Mentions Related to Women', fontsize=17)
plt.xlabel('Categories', fontsize=17, labelpad=20)  
plt.ylabel('Number of Mentions', fontsize=17)



plt.tight_layout() 
plt.show()

