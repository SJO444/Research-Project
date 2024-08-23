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


EXISTtraining.to_csv('translated_dataset.csv', index=False)


# # 2. Translating EXIST testing dataset

# In[13]:


# Load the TSV file
EXISTtesting = pd.read_csv('EXISTtesting.tsv', delimiter='\t')

# Display the first few rows of the DataFrame
print(EXISTtesting.head())

# Optionally, check the structure of your DataFrame
print(EXISTtesting.info())


# In[14]:


# Assuming there's a column named 'language' that specifies the language of each tweet
spanish_tweets2 = EXISTtesting[EXISTtesting['language'] == 'es']


# In[15]:


# Create a copy to avoid modifying the original DataFrame
spanish_tweets2 = spanish_tweets2.copy()


# In[16]:


# Apply translation and handle large dataset in chunks if necessary
spanish_tweets2.loc[:, 'translated_text'] = spanish_tweets2['text'].apply(safe_translate)


# In[17]:


print(spanish_tweets2[['text', 'translated_text']].head())


# In[ ]:


# Assuming 'df' is your main DataFrame and it includes all tweets, not just Spanish
EXISTtesting.loc[spanish_tweets2.index, 'text'] = spanish_tweets2['translated_text']


# In[ ]:


EXISTtesting.to_csv('EXISTtesting_translated_dataset.csv', index=False)


# # 3. Filtering and Translating G20 Dataset for Phase 1

# In[ ]:


# Load the TSV file
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

# Verify the changes
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

# List of keywords and phrases, escaping regex special characters where necessary
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
    # Replace German umlauts and sharp S
    replacements = {
        'ä': '(ae|ä)',
        'ö': '(oe|ö)',
        'ü': '(ue|ü)',
        'ß': '(ss|ß)'
    }
    # Use a regular expression to replace each umlaut in the keyword with the appropriate pattern
    for german_char, replacement in replacements.items():
        keyword = re.sub(re.escape(german_char), replacement, keyword, flags=re.IGNORECASE)
    return keyword

def enhance_keyword_regex(keyword):
    # First adjust for umlauts and special German characters
    keyword = adjust_umlauts_and_specials(keyword)
    
    # Escape all regex special characters first
    keyword = re.escape(keyword)
    
    # Replace escaped spaces with optional spaces (for compound words)
    keyword = keyword.replace(r'\ ', r'\s*')
    
    # Make certain punctuations optional
    punctuations = ["'", "-", "."]  # Add more if needed
    for punct in punctuations:
        keyword = keyword.replace(re.escape(punct), re.escape(punct) + '?')
    
    return keyword

# Create regex pattern using the enhanced function
regex_pattern = r'\b(' + '|'.join(enhance_keyword_regex(keyword) for keyword in keywords) + r')\b'


# In[ ]:


# Assuming G20data is loaded and contains a 'TweetContent' column
G20data['matches'] = G20data['TweetContent'].str.contains(regex_pattern, case=False, na=False)
filtered_G20data = G20data[G20data['matches']]


# In[ ]:


# Set display options
pd.set_option('display.max_colwidth', None)  # or use a large integer if 'None' doesn't work
pd.set_option('display.max_rows', 500)  # Adjust based on how many rows you want to see

# Assuming your DataFrame is already loaded
print(filtered_G20data.head())  


# In[ ]:


print(filtered_G20data['Tweet Language'].value_counts())

print(G20data['Tweet Language'].value_counts())


# In[ ]:


filtered_G20data.to_csv('G20_filtered_beforeT.csv', index=False)


# In[ ]:


# Load the TSV file
G20_filtered_beforeT = pd.read_csv('G20_filtered_beforeT.csv')


# Display the first few rows of the DataFrame
print(G20_filtered_beforeT.head())

# Optionally, check the structure of your DataFrame
print(G20_filtered_beforeT.info())


# In[ ]:


# Count occurrences of each keyword
keyword_counts = {keyword: 0 for keyword in keywords}

for keyword in keywords:
    keyword_regex = enhance_keyword_regex(keyword)
    pattern = re.compile(r'\b' + keyword_regex + r'\b', re.IGNORECASE)
    keyword_counts[keyword] = filtered_G20data['TweetContent'].apply(lambda x: bool(pattern.search(str(x)))).sum()

# Convert the keyword counts to a DataFrame for better visualization
keyword_counts_df = pd.DataFrame(keyword_counts.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)

# Save the counts to a CSV file
keyword_counts_df.to_csv('keyword_counts.csv', index=False)

# Print the keyword counts
print(keyword_counts_df)


# ## 3.2. Translating G20 dataset for Phase 1

# In[ ]:


# Assuming there's a column named 'language' that specifies the language of each tweet
G20data_filtered_Ger = filtered_G20data[filtered_G20data['Tweet Language'] == 'de']


# In[ ]:


# Create a copy to avoid modifying the original DataFrame
G20data_filtered_Ger = G20data_filtered_Ger.copy()


# In[ ]:


# Apply translation and handle large dataset in chunks if necessary
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


# Drop the temporary 'TweetContent_translated' column as it's no longer needed
merged_dataset.drop(columns=['translated_text'], inplace=True)

# Save the merged dataset back to a CSV or continue processing
merged_dataset.to_csv('G20_filtered_after_translation1.csv', index=False)


# In[ ]:


print("Original G20_filtered_beforeT data:")
print(G20_filtered_beforeT.head())


# In[ ]:


# Load the TSV file
merged_dataset2 = pd.read_csv('G20_filtered_after_translation1.csv')


# Display the first few rows of the DataFrame
print(merged_dataset2.head())

# Optionally, check the structure of your DataFrame
print(merged_dataset2.info())


# In[ ]:


# Verify the changes
print(G20translate.columns)

