#!/usr/bin/env python
# coding: utf-8

# # 1. Filtering G20 Dataset 

# In[ ]:


import pandas as pd
G20data = pd.read_csv('G20Hamburg.csv')
print(G20data.info())
print(G20data.describe())


# ## 1.1. Filtering male leaders

# In[1]:


# Rename columns
G20data = G20data.rename(columns={
    'Tweet': 'TweetNumber',
    'Tweet.1': 'TweetContent',
})

# Verify the changes
print(G20data.columns)


# In[ ]:


# Filter out retweets
G20data = G20data[G20data['Is Retweet'] == False]
# Remove duplicates based on tweet content
G20data = G20data.drop_duplicates(subset=['TweetContent'])
# Drop rows where the tweet content is missing
G20data = G20data.dropna(subset=['TweetContent'])


# In[ ]:


print(G20data.info())
print(G20data.describe())


# In[ ]:


import pandas as pd
import re

# Male leaders
leaders_info = {
    "Mauricio Macri": ["Argentina President", "Argentinan President", "President Argentina", "President of Argentina", 
                       "Präsident von Argentinien", "Argentinischer Präsident"],
    "Malcolm Turnbull": ["Prime Minister Australia", "Australian Prime Minister", "Australia Prime Minister", "Australian PM", "Prime Minister of Australia",
                         "Australischer Ministerpräsident", "Ministerpräsident von Australien"],
    "Michel Temer": ["President Brazil", "Brazil President", "Brazilian President", "President Brazil",
                     "Präsident von Brasilien", "Brasilianischer Präsident"],
    "Justin Trudeau": ["Prime Minister Canada", "canadian Prime Minister", "Canada Prime Minister", "canadian PM", "Prime Minister of Canada",
                       "Kanadischer Ministerpräsident", "Ministerpräsident von Kanada"],
    "Xi Jinping": ["President China", "Chinese President", "China President", "President Chinese", "President of China",
                   "Präsident von China", "Chinesischer Präsident"],
    "Emmanuel Macron": ["President France", "French President", "France President", "President French", "President of France",
                        "Präsident von Frankreich", "Französischer Präsident"],
    "Narendra Modi": ["Prime Minister India", "Indian Prime Minister", "India Prime Minister", "Indian PM", "Prime Minister of India",
                      "Indischer Ministerpräsident", "Ministerpräsident von Indien"],
    "Joko Widodo": ["President Indonesia", "Indonesian President", "Indonesia President", "President Indonesian", "President of Indonesia",
                    "Präsident von Indonesien", "Indonesischer Präsident"],
    "Paolo Gentiloni": ["Prime Minister Italy", "Italian Prime Minister", "Italy Prime Minister", "Italian PM", "Prime Minister of Italy",
                        "Italienischer Ministerpräsident", "Ministerpräsident von Italien"],
    "Shinzō Abe": ["Prime Minister Japan", "Japanese Prime Minister", "Japan Prime Minister", "Japanese PM", "Prime Minister of Japan",
                   "Japanischer Ministerpräsident", "Ministerpräsident von Japan"],
    "Enrique Peña Nieto": ["President Mexico", "Mexican President", "Mexico President", "President Mexican", "President of Mexico",
                           "Präsident von Mexiko", "Mexikanischer Präsident"],
    "Vladimir Putin": ["President Russia", "Russian President", "Russia President", "President Russian", "President of Russia",
                       "Russischer Präsident", "Präsident von Russland"],
    "Ibrahim Abdulaziz Al-Assaf": ["State Minister Saudi Arabia", "Saudi Arabia State Minister", "State Minister of Saudi Arabia",
                                   "Staatsminister von Saudi-Arabien"],
    "Moon Jae-in": ["President South Korea", "South Korean President", "South Korea President", "President South Korean", "President of South Korea",
                    "Präsident von Südkorea", "Südkoreanischer Präsident"],
    "Jacob Zuma": ["President South African", "South Africa President", "South Africa President", "President South African", "President of South Africa",
                   "Präsident von Südafrika", "Südafrikanischer Präsident"],
    "Recep Tayyip Erdoğan": ["President Turkey", "Turkish President", "Turkey President", "President Turkish", "President of Turkey",
                             "Türkischer Präsident", "Präsident von der Türkei"],
    "Donald Trump": ["US President", "President USA", "USA President", "US President", "President of America", "President of USA",
                     "US-Präsident", "Präsident der USA"],
    "Mark Rutte": ["Prime Minister Netherlands", "Dutch Prime Minister", "Dutch PM", "Prime Minister of Netherlands",
                   "Niederländischer Ministerpräsident", "Ministerpräsident der Niederlande"],
    "Macky Sall": ["President Senegal", "Senegalese President", "Senegal President", "President Senegalese", "President of Senegal",
                   "Präsident von Senegal", "Senegalesischer Präsident"],
    "Mariano Rajoy": ["Prime Minister Spain", "Spanish Prime Minister", "Spain Prime Minister", "Spanish PM", "Prime Minister of Spain",
                      "Spanischer Ministerpräsident", "Ministerpräsident von Spanien"],
    "Lee Hsien Loong": ["Prime Minister Singapore", "Singaporean Prime Minister", "Singapore Prime Minister", "Singaporean PM", "Prime Minister of Singapore",
                        "Singapurischer Ministerpräsident", "Ministerpräsident von Singapur"],
    "Nguyễn Xuân Phúc": ["Prime Minister Vietnam", "Vietnamese Prime Minister", "Vietnam Prime Minister", "Vietnamese PM", "Prime Minister of Vietnam",
                         "Vietnamesischer Ministerpräsident", "Ministerpräsident von Vietnam"]
}

# Expand the list to include common variations and titles
expanded_leaders = []
for leader, titles in leaders_info.items():
    first_name, *rest = leader.split()
    last_name = rest[-1] if rest else ''
    expanded_leaders.extend([
        leader,                      
        f"{first_name}'s",           
        f"{last_name}'s",            
        first_name,                 
        last_name                    
    ] + titles)

# Create a regex pattern to match any leader name or title
pattern = '|'.join(re.escape(leader) for leader in set(expanded_leaders))

# Filter tweets that mention any of the leaders
G20data['mentions_leader'] = G20data['TweetContent'].apply(lambda x: bool(re.search(pattern, x, flags=re.IGNORECASE)))

filtered_tweets = G20data[G20data['mentions_leader']]
filtered_tweets.to_csv('G20data_with_leader_mentions.csv', index=False)

print(filtered_tweets.head())


# ## 1.2. Filtering female leaders 

# In[ ]:


# Female leaders
FemaleLeaders = [
    "Ivanka Trump", "Ivanka", "Trump’s daughter", "Trump daughter" "Theresa May", "Theresa", "Theresa's", "Theresa May's", "Theresa May's", "UK Prime Minister", "UK PM",
    "Erna Solberg", "Erna Solberg's", "Erna's", "Solberg's", "Erna", "Solberg", "Norwegian prime minister", "Norwegian PM", "Angela Merkel", "Angela", "Merkel", "Angela Merkel's", "Merkel's", "Angela's",
    "Chancellor of Germany", "Chancellor", "Trump's Tochter", "Trump Tochter" "Premierminister des Vereinigten Königreichs", "UK Premierminister",
    "Norwegischer Premierminister", "Bundeskanzlerin von Deutschland", "Bundeskanzlerin Deutschlands",
    "Bundeskanzler", 
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
    for german_char, replacement in replacements.items():
        keyword = re.sub(re.escape(german_char), replacement, keyword, flags=re.IGNORECASE)
    return keyword

def enhance_keyword_regex(keyword):
    keyword = adjust_umlauts_and_specials(keyword)
    
    keyword = re.escape(keyword)
    
    keyword = keyword.replace(r'\ ', r'\s*')
    
    # Make certain punctuations optional
    punctuations = ["'", "-", "."]  
    for punct in punctuations:
        keyword = keyword.replace(re.escape(punct), re.escape(punct) + '?')
    
    return keyword

regex_pattern = r'\b(' + '|'.join(enhance_keyword_regex(keyword) for keyword in FemaleLeaders) + r')\b'


# In[ ]:


G20data['matches'] = G20data['TweetContent'].str.contains(regex_pattern, case=False, na=False)
filtered_G20data = G20data[G20data['matches']]

filtered_G20data.to_csv('G20_filtered_female_leaders.csv', index=False)

print(filtered_G20data[['TweetContent']].head())


# ## 2. Translating G20 male leaders dataset

# In[ ]:


Mpoliticians = pd.read_csv('G20data_with_leader_mentions.csv')

print(Mpoliticians.head())

print(Mpoliticians.info())
print(Mpoliticians['Tweet Language'].value_counts())


# In[ ]:


Mpoliticians_Ger = Mpoliticians[Mpoliticians['Tweet Language'] == 'de']


# In[ ]:


Mpoliticians_Ger = Mpoliticians_Ger.copy()


# In[ ]:


import time
from googletrans import Translator, constants

translator = Translator()

def translate_text(text, lang='en', max_retries=5):
    """Attempt to translate text with retries on failure."""
    for attempt in range(max_retries):
        try:
            translation = translator.translate(text, dest=lang)
            return translation.text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(2)  
            if attempt == max_retries - 1:
                return text  
def safe_translate(text):
    """A safe translation function that defaults to the original text on failure."""
    return translate_text(text, 'en')

import pandas as pd
Mpoliticians = pd.read_csv('G20data_with_leader_mentions.csv')
Mpoliticians_Ger = Mpoliticians[Mpoliticians['Tweet Language'] == 'de'].copy()

Mpoliticians_Ger['translated_text'] = Mpoliticians_Ger['TweetContent'].apply(safe_translate)

print(Mpoliticians_Ger[['TweetContent', 'translated_text']].head())


# In[ ]:


print(Mpoliticians_Ger[['TweetContent', 'translated_text']].head())
print(Mpoliticians_Ger[['TweetContent', 'translated_text']].info())


# In[ ]:


Mpoliticians_Ger.to_csv('MPoliticianTranslation.csv', index=False)


# In[ ]:


import pandas as pd

Mp_translation = pd.read_csv('MPoliticianTranslation.csv')


# In[ ]:


print(Mp_translation.head())


# ### 2.1. Merge translated tweets back with Engslish tweets

# In[ ]:


import pandas as pd

merged_data = pd.merge(Mpoliticians, Mp_translation[['UniqueID', 'Sheet#', 'TweetNumber', 'translated_text']], on=['UniqueID', 'Sheet#', 'TweetNumber'], how='left')

merged_data['TweetContent'] = merged_data['translated_text'].fillna(merged_data['TweetContent'])

merged_data.drop(columns=['translated_text'], inplace=True)

print(merged_data.head())



# In[ ]:


merged_data.to_csv('MalePoliticians.csv', index=False)

MalePoliticians = pd.read_csv('MalePoliticians.csv')

print(MalePoliticians.head())


# # 3. Preprocessing G20 male leaders dataset

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


MalePoliticians = pd.read_csv('MalePoliticians.csv')

MalePoliticians['cleaned_text'] = MalePoliticians['TweetContent'].apply(preprocess_text)
MalePoliticians['lemmatized_text'] = MalePoliticians['cleaned_text'].apply(lemmatize_text)


print(MalePoliticians[['TweetContent', 'cleaned_text', 'lemmatized_text']].head())


# In[ ]:


MalePoliticians.to_csv('MalePoliticiansCLEAN.csv', index=False)


# In[ ]:


MPPP = pd.read_csv('MalePoliticiansCLEAN.csv')
print(MPPP.head())


# In[ ]:


# Dictionary of male leaders and associated titles
leaders_info = {
    "Mauricio Macri": ["Argentina President", "Argentinan President", "President Argentina", "President of Argentina", 
                       "Präsident von Argentinien", "Argentinischer Präsident"],
    "Malcolm Turnbull": ["Prime Minister Australia", "Australian Prime Minister", "Australia Prime Minister", "Australian PM", "Prime Minister of Australia",
                         "Australischer Ministerpräsident", "Ministerpräsident von Australien"],
    "Michel Temer": ["President Brazil", "Brazil President", "Brazilian President", "President Brazil",
                     "Präsident von Brasilien", "Brasilianischer Präsident"],
    "Justin Trudeau": ["Prime Minister Canada", "canadian Prime Minister", "Canada Prime Minister", "canadian PM", "Prime Minister of Canada",
                       "Kanadischer Ministerpräsident", "Ministerpräsident von Kanada"],
    "Xi Jinping": ["President China", "Chinese President", "China President", "President Chinese", "President of China",
                   "Präsident von China", "Chinesischer Präsident"],
    "Emmanuel Macron": ["President France", "French President", "France President", "President French", "President of France",
                        "Präsident von Frankreich", "Französischer Präsident"],
    "Narendra Modi": ["Prime Minister India", "Indian Prime Minister", "India Prime Minister", "Indian PM", "Prime Minister of India",
                      "Indischer Ministerpräsident", "Ministerpräsident von Indien"],
    "Joko Widodo": ["President Indonesia", "Indonesian President", "Indonesia President", "President Indonesian", "President of Indonesia",
                    "Präsident von Indonesien", "Indonesischer Präsident"],
    "Paolo Gentiloni": ["Prime Minister Italy", "Italian Prime Minister", "Italy Prime Minister", "Italian PM", "Prime Minister of Italy",
                        "Italienischer Ministerpräsident", "Ministerpräsident von Italien"],
    "Shinzō Abe": ["Prime Minister Japan", "Japanese Prime Minister", "Japan Prime Minister", "Japanese PM", "Prime Minister of Japan",
                   "Japanischer Ministerpräsident", "Ministerpräsident von Japan"],
    "Enrique Peña Nieto": ["President Mexico", "Mexican President", "Mexico President", "President Mexican", "President of Mexico",
                           "Präsident von Mexiko", "Mexikanischer Präsident"],
    "Vladimir Putin": ["President Russia", "Russian President", "Russia President", "President Russian", "President of Russia",
                       "Russischer Präsident", "Präsident von Russland"],
    "Ibrahim Abdulaziz Al-Assaf": ["State Minister Saudi Arabia", "Saudi Arabia State Minister", "State Minister of Saudi Arabia",
                                   "Staatsminister von Saudi-Arabien"],
    "Moon Jae-in": ["President South Korea", "South Korean President", "South Korea President", "President South Korean", "President of South Korea",
                    "Präsident von Südkorea", "Südkoreanischer Präsident"],
    "Jacob Zuma": ["President South African", "South Africa President", "South Africa President", "President South African", "President of South Africa",
                   "Präsident von Südafrika", "Südafrikanischer Präsident"],
    "Recep Tayyip Erdoğan": ["President Turkey", "Turkish President", "Turkey President", "President Turkish", "President of Turkey",
                             "Türkischer Präsident", "Präsident von der Türkei"],
    "Donald Trump": ["US President", "President USA", "USA President", "US President", "President of America", "President of USA",
                     "US-Präsident", "Präsident der USA"],
    "Mark Rutte": ["Prime Minister Netherlands", "Dutch Prime Minister", "Dutch PM", "Prime Minister of Netherlands",
                   "Niederländischer Ministerpräsident", "Ministerpräsident der Niederlande"],
    "Macky Sall": ["President Senegal", "Senegalese President", "Senegal President", "President Senegalese", "President of Senegal",
                   "Präsident von Senegal", "Senegalesischer Präsident"],
    "Mariano Rajoy": ["Prime Minister Spain", "Spanish Prime Minister", "Spain Prime Minister", "Spanish PM", "Prime Minister of Spain",
                      "Spanischer Ministerpräsident", "Ministerpräsident von Spanien"],
    "Lee Hsien Loong": ["Prime Minister Singapore", "Singaporean Prime Minister", "Singapore Prime Minister", "Singaporean PM", "Prime Minister of Singapore",
                        "Singapurischer Ministerpräsident", "Ministerpräsident von Singapur"],
    "Nguyễn Xuân Phúc": ["Prime Minister Vietnam", "Vietnamese Prime Minister", "Vietnam Prime Minister", "Vietnamese PM", "Prime Minister of Vietnam",
                         "Vietnamesischer Ministerpräsident", "Ministerpräsident von Vietnam"]
}

for leader, titles in leaders_info.items():
    first_name, *rest = leader.split()
    last_name = rest[-1] if rest else ''
    variations = [leader, f"{first_name}'s", f"{last_name}'s", first_name, last_name] + titles
    pattern = '|'.join(re.escape(term) for term in variations)
    regex = re.compile(pattern, re.IGNORECASE)
    
    MPPP[leader] = MPPP['UniqueContent'].apply(lambda x: bool(regex.search(x)))

# Calculate the number of tweets mentioning each leader
leader_counts = {leader: MPPP[leader].sum() for leader in leaders_info}
print(leader_counts)


# # 4. Sentiment Analysis for male leaders

# In[ ]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Function to classify sentiment based on compound score
def classify_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'

MPPP['sentiment'] = MPPP['lemmatized_text'].apply(classify_sentiment)

print(MPPP[['lemmatized_text', 'sentiment']])


# In[ ]:


# Count sentiments by male leader

leaders_list = [
    "Mauricio Macri", "Malcolm Turnbull", "Michel Temer", "Justin Trudeau",
    "Xi Jinping", "Emmanuel Macron", "Narendra Modi", "Joko Widodo",
    "Paolo Gentiloni", "Shinzō Abe", "Enrique Peña Nieto", "Vladimir Putin",
    "Ibrahim Abdulaziz Al-Assaf", "Moon Jae-in", "Jacob Zuma",
    "Recep Tayyip Erdoğan", "Donald Trump", "Mark Rutte", "Macky Sall",
    "Mariano Rajoy", "Lee Hsien Loong", "Nguyễn Xuân Phúc"
]

sentiment_data = []

for leader in leaders_list:
    filtered_mpp = MPPP[MPPP[leader] == True]
    positive_count = filtered_mpp['sentiment'].value_counts().get('positive', 0)
    neutral_count = filtered_mpp['sentiment'].value_counts().get('neutral', 0)
    negative_count = filtered_mpp['sentiment'].value_counts().get('negative', 0)
    
    sentiment_data.append({
        'Leader': leader,
        'Positive': positive_count,
        'Neutral': neutral_count,
        'Negative': negative_count
    })

sentiment_counts = pd.DataFrame(sentiment_data)

print(sentiment_counts)


# In[ ]:


print(MPPP.info())


# # 5. Sentiment Analysis for female leaders

# In[ ]:


FPPP = pd.read_csv('G20_CAT.csv')


# In[ ]:


# Apply sentiment analysis on female leaders
FPPP['sentiment'] = FPPP['lemmatized_text'].apply(classify_sentiment)

print(FPPP[['lemmatized_text', 'sentiment']].head())


# In[ ]:


print(FPPP.head())
print(FPPP.info())


# # 6. Merge male and female leaders datasets

# In[ ]:


import pandas as pd

merged_data = pd.merge(MPPP, FPPP, on='UniqueID', how='outer', suffixes=('_MPPP', '_FPPP'))

print(merged_data.head())

print(merged_data.info())


# In[ ]:


merged_data.to_csv('4sentimentAnalysis.csv', index=False)


# In[ ]:


APPP = pd.read_csv('4sentimentAnalysis.csv')
print(APPP.info())


# # 7. Compare sentiments between male and female leaders

# In[ ]:


import pandas as pd

# Define leader categories by gender
male_leaders = [
    "Mauricio Macri", "Malcolm Turnbull", "Michel Temer", "Justin Trudeau",
    "Xi Jinping", "Emmanuel Macron", "Narendra Modi", "Joko Widodo",
    "Paolo Gentiloni", "Shinzō Abe", "Enrique Peña Nieto", "Vladimir Putin",
    "Ibrahim Abdulaziz Al-Assaf", "Moon Jae-in", "Jacob Zuma",
    "Recep Tayyip Erdoğan", "Donald Trump", "Mark Rutte", "Macky Sall",
    "Mariano Rajoy", "Lee Hsien Loong", "Nguyễn Xuân Phúc"
]

female_leaders = ["angela_merkel", "theresa_may", "erna_solberg"]


# In[ ]:


# Identify non-overlapping tweets
unique_mppp = APPP[APPP['sentiment_FPPP'].isna() & APPP['sentiment_MPPP'].notna()]

unique_fppp = APPP[APPP['sentiment_MPPP'].isna() & APPP['sentiment_FPPP'].notna()]

# Function to count sentiments
def count_sentiments(df, sentiment_column):
    # Initialize sentiment counts
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    # Count each type of sentiment
    if sentiment_column in df.columns:
        for key in sentiments.keys():
            sentiments[key] = df[df[sentiment_column] == key].shape[0]
    return sentiments

# Count sentiments for MPPP and FPPP
mppp_sentiments = count_sentiments(unique_mppp, 'sentiment_MPPP')
fppp_sentiments = count_sentiments(unique_fppp, 'sentiment_FPPP')

print("Unique MPPP Sentiments:", mppp_sentiments)
print("Unique FPPP Sentiments:", fppp_sentiments)


# In[ ]:


# Define a function to calculate percentages from sentiment counts
def calculate_percentages(sentiment_counts):
    total = sum(sentiment_counts.values())
    if total == 0:
        return {sentiment: 0 for sentiment in sentiment_counts} 
    percentages = {sentiment: (count / total * 100) for sentiment, count in sentiment_counts.items()}
    return percentages

mppp_percentages = calculate_percentages(mppp_sentiments)  
fppp_percentages = calculate_percentages(fppp_sentiments)  

print("Percentages of Sentiments for MPPP Tweets:")
for sentiment, percentage in mppp_percentages.items():
    print(f"{sentiment.capitalize()}: {percentage:.2f}%")

print("\nPercentages of Sentiments for FPPP Tweets:")
for sentiment, percentage in fppp_percentages.items():
    print(f"{sentiment.capitalize()}: {percentage:.2f}%")


# In[ ]:


import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Sentiment counts
mppp_sentiments = {'positive': 10158, 'neutral': 10878, 'negative': 9950}
fppp_sentiments = {'positive': 1444, 'neutral': 1202, 'negative': 1354}

# Total tweets in each category
mppp_total = sum(mppp_sentiments.values())
fppp_total = sum(fppp_sentiments.values())

# Lists to hold results
sentiments = ['positive', 'neutral', 'negative']
results = []

# Perform z-tests for each sentiment category
for sentiment in sentiments:
    count = np.array([mppp_sentiments[sentiment], fppp_sentiments[sentiment]])
    nobs = np.array([mppp_total, fppp_total])
    stat, pval = proportions_ztest(count, nobs)
    results.append((sentiment, stat, pval))

for result in results:
    sentiment, stat, pval = result
    print(f"{sentiment.capitalize()} Sentiment:")
    print(f"Z-statistic: {stat:.2f}")
    print(f"P-value: {pval:.4f}")
    if pval < 0.05:
        print(f"The difference in {sentiment} sentiment between MPPP and FPPP is statistically significant.")
    else:
        print(f"No significant difference in {sentiment} sentiment between MPPP and FPPP.")
    print()


# # 8. Descriptive Statistics and Visualisations

# In[ ]:


leader_columns = ['angela_merkel', 'theresa_may', 'erna_solberg']

tweets_mentioning_any_leader = APPP[APPP[leader_columns].any(axis=1)]

num_tweets_mentioning_any = tweets_mentioning_any_leader.shape[0]

print(f"Number of tweets mentioning any of Angela Merkel, Theresa May, or Erna Solberg: {num_tweets_mentioning_any}")


# In[ ]:


tweets_mentioning_leaders = APPP[APPP[leader_columns].any(axis=1)]
tweets_by_language = tweets_mentioning_leaders.groupby('Tweet Language_FPPP').size()

print("Number of tweets mentioning any of Angela Merkel, Theresa May, or Erna Solberg, by language:")
print(tweets_by_language)


# In[ ]:


# Count individual mentions within the filtered tweets
angela_mentions = tweets_mentioning_any_leader['angela_merkel'].sum()
theresa_mentions = tweets_mentioning_any_leader['theresa_may'].sum()
erna_mentions = tweets_mentioning_any_leader['erna_solberg'].sum()

print(f"Angela Merkel was mentioned in {angela_mentions} tweets.")
print(f"Theresa May was mentioned in {theresa_mentions} tweets.")
print(f"Erna Solberg was mentioned in {erna_mentions} tweets.")


# In[ ]:


# Sentiment distribution
sentiment_distribution = unique_mppp['sentiment_MPPP'].value_counts()

print("\nSentiment Distribution in MPPP Tweets:")
print(sentiment_distribution)


# In[ ]:


unique_mppp.to_csv('unique_mppp.csv', index=False)
unique_mppp2 = pd.read_csv('unique_mppp.csv') 


# In[ ]:


columns_to_check = [
    "Mauricio Macri", "Malcolm Turnbull", "Michel Temer", "Justin Trudeau",
    "Xi Jinping", "Emmanuel Macron", "Narendra Modi", "Joko Widodo",
    "Paolo Gentiloni", "Shinzō Abe", "Enrique Peña Nieto", "Vladimir Putin",
    "Ibrahim Abdulaziz Al-Assaf", "Moon Jae-in", "Jacob Zuma",
    "Recep Tayyip Erdoğan", "Donald Trump", "Mark Rutte", "Macky Sall",
    "Mariano Rajoy", "Lee Hsien Loong", "Nguyễn Xuân Phúc"
]

true_counts = unique_mppp2[columns_to_check].sum()

for column, count in true_counts.items():
    print(f"There are {count} TRUE {column}")


# In[ ]:


import pandas as pd

leaders = [
    "Mauricio Macri", "Malcolm Turnbull", "Michel Temer", "Justin Trudeau",
    "Xi Jinping", "Emmanuel Macron", "Narendra Modi", "Joko Widodo",
    "Paolo Gentiloni", "Shinzō Abe", "Enrique Peña Nieto", "Vladimir Putin",
    "Ibrahim Abdulaziz Al-Assaf", "Moon Jae-in", "Jacob Zuma",
    "Recep Tayyip Erdoğan", "Donald Trump", "Mark Rutte", "Macky Sall",
    "Mariano Rajoy", "Lee Hsien Loong", "Nguyễn Xuân Phúc"
]
counts = [640, 535, 459, 825, 2214, 1153, 761, 422, 380, 6895, 64, 7036, 7, 421, 89, 98, 17093, 1226, 54, 12, 463, 0]

df = pd.DataFrame({
    'Leader': leaders,
    'Count': counts
})

print(df)


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

leaders = [
    "Mauricio Macri", "Malcolm Turnbull", "Michel Temer", "Justin Trudeau",
    "Xi Jinping", "Emmanuel Macron", "Narendra Modi", "Joko Widodo",
    "Paolo Gentiloni", "Shinzō Abe", "Enrique Peña Nieto", "Vladimir Putin",
    "Ibrahim Abdulaziz Al-Assaf", "Moon Jae-in", "Jacob Zuma",
    "Recep Tayyip Erdoğan", "Donald Trump", "Mark Rutte", "Macky Sall",
    "Mariano Rajoy", "Lee Hsien Loong", "Nguyễn Xuân Phúc", "Angela Merkel", "Theresa May", "Erna Solberg"
]
counts = [640, 535, 459, 825, 2214, 1153, 761, 422, 380, 6895, 64, 7036, 7, 421, 89, 98, 17093, 1226, 54, 12, 463, 0, 4780, 391, 11]

df = pd.DataFrame({
    'Leader': leaders,
    'Count': counts
})

df_sorted = df.sort_values('Count', ascending=False)

plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
bar_plot = sns.barplot(
    y='Leader', 
    x='Count', 
    data=df_sorted, 
    palette='coolwarm'
)

for p in bar_plot.patches:
    width = p.get_width()    
    plt.text(5 + width,     
             p.get_y() + p.get_height() / 2, 
             '{:1.0f}'.format(width), 
             va='center')  

plt.title('Mentions of Leaders in Tweets')
plt.xlabel('Number of Mentions')
plt.ylabel('Leaders')

plt.show()

