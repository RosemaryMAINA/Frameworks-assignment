import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def clean_metadata(df):
    """Clean the metadata DataFrame: fix dates, drop duplicates, add new columns."""
    df = df.rename(columns=lambda c: c.strip())
    if 'publish_time' in df.columns:
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year

    df['abstract'] = df['abstract'].fillna('')
    df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))

    df['title'] = df['title'].fillna('')
    df['title_clean'] = df['title'].str.lower().fillna('').str.replace(r'[^a-z0-9\s]', '', regex=True)

    if 'title' in df.columns and 'authors' in df.columns:
        df = df.drop_duplicates(subset=['title', 'authors'])
    return df

def drop_many_missing(df, thresh=0.7):
    missing_frac = df.isnull().mean()
    to_drop = missing_frac[missing_frac > thresh].index.tolist()
    return df.drop(columns=to_drop), to_drop

def plot_publications_over_time(df, year_min=None, year_max=None):
    years = df['year'].dropna().astype(int)
    if year_min is not None and year_max is not None:
        years = years[(years >= year_min) & (years <= year_max)]
    counts = years.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of papers')
    ax.set_title('Publications by Year')
    plt.tight_layout()
    return fig

def plot_top_journals(df, top_n=10):
    journals = df['journal'].fillna('Unknown')
    counts = journals.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(counts.index[::-1], counts.values[::-1])
    ax.set_xlabel('Count')
    ax.set_title(f'Top {top_n} Journals')
    plt.tight_layout()
    return fig

def get_top_words_from_titles(df, top_n=30, stopwords=None):
    text = ' '.join(df['title_clean'].dropna().astype(str).tolist())
    words = re.findall(r'\w+', text)
    if not stopwords:
        stopwords = set(['the','and','to','of','in','a','for','on','with','by','from','is','an','et','al','covid','sars','coronavirus'])
    words = [w for w in words if w not in stopwords and len(w) > 2]
    c = Counter(words)
    return c.most_common(top_n)

def plot_top_words_bar(df, top_n=20):
    top_words = get_top_words_from_titles(df, top_n=top_n)
    if not top_words:
        return None
    words, counts = zip(*top_words)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(words[::-1], counts[::-1])
    ax.set_title('Top words in titles')
    plt.tight_layout()
    return fig

def make_wordcloud_from_titles(df, max_words=100):
    text = ' '.join(df['title'].dropna().astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words)
    img = wc.generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(img, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    return fig
