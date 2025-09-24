import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
df = pd.read_csv("data/metadata_cleaned.csv")

st.title("CORD-19 Data Explorer")
st.write("Explore COVID-19 research metadata interactively.")

# Select year range
year_min = int(df['year'].min())
year_max = int(df['year'].max())
year_range = st.slider("Select Year Range", year_min, year_max, (year_min, year_max))

df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

st.subheader(f"Showing data for {year_range[0]} - {year_range[1]}")
st.write(df_filtered.head())

# Publications by year
year_counts = df_filtered['year'].value_counts().sort_index()
fig1, ax1 = plt.subplots()
ax1.bar(year_counts.index, year_counts.values)
ax1.set_title("Publications by Year")
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Papers")
st.pyplot(fig1)

# Top Journals
top_journals = df_filtered['journal'].value_counts().head(10)
fig2, ax2 = plt.subplots()
sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax2)
ax2.set_title("Top 10 Journals")
ax2.set_xlabel("Number of Papers")
ax2.set_ylabel("Journal")
st.pyplot(fig2)

# Word Cloud
text = " ".join(title for title in df_filtered['title'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis("off")
ax3.set_title("Word Cloud of Titles")
st.pyplot(fig3)
