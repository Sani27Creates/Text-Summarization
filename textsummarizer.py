import re
import streamlit as st

# NLTK Packages
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# SPACY Packages
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Import heapq for selecting top N elements
import heapq

# Download NLTK data if not already done
nltk.download('stopwords')
nltk.download('punkt')

# Function for NLTK summarization
def nltk_summarizer(docx, num_lines):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(docx)
    freqTable = dict()

    for word in words:
        word = word.lower()
        if word not in stopWords:
            freqTable[word] = freqTable.get(word, 0) + 1

    sentence_list = sent_tokenize(docx)
    max_freq = max(freqTable.values())
    for word in freqTable.keys():
        freqTable[word] = freqTable[word] / max_freq

    sentence_scores = {}
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in freqTable:
                if len(sent.split(' ')) < 30:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + freqTable[word]

    summary_sentences = heapq.nlargest(num_lines, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Function for SPACY summarization
def spacy_summarizer(docx, num_lines):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(docx)
    stopWords = list(STOP_WORDS)
    words = word_tokenize(docx.text)
    freqTable = dict()

    for word in words:
        word = word.lower()
        if word not in stopWords:
            freqTable[word] = freqTable.get(word, 0) + 1

    sentence_list = sent_tokenize(docx.text)
    max_freq = max(freqTable.values())
    for word in freqTable.keys():
        freqTable[word] = freqTable[word] / max_freq

    sentence_scores = {}
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in freqTable:
                if len(sent.split(' ')) < 30:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + freqTable[word]

    summary_sentences = heapq.nlargest(num_lines, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

def main():
    st.title("Text Summarizer App")
    st.subheader("Summary using NLP")
    article_text = st.text_area("Enter Text Here", "Type here")
    
    # Cleaning of input text
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub('[^a-zA-Z.,]', ' ', article_text)
    article_text = re.sub(r'\b[a-zA-Z]\b', '', article_text)
    article_text = re.sub('[A-Z]\Z', '', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    num_lines = st.number_input("Number of lines for summary", min_value=1, max_value=20, value=8)

    if st.button("Summarize"):
        # Use NLTK summarization directly
        summary_result = nltk_summarizer(article_text, num_lines)
        st.write(summary_result)

if __name__ == '__main__':
    main()