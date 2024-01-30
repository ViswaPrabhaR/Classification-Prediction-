import streamlit as st
import streamlit.components.v1 as v1
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from rake_nltk import Rake
import spacy
import requests

from bs4 import BeautifulSoup
nlp = spacy.load("en_core_web_sm")
pd.set_option("display.max_rows", 200)



def app():
    st.markdown(
        """
         <style>
    .stTextArea textarea {        
        background-color:#FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title('NLP Analysis')
    paragraph = "Linguistics is the scientific study of language. It encompasses the analysis of every aspect of language, as well as the methods for studying and modeling them. The traditional areas of linguistic analysis include phonetics, phonology, morphology, syntax, semantics, and pragmatics."
    st.text_area(":green[**Input Text**]",paragraph,key="placeholder",)
    with st.container(border=True):
        st.subheader("Text Pre-processing Techniques")
        st.markdown("--------------------------------------")
        col1, col2 = st.columns([2,4])
        with col1:
            LowerCaseText = st.checkbox('Lower Case')
            Tokenize = st.checkbox('Tokenize')
            Special_Character_Removal = st.checkbox('Special Character Removal')
            Stopword_Removal = st.checkbox('Stopword Removal')
            Stemming = st.checkbox('Stemming')
            Lemmatization = st.checkbox('Lemmatization')
            POS = st.checkbox('Ports Of Speech')
            N_Gram = st.checkbox('N Gram')
            keywords = st.checkbox('Keywords')
            NER = st.checkbox('Named Entity Recognition')
            word_cloud = st.checkbox('Word Cloud')



        with col2:
            words = nltk.word_tokenize(paragraph.lower())

            if LowerCaseText:
                lowercase_text = paragraph.lower()
                st.text_area(f":green[**Lower Case**]", lowercase_text, height=150)

            if Tokenize:
                radio = st.radio(':green[**Select Tokenize**]',
                                 options=['Word', 'Sentence', 'Character'],
                                 horizontal=True)

                if radio == 'Word':
                     # Tokenize Words
                    tokens = word_tokenize(paragraph)
                    st.text_area(f":green[**Tokenize of Words**]", tokens, height=150)
                    st.write(f":green[**No.of.Words:**] {len(words)}")
                if radio == 'Sentence':
                    sentences = nltk.sent_tokenize(paragraph)
                    st.text_area(f":green[**Tokenize of Sentence**]", sentences, height=150)
                    st.write(f":green[**No.of.Sentence:**] {len(sentences)}")
                if radio == 'Character':
                    characters = list(paragraph)
                    st.text_area(f":green[**Tokenize of Characters**]", characters, height=150)
                    st.write(f":green[**No.of.Characters:**] {len(characters)}")

            if Special_Character_Removal:
                # Special Character Removal
                Special_Character = [re.sub(r'[^a-zA-Z0-9]', '', i) for i in words]
                st.text_area(f":green[**Special Character Removal**]", Special_Character, height=150)

            if Stopword_Removal:
                # Stopword Removal
                new_words = [word for word in words if word.isalnum()]
                WordSet = []
                for word in new_words:
                    if word not in set(stopwords.words("english")):
                        WordSet.append(word)
                st.write(f":green[**Stop Word Removal**]", WordSet, height=150)

            if Stemming:
                stemmer = PorterStemmer()

                def stem_words(text):
                    word_tokens = text.split()
                    stems = [stemmer.stem(word) for word in word_tokens]
                    return stems
                stems = stem_words(paragraph)
                st.write(f":green[**Stemming**]", stems, height=150)

            if Lemmatization:
                #Lemmatization
                lemmatizer = WordNetLemmatizer()

                def lemmatize_word(text):
                    word_tokens = text.split()
                    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
                    return lemmas
                lemmatize = lemmatize_word(paragraph)
                st.write(f":green[**Lemmatization**]", lemmatize, height=150)

            if POS:
                tokens = nltk.word_tokenize(paragraph)
                pos_tags = nltk.pos_tag(tokens)
                st.text_area(f":green[**Port Of Speech**]", pos_tags, height=150)
                noun_list = []
                non_noun = []
                for i, j in pos_tags:
                    if j not in ['PRP', 'NN']:
                        non_noun.append(i)

                    else:
                        noun_list.append(i)
                st.write(f":green[**Noun List**]", noun_list, height=150)
                st.write(f":green[**Non Noun List**]", non_noun, height=150)

            if N_Gram:
                tokens = nltk.word_tokenize(paragraph)
                range_n = st.slider(':green[*Select Range*]', 0, 10, 2)
                gram_2 = list(ngrams(tokens,range_n))
                st.write(f":green[**N Gram**]", gram_2, height=150)

            if word_cloud:
                # Generate the word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(paragraph)
                # Display the word cloud
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.write(":green[**Word Cloud**]")
                st.pyplot(fig)

            if  keywords:

                nltk.download('stopwords')
                nltk.download('punkt')

                r = Rake()
                my_text = """
                When it comes to evaluating the performance of keyword extractors, you can use some of the standard metrics in machine learning: accuracy, precision, recall, and F1 score. However, these metrics don’t reflect partial matches; they only consider the perfect match between an extracted segment and the correct prediction for that tag.
                Fortunately, there are some other metrics capable of capturing partial matches. An example of this is ROUGE.
                """
                r.extract_keywords_from_text(paragraph)
                keywordList = []
                rankedList = r.get_ranked_phrases_with_scores()
                for keyword in rankedList:
                    keyword_updated = keyword[1].split()
                    keyword_updated_string = " ".join(keyword_updated[:2])
                    keywordList.append(keyword_updated_string)
                    if (len(keywordList) > 9):
                        break

                st.write(f":green[**Keyword List**]", keywordList, height=150)

            if NER:
                content = "Tesla last Monday announced it will revamp its top-selling Model Y electric car"
                st.text_area(":green[**Input Content**]", content )
                doc = nlp(content)
                for ent in doc.ents:
                    st.write(ent.text, ent.label_, height=150)
                    st.write()

