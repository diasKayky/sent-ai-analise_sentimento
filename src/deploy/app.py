import streamlit as st
import pickle as pl
import tensorflow.keras.backend as K
import numpy as np
import time
from libretranslatepy import LibreTranslateAPI
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train_deploy = pl.load(open('C:/Users/Kayky/Downloads/Python/Archive/Sentiment_Analysis/data/processed/X_train_deploy.pkl', 'rb'))
modelo = pl.load(open('C:/Users/Kayky/Downloads/Python/Archive/Sentiment_Analysis/src/model/modelo.pkl', 'rb'))
#modelo = tf.keras.models.load_model('C:/Users/Kayky/Downloads/Python/Archive/Sentiment_Analysis/src/model/modelo.h5')

def prep_dados(texto: str):
    # Limpa o texto (remove alfanuméricos, simbolos, etc.)
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    texto = texto.lower()

    # Remove palavras desnecessárias
    stopwords_ = set(stopwords.words('english'))
    ma = lambda x: ' '.join([word for word in x.split() if word not in (stopwords_)])
    dados_prep = ma(texto)

    # Lematiza o texto (pega somente a raiz das palavras)
    tokenizador = nltk.tokenize.WhitespaceTokenizer()
    lematizador = nltk.stem.WordNetLemmatizer()

    st = ""

    for palavra in tokenizador.tokenize(dados_prep):
        st = st + lematizador.lemmatize(palavra) + " "

    # Tranforma em array do numpy
    dados_prep = np.array([dados_prep])

    # Tokeniza

    oov_tok = "<OOV>"
    tokenizador = Tokenizer(oov_token=oov_tok)
    tokenizador.fit_on_texts(X_train_deploy)
    dados_prep = tokenizador.texts_to_sequences(dados_prep)

    # Padding para adequar o tamanho da sequência
    tamanho_frase = 400
    dados_prep = pad_sequences(dados_prep, maxlen=tamanho_frase, padding='post', truncating='post')

    return dados_prep


st.title("Sent.IA")
st.markdown("Sent.AI é um modelo de Machine Learning que utiliza rede neural LSTM (memória de curto longo prazo) para analisar sentimento de textos e classificar em 'positivo' ou 'negativo'.")

st.subheader("Texto para Analisar")
texto = st.text_area(label="Digite:")

if st.button("Enviar para predição"):

    t = "Coletando dados e calculando predição..."
    barra = st.progress(0, text=t)

    for porcento in range(100):
        time.sleep(0.2)
        barra.progress(porcento + 1, text=t)

    lt = LibreTranslateAPI("https://translate.argosopentech.com/")

    transl = lt.translate(texto, "pt", "en")

    dados = prep_dados(texto)

    predicted = modelo.predict(dados)

    ma = lambda x: 1 if x > 0.5 else 0

    predicted = ma(predicted)

    if predicted == 1:
        predicted = "positivo"
    else:
        predicted = "negativo"

    st.subheader("Resultado")
    st.write(f"A análise de sentimento revelou que o tom do seu texto é {predicted}.")

