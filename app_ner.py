import streamlit as st
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import MultiTagger
from flair.visual.ner_html import render_ner_html
import flair
from streamlit import caching
from pathlib import Path
import torch
from helper import render_ner_html_custom
import pandas as pd 


colors = {
        "PRS": "#F7FF53", # YELLOW
        "PER": "#06b300", # GREEN
        "ORG": "#E8902E", # ORANGE (darker)
        "LOC": "#FF40A3", # PINK
        "MISC": "#4647EB", # PURPLE
        #"EVN": "#06b300", # GREEN
        #"MSR": "#FFEDD5", # ORANGE (lighter)
        #"TME": "#ff7398", # PINK (pig)
        #"WRK": "#c5ff73", # YELLOW (REFLEX)
        #"OBJ": "#4ed4b0", # TURQUOISE
        "O": "#ddd",      # GRAY
        }


colors_2 = {
        "Chemical": "#06b300", # GREEN
        "Disease": "#FFEDD5", # ORANGE (lighter)
        "Gene": "#ff7398", # PINK (pig)
        "Species": "#4ed4b0", # TURQUOISE
        "O": "#ddd",      # GRAY
        }




RANDOM_EMOJIS = list(
        "๐ฅโข๐๐๐๐ฃโจ๐๐๐๐ฅ๐คฉ๐ค๐๐คโฌ๐ก๐คช๐ฅโก๐จ๐ ๐๐ฟ๐๐ฎ๐ค๐๐๐พ๐ซโช๐ด๐๐ฌ๐๐๐โโฝ๐๐๐๐ธ๐จ๐ฅณโ๐๐ฑ๐๐ป๐๐๐ฆ๐๐ฆ๐๐๐ป๐๐คค๐ฏ๐ปโผ๐๐๐๐๐๐ซ๐๐ฝ๐ฌ๐โ๐ท๐ญโ๐๐๐๐ฅฐ๐๐๐ฅ๐บโ๐งกโ๐๐ปโ๐ธ๐ฌ๐ค๐นยฎโบ๐ช๐โ๐ค โ๐ค๐ต๐ค๐๐ฏ๐๐ป๐๐๐๐โฃ๐๐๐๐ค๐ฟ๐ฆ๐โ๐ฟ๐๐น๐ฅด๐ฝ๐๐ญ๐ค๐๐โช๐โ๐๐ญ๐ป๐ฅ๐๐คง๐๐๐๐ค๐๐๐โโฐ๐๐งโ โฅ๐ณ๐พ๐โญ๐๐ณ๐๐๐ธโค๐ช๐๐พโ๐๐๐ โ๐๐ต๐จ๐๐คซ๐คญ๐๐๐๐๐๐๐๐๐๐๐๐๐ฉ๐๐คทโ๐ธ๐๐ฎ๐๐ณ๐ฝ๐๐ฟ๐๐ ๐ผ๐ถ๐ค๐โ๐๐ต๐ค๐ฐ๐๐๐ฒ๐ฎ๐๐๐๐๐ต๐ฃโ๐บ๐๐๐๐ฅบ๐๐กโฆ๐๐ฑ๐๐โ๐พ๐ฉ๐ฅถ๐ฃ๐ผ๐คฃโฏ๐ต๐ซโก๐๐โ๐๐๐น๐๐ผ๐โซ๐๐ช๐จ๐ผ๐๐๐ณ๐๐๐๐ธ๐ง๐๐๐๐โ๐ป๐ด๐ผ๐ฟ๐โ ๐ฆโ๐คโฎ๐ข๐๐ค๐๐บ๐๐ด๐บโน๐ฒ๐๐ญ๐๐๐๐ต๐๐ด๐๐ง๐ฐโ๐๐คก๐ ๐ฒ๐๐๐ฌโ๐๐ฑ๐ฐ๐ฑ๐ง๐๐๐๐ฃ๐ซ๐๐ธ๐ฆ๐๐๐ฏ๐ข๐ถ๐ฆ๐ง๐ข๐๐ซ๐๐๐ฝ๐๐๐๐ฒ๐๐ฅ๐ธโโฃ๐โโ๐ฏ๐๐ฐ๐ง๐ฟ๐ณ๐ท๐บ๐๐๐๐ค๐ฒ๐๐น๐๐ท๐๐ฅ๐ต๐๐ธโ โ๐ฉโ๐ผ๐โฌโพ๐๐๐โฝ๐ญ๐๐ท๐โ๐๐๐๐๐๐ค๐ฅ๐ฝ๐๐๐ฐ๐๐ดโ๐ฆ๐โ๐๐ด๐๐๐ก๐๐ฉ๐๐บโ๐ผ๐๐ถ๐บ๐๐ฌ๐๐ป๐พโฌโฌโถ๐ฎ๐โ๐ธ๐ถ๐ฎ๐ชโณ๐๐พ๐๐ด๐จ๐๐นยฉ๐ฃ๐ฆ๐ฃ๐จ๐๐ฌโญ๐น๐ท"
        )



# load tagger for NER
@st.cache(allow_output_mutation=True)
def load_flair_model():
    tagger = SequenceTagger.load('ner')
    q_tagger = torch.quantization.quantize_dynamic(
        tagger, {torch.nn.Linear}, dtype=torch.qint8
        #tagger, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    )
    del tagger
    return q_tagger
    #return tagger






@st.cache(allow_output_mutation=True, hash_funcs={SequenceTagger: lambda _: None})
def predict_flair(model, text):
    manual_sentence = Sentence(text)
    model.predict(manual_sentence)
    #return render_ner_html(manual_sentence, colors=colors, wrap_page=False)
    return manual_sentence



# HunFlair model trained on 23 biomedical NER data sets
#@st.cache(allow_output_mutation=True, hash_funcs={MultiTagger: lambda _: None})
#@st.cache(suppress_st_warning=True)
def load_hunflair_model():
    # load biomedical tagger
    tagger = MultiTagger.load('hunflair')
    #tagger = MultiTagger.load('hunflair-gene')

    #q_tagger = torch.quantization.quantize_dynamic(
    #    tagger, {torch.nn.Linear}, dtype=torch.qint8
        #tagger, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    #)
    #del tagger
    #return q_tagger
    return tagger


def predict_hunflair(model, text):
    manual_sentence = Sentence(text)
    model.predict(manual_sentence)
    #return render_ner_html_custom(manual_sentence, colors=colors_2)
    #return manual_sentence.to_tagged_string()
    return manual_sentence


def get_tag(sentence):
    #spans = [(entity.text, entity.tag, entity.score) for entity in sentence.get_spans()]
    #df = pd.DataFrame(spans, columns=['text', 'tag', 'score'])
    spans = [(entity.text, entity.tag) for entity in sentence.get_spans()]
    df = pd.DataFrame(spans, columns=['text', 'tag' ])   
    df = df.drop_duplicates(keep=False)
    return df




st.title('Topic Modeling and Named Entity Recognition (NER) tagger')
st.subheader('Created with ๐ค๐ค๐ค by [Aidan Gawronski](https://github.com/agawronski), [Duc Vu](https://github.com/dvu4), [Florencia Diaz](https://github.com/fldiaz) ')

acttivities = ['Topic Model', 'NER']

choice = st.sidebar.selectbox('Choose task', acttivities)

if choice =='NER':
    st.info('Named Entity Recognition')

    user_input = st.text_area('Enter here', '')

    ner_acttivities = ['Flair', 'HunFlair']
    ner_choice = st.sidebar.selectbox('Choose models', ner_acttivities)
    if ner_choice == 'Flair':
        model = load_flair_model()

        if st.button('Analyze'):
            sentence = predict_flair(model, user_input)
            st.success("Below is your tagged string.")
            #st.write(sentence, unsafe_allow_html=True)
            df = get_tag(sentence)
            st.dataframe(df)

    if ner_choice == 'HunFlair':
        model = load_hunflair_model()
        if st.button('Analyze'):
            sentence = predict_hunflair(model, user_input)
            df = get_tag(sentence)
            st.success("Below is your tagged string.")
            st.dataframe(df)
            #st.write(sentence, unsafe_allow_html=True)



if choice =='Topic Model':
    st.info('Topic Model')

    user_input = st.text_area('Enter here', '')

    tm_acttivities = ['LDA', 'BERT']
    tm_choice = st.sidebar.selectbox('Choose models', tm_acttivities)


    if tm_choice == 'LDA':
        #model = load_hunflair_model()
        if st.button('Analyze'):
            st.success("Below is your tagged string.")
            st.write(user_input, unsafe_allow_html=True)

    if tm_choice == 'BERT':
        #model = load_hunflair_model()
        if st.button('Analyze'):
            st.success("Below is your tagged string.")
            st.write(user_input, unsafe_allow_html=True)


#https://github.com/streamlit/streamlit/blob/54349972db99a725567f0454d336c948e5dc0b82/lib/streamlit/commands/page_config.py#L107
