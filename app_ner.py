import streamlit as st
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import MultiTagger
from flair.visual.ner_html import render_ner_html
import flair
from streamlit import caching
from pathlib import Path
import torch



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
        "🔥™🎉🚀🌌💣✨🌙🎆🎇💥🤩🤙🌛🤘⬆💡🤪🥂⚡💨🌠🎊🍿😛🔮🤟🌃🍃🍾💫▪🌴🎈🎬🌀🎄😝☔⛽🍂💃😎🍸🎨🥳☀😍🅱🌞😻🌟😜💦💅🦄😋😉👻🍁🤤👯🌻‼🌈👌🎃💛😚🔫🙌👽🍬🌅☁🍷👭☕🌚💁👅🥰🍜😌🎥🕺❕🧡☄💕🍻✅🌸🚬🤓🍹®☺💪😙☘🤠✊🤗🍵🤞😂💯😏📻🎂💗💜🌊❣🌝😘💆🤑🌿🦋😈⛄🚿😊🌹🥴😽💋😭🖤🙆👐⚪💟☃🙈🍭💻🥀🚗🤧🍝💎💓🤝💄💖🔞⁉⏰🕊🎧☠♥🌳🏾🙉⭐💊🍳🌎🙊💸❤🔪😆🌾✈📚💀🏠✌🏃🌵🚨💂🤫🤭😗😄🍒👏🙃🖖💞😅🎅🍄🆓👉💩🔊🤷⌚👸😇🚮💏👳🏽💘💿💉👠🎼🎶🎤👗❄🔐🎵🤒🍰👓🏄🌲🎮🙂📈🚙📍😵🗣❗🌺🙄👄🚘🥺🌍🏡♦💍🌱👑👙☑👾🍩🥶📣🏼🤣☯👵🍫➡🎀😃✋🍞🙇😹🙏👼🐝⚫🎁🍪🔨🌼👆👀😳🌏📖👃🎸👧💇🔒💙😞⛅🏻🍴😼🗿🍗♠🦁✔🤖☮🐢🐎💤😀🍺😁😴📺☹😲👍🎭💚🍆🍋🔵🏁🔴🔔🧐👰☎🏆🤡🐠📲🙋📌🐬✍🔑📱💰🐱💧🎓🍕👟🐣👫🍑😸🍦👁🆗🎯📢🚶🦅🐧💢🏀🚫💑🐟🌽🏊🍟💝💲🐍🍥🐸☝♣👊⚓❌🐯🏈📰🌧👿🐳💷🐺📞🆒🍀🤐🚲🍔👹🙍🌷🙎🐥💵🔝📸⚠❓🎩✂🍼😑⬇⚾🍎💔🐔⚽💭🏌🐷🍍✖🍇📝🍊🐙👋🤔🥊🗽🐑🐘🐰💐🐴♀🐦🍓✏👂🏴👇🆘😡🏉👩💌😺✝🐼🐒🐶👺🖕👬🍉🐻🐾⬅⏬▶👮🍌♂🔸👶🐮👪⛳🐐🎾🐕👴🐨🐊🔹©🎣👦👣👨👈💬⭕📹📷"
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
    return render_ner_html(manual_sentence, colors=colors, wrap_page=False)



# HunFlair model trained on 23 biomedical NER data sets
#@st.cache(allow_output_mutation=True, hash_funcs={MultiTagger: lambda _: None})
#@st.cache(suppress_st_warning=True)
def load_hunflair_model():
    # load biomedical tagger
    tagger = MultiTagger.load("hunflair")
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
    return manual_sentence.to_tagged_string()
    #return render_ner_html(manual_sentence, colors=colors_2, wrap_page=False)



st.title('Topic Modeling and Named Entity Recognition (NER) tagger')
st.subheader('Created with 🤗🤗🤗 by [Aidan Gawronski](https://github.com/agawronski), [Duc Vu](https://github.com/dvu4), [Florencia Diaz](https://github.com/fldiaz) ')

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
            st.write(sentence, unsafe_allow_html=True)

    if ner_choice == 'HunFlair':
        model = load_hunflair_model()
        if st.button('Analyze'):
            sentence = predict_hunflair(model, user_input)
            st.success("Below is your tagged string.")
            st.write(sentence, unsafe_allow_html=True)



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
