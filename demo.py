'''
@author: taridzo, 2022
First version of a DEMO for clincode. It uses a fine-tuned model and flask for the view.
'''
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from tensorflow import keras
from flask import Flask
from flask import request
from flask import render_template
from bs4 import BeautifulSoup

from demo_fuzzy_sentence import get_fuzzy_sentence_top_n

MODEL_NAME = "static/swedClinBert"
icd_desc_f = 'static/icd/gastro_se_blk.csv'
choices = pd.read_csv(icd_desc_f, sep=',', header=0)
modelx = keras.models.load_model('static/model')

label_dict = ['K209', 'K210', 'K219', 'K222', 'K250', 'K251', 'K253', 'K259', 'K260', 'K261', 'K263', 'K279', 'K295',
              'K297', 'K298', 'K299', 'K309', 'K310', 'K317', 'K351', 'K352', 'K353', 'K358', 'K359', 'K369', 'K379',
              'K403', 'K409', 'K420', 'K429', 'K430', 'K432', 'K435', 'K439', 'K449', 'K469', 'K500', 'K501', 'K508',
              'K509', 'K510', 'K512', 'K519', 'K523', 'K528', 'K529', 'K550', 'K560', 'K561', 'K562', 'K564', 'K565',
              'K566', 'K567', 'K571', 'K572', 'K573', 'K578', 'K579', 'K580', 'K590', 'K591', 'K602', 'K603', 'K604',
              'K610', 'K611', 'K613', 'K621', 'K622', 'K624', 'K625', 'K626', 'K630', 'K631', 'K632', 'K635', 'K649',
              'K650', 'K658', 'K660', 'K703', 'K746', 'K750', 'K754', 'K760', 'K766', 'K768', 'K769', 'K800', 'K801',
              'K802', 'K803', 'K804', 'K805', 'K810', 'K819', 'K830', 'K831', 'K850', 'K851', 'K852', 'K858', 'K859',
              'K860', 'K861', 'K862', 'K863', 'K868', 'K900', 'K912', 'K913', 'K914', 'K920', 'K921', 'K922']

MAX_LENGTH = 512
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME,
                                          add_special_tokens=True,
                                          max_length=MAX_LENGTH,
                                          pad_to_max_length=True)


def tokenize(sentences):
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence,
                                       add_special_tokens=True,
                                       max_length=MAX_LENGTH,
                                       pad_to_max_length=True,
                                       return_attention_mask=True,
                                       return_token_type_ids=True,
                                       truncation=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')


def get_top_n(clinical_note):
    if not isinstance(clinical_note, str): return []
    if not clinical_note: return []
    fuzzy_top_n = get_fuzzy_sentence_top_n(clinical_note, choices)
    clinical_note = tokenize([clinical_note])
    y_test_probs = modelx.predict(clinical_note)
    bert_top_n = [x for _, x in sorted(zip(y_test_probs[0], label_dict), key=lambda pair: pair[0], reverse=True)][:5]

    return list(set(bert_top_n + fuzzy_top_n[:2]))


def get_finnkode():
    global code
    ICD_DESCRIPTIONS = "static/icd/converted.txt"
    f = open(ICD_DESCRIPTIONS, "r")
    soup = BeautifulSoup(f.read(), features="html.parser")
    record = []
    rec = soup.find_all("span", {"class": "kodekode"})
    for recs in rec:
        code = recs.find("span", {"class": "kodenr"})
        title = recs.find("span", {"class": "hovedterm"})
        subtitles = recs.find_all("span", {"class": "tekst"})
        count = 0
        subs = []
        for sub_t in subtitles:
            subs.append(['tekst' + str(count), sub_t.text])
            count += 1
        record.append([['code', code.text + ' | ' + title.text], subs])
    return record


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('icd_demo.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        clinical_note = request.form['notes']
        top_n = get_top_n(clinical_note)
        top_n_results = []
        for icd_code in top_n:
            icd_desc = choices.loc[choices.code == icd_code, 'description'].values[0]
            top_n_results.append([icd_code, icd_code + ' | ' + icd_desc])

        return render_template('icd.html',
                               top_results=top_n_results,
                               clinical_note=clinical_note,
                               finnkode=get_finnkode())


app.run(debug=True, host='0.0.0.0', use_reloader=False)
