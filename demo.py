'''
@author: taridzo, 2021
First version of a DEMO for clincode. It uses a fine-tuned model and flask for the view.
'''
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow import keras
from flask import Flask, redirect, request, render_template, make_response
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib import colors
import demo_fuzzy_sentence as fz
import time

MODEL_NAME = "static/swedClinBert"
icd_desc_f = 'static/icd/gastro_se_blk.csv'
data_f = 'static/data/data.csv'
choices = pd.read_csv(icd_desc_f, sep=',', header=0)
modelx = keras.models.load_model('static/model')
modelx = tf.keras.Model(inputs=modelx.input, outputs=[modelx.output, modelx.get_layer('attention_vec').output])

label_dict = ['K209', 'K210', 'K219', 'K222', 'K250', 'K251', 'K253', 'K259', 'K260', 'K261', 'K263', 'K279', 'K295',
              'K297', 'K298', 'K299', 'K309', 'K310', 'K317', 'K351', 'K352', 'K353', 'K358', 'K359', 'K369', 'K379',
              'K403', 'K409', 'K420', 'K429', 'K430', 'K432', 'K435', 'K439', 'K449', 'K469', 'K500', 'K501', 'K508',
              'K509', 'K510', 'K512', 'K519', 'K523', 'K528', 'K529', 'K550', 'K560', 'K561', 'K562', 'K564', 'K565',
              'K566', 'K567', 'K571', 'K572', 'K573', 'K578', 'K579', 'K580', 'K590', 'K591', 'K602', 'K603', 'K604',
              'K610', 'K611', 'K613', 'K621', 'K622', 'K624', 'K625', 'K626', 'K630', 'K631', 'K632', 'K635', 'K649',
              'K650', 'K658', 'K660', 'K703', 'K746', 'K750', 'K754', 'K760', 'K766', 'K768', 'K769', 'K800', 'K801',
              'K802', 'K803', 'K804', 'K805', 'K810', 'K819', 'K830', 'K831', 'K850', 'K851', 'K852', 'K858', 'K859',
              'K860', 'K861', 'K862', 'K863', 'K868', 'K900', 'K912', 'K913', 'K914', 'K920', 'K921', 'K922']

data = pd.read_csv(data_f, header=0)

clinical_notes = data.note.to_list()

MAX_LENGTH = 512
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME,
                                          add_special_tokens=True,
                                          max_length=MAX_LENGTH,
                                          pad_to_max_length=True)


def get_random_allocation():
    r_allocation = open("randomization.log", "r")
    random_alloc = r_allocation.read()
    r_allocation.close()
    rnd = open("randomization.log", "w")
    if random_alloc == "1":
        rnd.write("0")
        rnd.close()
        return "1"
    else:
        rnd.write("1")
        rnd.close()
        return "0"


def get_study_id():
    study_time = int(time.time())
    return study_time


def log_event(s_id, txt):
    log_txt = open("study.log", "a")
    log_txt.write(str(s_id) + " : " + txt + "\n")
    log_txt.close()


def log_event_demo(s_id, txt):
    log_txt = open("demo.log", "a")
    log_txt.write(str(s_id) + " : " + txt + "\n")
    log_txt.close()


def colorize(words, color_array, cm):
    cmap = cm
    template = '<span style="white-space:normal;color: black; background-color: {}"> {} </span>'
    colored_string = ''
    norm = colors.Normalize(color_array.min(), color_array.max())
    for word, color in zip(words, color_array):
        color = colors.rgb2hex(cmap(norm(color))[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string


def tokenize(sentences):
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence,
                                       add_special_tokens=True,
                                       max_length=MAX_LENGTH,
                                       pad_to_max_length=True,
                                       return_attention_mask=True,
                                       return_token_type_ids=False,
                                       truncation=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')


def get_top_n(clinical_note):
    if not isinstance(clinical_note, str): return [], [], []
    if not clinical_note: return [], [], []

    fuzzy_top_n = fz.get_fuzzy_sentence_top_n(clinical_note, choices)
    clinical_note = tokenize([clinical_note])
    y_test_probs, attention_weights = modelx.predict(clinical_note)
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
def consent():
    return render_template('consent.html')


@app.route('/start', methods=['POST', 'GET'])
def index():
    work_experience = "missing"
    if request.method == 'POST':
        if "experience" in request.form:
            work_experience = request.form['experience']
        consent = request.form['consent']

        if consent == 'Accept':
            random_allocation = get_random_allocation()
            session_id = str(get_study_id())
            log_event(session_id, consent + "|" + random_allocation + "|" + work_experience)
            if random_allocation == "1":
                intervention_help = "<img src='static/images/demo.png' alt='Intervention interface' >"
            else:
                intervention_help = "<p></p>"
            howto = make_response(render_template('howto.html', intervention_help=intervention_help))
            howto.set_cookie('session_id', str(session_id))
            howto.set_cookie("random_allocation", random_allocation)

            # log_event(session_id, "Timer: " + ":" + str(time.time()))

            return howto
    return redirect("https://icd.who.int/browse10/2019/en#/", code=302)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global clinical_notes
    # print('Clinical notes counter = ', clinical_notes_counter, file=sys.stdout)
    # log_event(study_id, " Note: " + str(clinical_notes_counter))
    selected_codes = " "
    selected_usefulness = " "
    selected_feedback = " "
    # clinical_notes_counter = 0
    next_btn = "Next"

    if "note_id" in request.cookies:
        clinical_notes_counter = int(request.cookies.get("note_id"))
    else:
        clinical_notes_counter = 0

    random_allocation = request.cookies.get("random_allocation")
    session_id = request.cookies.get("session_id")
    if "codes" in request.form: selected_codes = request.form.getlist('codes')
    if "rating" in request.form:     selected_usefulness = request.form['rating']
    if "feedback" in request.form:     selected_feedback = request.form['feedback']

    print('Selected codes = ', selected_codes, file=sys.stdout)

    log_event(session_id, "Timer: " + str(clinical_notes_counter) + ":" + time.strftime("%d.%m.%Y %H:%M:%S"))

    if clinical_notes_counter > 0:
        log_event(session_id, random_allocation + "|" + str(clinical_notes_counter) + "|" + ','.join(
            selected_codes) + "|" + selected_usefulness + "|" + selected_feedback)

    if clinical_notes_counter > len(clinical_notes) - 1:
        complete = make_response(render_template('thanks.html'))
        # complete.set_cookie('note_id', '', expires=0)
        # complete.set_cookie('session_id', '', expires=0)
        return complete
    if clinical_notes_counter == len(clinical_notes) - 1:
        next_btn = "Complete"
    if clinical_notes_counter < len(clinical_notes) - 1:
        next_btn = "Next"

    if request.method == 'POST':
        clinical_note = clinical_notes[clinical_notes_counter]
        clinical_notes_counter += 1
        progress = str(clinical_notes_counter) + " of " + str(len(clinical_notes))
        if clinical_note:
            top_n = get_top_n(clinical_note)
            top_n_results = []
            icd_txt = []
            for icd_code in top_n:
                icd_desc = choices.loc[choices.code == icd_code, 'description'].values[0]
                top_n_results.append([icd_code, icd_code + ' | ' + icd_desc])
                icd_txt.append(icd_desc)
            txt_fz, weights_fz = fz.get_fuzzy_colour(clinical_note, ' '.join([str(txt) for txt in icd_txt]))
            colour_coded_fz = colorize(txt_fz, weights_fz, plt.cm.Oranges)

            next_item_intervention = make_response(render_template('intervention.html',
                                                                   top_results=top_n_results,
                                                                   clinical_note=clinical_note,
                                                                   colour_coded_fz=colour_coded_fz,
                                                                   progress=progress,
                                                                   next_btn=next_btn,
                                                                   finnkode=get_finnkode()))
            next_item_control = make_response(render_template('control.html',
                                                              top_results=top_n_results,
                                                              clinical_note=clinical_note,
                                                              colour_coded_fz=colour_coded_fz,
                                                              progress=progress,
                                                              next_btn=next_btn,
                                                              finnkode=get_finnkode()))

            next_item_intervention.set_cookie('note_id', str(clinical_notes_counter))
            next_item_control.set_cookie('note_id', str(clinical_notes_counter))

            if random_allocation == "1":
                return next_item_intervention
            else:
                return next_item_control


@app.route('/demo', methods=['POST', 'GET'])
def demo():
    selected_codes = " "
    selected_usefulness = " "
    selected_feedback = " "
    clinical_note = None

    if "codes" in request.form: selected_codes = request.form.getlist('codes')
    if "rating" in request.form:     selected_usefulness = request.form['rating']
    if "feedback" in request.form:     selected_feedback = request.form['feedback']
    if "notes" in request.form: clinical_note = request.form['notes']

    log_event_demo(time.strftime("%d.%m.%Y %H:%M:%S"),
                   str(clinical_note or '') + "|" + ','.join(
                       selected_codes) + "|" + selected_usefulness + "|" + selected_feedback)
    return render_template('demo.html')


@app.route('/demo_predict', methods=['POST', 'GET'])
def demo_predict():

    clinical_note = None
    if "notes" in request.form: clinical_note = request.form['notes']

    if clinical_note:
        top_n = get_top_n(clinical_note)
        top_n_results = []
        icd_txt = []
        for icd_code in top_n:
            icd_desc = choices.loc[choices.code == icd_code, 'description'].values[0]
            top_n_results.append([icd_code, icd_code + ' | ' + icd_desc])
            icd_txt.append(icd_desc)
        txt_fz, weights_fz = fz.get_fuzzy_colour(clinical_note, ' '.join([str(txt) for txt in icd_txt]))
        colour_coded_fz = colorize(txt_fz, weights_fz, plt.cm.Oranges)

        demo = make_response(render_template('prediction.html',
                                             top_results=top_n_results,
                                             clinical_note=clinical_note,
                                             colour_coded_fz=colour_coded_fz,
                                             finnkode=get_finnkode()))

        return demo

    return render_template('demo.html')


app.run(debug=True, host='0.0.0.0', use_reloader=False)
