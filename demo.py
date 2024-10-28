'''
@author: taridzo, 2021
First version of a DEMO for clincode. It uses a fine-tuned model and flask for the view.
'''
import pandas as pd
from flask import Flask, jsonify, request, render_template, make_response
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib import colors
import demo_fuzzy_sentence as fz
import time
import json
from classifier_utils import Classifier
import os
import string
from nltk.corpus import stopwords

MODEL_DIR = "model_dir"
stop_words_se = set(stopwords.words('swedish'))
stop_words_no = set(stopwords.words('norwegian'))

punct = string.punctuation

data_f = 'static/data/data.csv'
data = pd.read_csv(data_f, sep=',', header=0)

clinical_notes = data.note.to_list()

icd_desc_f = 'static/icd/gastro_se_blk.csv'
icd_descriptions = pd.read_csv(icd_desc_f, sep=',')

stop_f = 'static/icd/stoppord_ext_se.csv'
stop_ext = pd.read_csv(stop_f, sep=',')

stop_ext_se = stop_ext.stopword.to_list()
stop_words = list(set(list(stop_words_se) + list(stop_words_no) + stop_ext_se))

choices = pd.read_csv(icd_desc_f, sep=',', header=0)

icd_index2code = open('id2labels.json')
icd_codes = json.load(icd_index2code)

#Most frequent codes??
correct_codes = ["K076","K732","K225","K402","K409","K513","K509","K522","K635","K649","K222","K573","K358","K359","K439","K567","K011","K803","K112","K602","K115"]

MODEL_PATH = os.environ.get('MODEL_PATH', '/models/' + MODEL_DIR)
ID2CAT_PATH = os.environ.get('ID2CAT_PATH', 'id2labels.json')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', '/models/' + MODEL_DIR)

classifier = Classifier(model_path=MODEL_PATH,
                             id2cat_path=ID2CAT_PATH,
                             tokenizer_path=MODEL_PATH)


def get_random_allocation():
    r_allocation = open("experiment_logs/randomization.log", "r")
    random_alloc = r_allocation.read()
    r_allocation.close()
    rnd = open("experiment_logs/randomization.log", "w")
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

def log_event(detailed, s_id, txt):
    if detailed:
        log_file = "experiment_logs/study_detailed.log"
    else:
        log_file = "experiment_logs/study.log"
    log_txt = open(log_file, "a")
    log_txt.write(str(s_id) + "|" + txt + "\n")
    log_txt.close()


def log_event_demo(s_id, txt):
    log_txt = open("experiment_logs/demo.log", "a")
    log_txt.write(str(s_id) + "|" + txt + "\n")
    log_txt.close()


def colorize(words, color_array, cm):
    cmap = cm
    template = '<span style="background-color: {}">{} </span>'
    colored_string = ''
    norm = colors.Normalize(color_array.min(), color_array.max()*3)
    for word, color in zip(words, color_array):
        if word.strip().lower().translate(str.maketrans('', '', punct)) in stop_words: color = "transparent"
        if color is not "transparent":
            if color < 60: color = "transparent"
        if color is not "transparent":
            color = colors.rgb2hex(cmap(norm(color))[:3])
        colored_string += template.format(color, word )

    return colored_string


def get_top_n(clinical_note):
    if not isinstance(clinical_note, str): return [], [], []
    if not clinical_note: return [], [], []

    fuzzy_top_n = fz.get_fuzzy_sentence_top_n(clinical_note, choices)
    preds, attentions, tokens = classifier(clinical_note)
    bert_top_n = []
    for code_index in preds:
        bert_top_n.append(icd_codes[str(code_index)])
    return list(set(bert_top_n + fuzzy_top_n[:2]))


def get_finnkode_no():
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

def get_finnkode():
    record = []
    for index, row in choices.iterrows():
        record.append([['code', row.code[:3] + '.' + row.code[3:] + ' | ' + row.description], []])
    return  record

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
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


@app.route('/demo', methods=['POST', 'GET'])
def consent():
    return render_template('consent.html')


@app.route('/start', methods=['POST', 'GET'])
def index():
    work_experience = "missing"
    primary_language1 = ""
    primary_language2 = ""
    if request.method == 'POST':
        consent = request.form['consent']
        if "experience" in request.form:
            work_experience = request.form['experience']
        if "language1" in request.form:
            primary_language1 = request.form['language1']
        if "language2" in request.form:
            primary_language2 = request.form['language2']

        if consent == 'Accept':
            random_allocation = get_random_allocation()
            session_id = str(get_study_id())
            log_event(True, session_id, consent + "|" + random_allocation + "|" + work_experience + "|" + primary_language1 + " " + primary_language2)
            howto = make_response(render_template('howto.html'))
            howto.set_cookie('session_id', str(session_id))
            howto.set_cookie("random_allocation", random_allocation)

            return howto

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global clinical_notes
    selected_codes = " "
    selected_usefulness = " "
    selected_feedback = " "
    next_btn = "Next"


    if "coding_timer" in request.cookies:
        coding_start_time = int(request.cookies.get("coding_timer"))
        coding_duration = int(time.time()) - coding_start_time
    else:
        coding_duration = -1

    if "note_id" in request.cookies:
        clinical_notes_counter = int(request.cookies.get("note_id"))
    else:
        clinical_notes_counter = 0

    random_allocation = request.cookies.get("random_allocation")
    session_id = request.cookies.get("session_id")
    if "codes" in request.form: selected_codes = request.form.getlist('codes')
    if "rating" in request.form:     selected_usefulness = request.form['rating']
    if "feedback" in request.form:     selected_feedback = request.form['feedback']

    #print('Selected codes = ', selected_codes, file=sys.stdout)

    log_event(True, session_id, "Timer_" + str(clinical_notes_counter) + " @ " + time.strftime("%d.%m.%Y %H:%M:%S"))

    if clinical_notes_counter > 0:
        log_event(True, session_id, random_allocation + "|" + str(clinical_notes_counter) + "|" + str(correct_codes[clinical_notes_counter-1]) + "|" + ','.join(
            selected_codes) + "|" + selected_usefulness + "|" + selected_feedback + "|" + str(coding_duration))
        log_event(False, session_id, random_allocation + "|" + str(clinical_notes_counter) + "|" + str(correct_codes[clinical_notes_counter-1]) +  "|" + ','.join(
            selected_codes) + "|" + selected_usefulness + "|" + selected_feedback + "|" + str(coding_duration))

    if clinical_notes_counter > len(clinical_notes) - 1:
        complete = make_response(render_template('thanks.html'))
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
                try:
                    icd_desc = choices.loc[choices.code == icd_code, 'description'].values[0]
                except:
                    icd_desc = "Code does not exist!"
                top_n_results.append([icd_code, icd_code[:3] + '.' + icd_code[3:]  + ' | ' + icd_desc])
                icd_txt.append(icd_desc)
            txt_fz, weights_fz = fz.get_fuzzy_colour(clinical_note, ' '.join([str(txt) for txt in icd_txt]))
            colour_coded_fz = colorize(txt_fz, weights_fz, plt.cm.Oranges)

            log_event(True, session_id, random_allocation + "|" + str(clinical_notes_counter) + "|" + ' '.join(top_n))

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
            next_item_intervention.set_cookie('coding_timer', str(int(time.time())))
            next_item_control.set_cookie('note_id', str(clinical_notes_counter))
            next_item_control.set_cookie('coding_timer', str(int(time.time())))

            if clinical_notes_counter > int(len(clinical_notes)/2):
                if random_allocation == "1":
                    random_allocation="0"
                else:
                    random_allocation="1"

            if random_allocation == "1":
                return next_item_intervention
            else:
                return next_item_control




@app.route('/demo_predict', methods=['POST', 'GET'])
def demo_predict():
    clinical_note = None
    if "notes" in request.form: clinical_note = request.form['notes']

    if clinical_note:
        top_n = get_top_n(clinical_note)
        top_n_results = []
        icd_txt = []
        #print('no. codes = ', top_n, file=sys.stdout)
        for icd_code in top_n:
            #print('choices len = ', icd_code, file=sys.stdout)
            try:
                icd_desc = choices.loc[choices.code == icd_code, 'description'].values[0]
            except:
                icd_desc = "Code does not exist!"
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

@app.route('/api', methods=['POST', 'GET'])
def api_request():
    clinical_note = "ERROR: No clinical_note or note_id received!"
    batch_results =[]

    request_data = request.get_json()
    if request_data:
        for req in request_data:
            if 'clinical_note' in req:
                if 'note_id' in req:
                    clinical_note = req['clinical_note']
                    note_id = req['note_id']
                    batch_results.append((note_id,(api_get_top_n(clinical_note, note_id))))
        if batch_results:
            return jsonify(batch_results)
    return clinical_note

def api_get_top_n(clinical_note, note_id):
    top_n = get_top_n(clinical_note)
    top_n_results = []
    for icd_code in top_n:
        try:
            icd_desc = choices.loc[choices.code == icd_code, 'description'].values[0]
        except:
            icd_desc = "Code does not exist!"
        top_n_results.append((icd_code[:3] + '.' + icd_code[3:] , icd_code[:3] + '.' + icd_code[3:] + ' | ' + icd_desc ))
    return dict(top_n_results)


app.run(debug=True, host='0.0.0.0', port='5001', use_reloader=False)