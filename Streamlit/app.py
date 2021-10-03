import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import itertools
import string
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
MAX_SEQ_LEN = 512
num_labels = 19
lr = 5e-5
max_grad_norm = 1.0
num_training_steps = 5  # TODO
num_warmup_steps = max(1, num_training_steps // 10)
train_data_file = "../data/clean.csv"
model_ckpt = 'model0508-whole.ckpt'
codes = ['Infectious and parasitic diseases', 'Neoplasms',
         'Endocrine, nutritional and metabolic diseases, and immunity disorders',
         'Diseases of the blood and blood-forming organs', 'Mental disorders',
         'Diseases of the nervous system and sense organs', 'Diseases of the circulatory system',
         'Diseases of the respiratory system', 'Diseases of the digestive system',
         'Diseases of the genitourinary system', 'Complications of pregnancy, childbirth, and the puerperium',
         'Diseases of the skin and subcutaneous tissue', 'Diseases of the musculoskeletal system and connective tissue',
         'Congenital anomalies', 'Certain conditions originating in the perinatal period',
         'Symptoms, signs, and ill-defined conditions', 'Injury and poisoning',
         'Supplementary Classification of Factors influencing Health Status and Contact with Health Services',
         'Supplementary Classification of External Causes of Injury and Poisoning']

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinic_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", return_dict=False)

def clean_text(text):
    return [x for x in list(itertools.chain.from_iterable([t.split("<>") for t in text.replace("\n"," ").split("|")])) if len(x) > 0]

def updating_text(text):
    irrelevant_tags = ['Admission Date:', 'Date of Birth:', 'Service:', 'Allergies:', 'Attending:',
                       'Discharge Diagnosis:', 'Major Surgical or Invasive Procedure:', 'Physical Exam:',
                       'Followup Instructions:', 'Facility:']
    text = "<>".join(["|".join(re.split("\n\d|\n\s+", re.sub("^(.*?):", "", x).strip())) for x in text.split("\n\n") if
         pd.notnull(re.match("^(.*?):", x)) and re.match("^(.*?):", x).group() not in irrelevant_tags])
    text = re.sub("(\[.*?\])", "", text)
    text = "|".join(clean_text(text))
    text = " ".join([y for y in text.split("|") if len(y.split()) > 3])
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = " ".join([word for word in word_tokens if word not in stop_words])
    return filtered_text

def preprocess(note):
    note = note.replace('\n',' ')
    note = note.replace('w/', 'with')
    note = note.lower() #lower case
    note = re.sub(r'\d+', '', note) #remove numbers
    note = note.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
    note = " ".join(note.split())
    note = remove_stopwords(note)
    return note

def prepare_data(note):
    input_data = tokenizer.encode_plus(note,
                                       max_length=MAX_SEQ_LEN,
                                       truncation=True,
                                       pad_to_max_length=True,
                                       return_tensors='pt')
    input_ids = input_data['input_ids']  # IntTensor [batch_size, MAX_SEQ_LEN]

    return input_ids

class ClinicBertMultiLabelClassifier(nn.Module):
    def __init__(self, grad_clip=True):
        super(ClinicBertMultiLabelClassifier, self).__init__()
        self.grad_clip = grad_clip
        self.num_labels = num_labels

        # evaluation metrics
        self.best_f1 = 0
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_f1_micro_list = []
        self.train_f1_macro_list = []
        self.val_f1_micro_list = []
        self.val_f1_macro_list = []
        self.train_precision_list = []
        self.train_recall_list = []
        self.val_precision_list = []
        self.val_recall_list = []

        # loss function
        self.pos_weight = torch.ones([num_labels]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # network modules
        self.bert = clinic_bert
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.activate = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, aggregated_output = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.classifier(aggregated_output)

        # to avoid gradients vanishing and sigmoid nan
        if self.grad_clip:
            logits = logits.clamp(min=-14.0, max=14.0)

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = logits.reshape([-1, self.num_labels])[active_loss]
                active_labels = labels.reshape([-1, self.num_labels])[active_loss]
                loss = self.criterion(active_logits, active_labels)
            else:
                loss = self.criterion(logits.reshape([-1, self.num_labels]),
                                      labels.reshape([-1, self.num_labels]))
            return loss, self.activate(logits)
        else:
            return self.activate(logits)

def run():
    st.title("MEDICAL BILLING CODES (ICD9) PREDICTION APP")

    note = st.text_area("Enter note here...")
    if st.button("Predict"):
        note = updating_text(note)
        note = preprocess(note)
        input_ids = prepare_data(note)

        net = ClinicBertMultiLabelClassifier()
        if os.path.exists(model_ckpt):
            print("Load model from %s" % model_ckpt)
            checkpoint = torch.load(model_ckpt, map_location=device)
            net.load_state_dict(checkpoint['model'])

        with torch.no_grad():
            logits = net(input_ids, labels=None)
            probs = np.array([x for l in logits.detach().numpy() for x in l])
            st.write(pd.DataFrame({
                'ICD9 Codes': codes,
                'Prediction': probs
            }))
            #for label, prediction in zip(codes, probs):
            #    st.write(f"{label}: {prediction}")

if __name__ == '__main__':
    run()