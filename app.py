import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel, DebertaTokenizer, DebertaModel
from wordcloud import WordCloud
import torch.nn as nn


st.title("Sarcasm Detection Classification Model 🗯️")

# ========== CARGA DE MODELOS ==========

# -------- MODELO LSTM (Keras) --------
@st.cache_resource
def load_lstm():
    model = load_model("sarcasm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

maxlen = 50
vocab_size = 10000

# -------- MODELO BERT --------
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        x = self.dropout(pooled)
        return self.classifier(x)

@st.cache_resource
def load_bert():
    model = BertClassifier()
    state_dict = torch.load("bert_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with open("bert_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# -------- MODELO DeBERTa --------
class DebertaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        return self.classifier(x)

@st.cache_resource
def load_deberta():
    model = DebertaClassifier()
    state_dict = torch.load("deberta_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with open("deberta_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


# ========== CARGA DE DATA ==========
@st.cache_data
def load_data():
    df = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
    df = df.rename(columns={"is_sarcastic": "label"})
    return df

df = load_data()

# ========== PREDICCIÓN ==========
def predict(text, model_name):
    if model_name == "LSTM":
        model, tokenizer = load_lstm()
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')
        prob = model.predict(padded)[0][0]
        return prob

    elif model_name == "BERT":
        model, tokenizer = load_bert()
        inputs = tokenizer([text], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"]).squeeze()
            prob = torch.sigmoid(output).item()
        return prob

    elif model_name == "DeBERTa":
        model, tokenizer = load_deberta()
        inputs = tokenizer([text], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"]).squeeze()
            prob = torch.sigmoid(output).item()
        return prob

# ========== INFERENCE PAGE ==========
def inference_interface():
    st.header("Sarcasm Detection 🌀🔍")
    model_choice = st.selectbox("Select model:", ["LSTM", "BERT", "DeBERTa"])
    user_input = st.text_area("Write a text to predict if it contains sarcasm:", "Yeah, because that totally worked last time")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please write a text to analyze")
        else:
            prob = predict(user_input, model_choice)
            if prob > 0.5:
                st.success(f"{model_choice} → Sarcastic 🌀: Trust of {prob:.2%}")
            else:
                st.error(f"{model_choice} → Not sarcastic 🚫: Trust of {1 - prob:.2%}")

# ========== VISUALIZATION PAGE ==========
def visualization():
    st.header("Data visualization 📊📈")

    # --- Class Distribution ---
    st.subheader("Class distribution 📊")
    st.markdown("This chart shows how balanced the dataset is between sarcastic and non-sarcastic headlines.")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="label", ax=ax1, palette="coolwarm")
    ax1.set_xticklabels(["Not Sarcastic 🚫", "Sarcastic 🌀"])
    ax1.set_ylabel("Number of headlines")
    ax1.set_xlabel("Class")
    st.pyplot(fig1)

    # --- Token Length Histogram ---
    st.subheader("Token length histograms 📈")
    st.markdown("We measure how long headlines are in terms of number of words.")
    df["length"] = df["headline"].apply(lambda x: len(x.split()))
    fig2, ax2 = plt.subplots()
    sns.histplot(df["length"], bins=20, kde=True, ax=ax2)
    ax2.set_xlabel("Number of words per Headline")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    # --- Word Clouds ---
    st.subheader("Word clouds ☁️")
    st.markdown("Frequent words in each class may hint at language patterns linked to sarcasm.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("*Not Sarcastic 🚫*")
        text = " ".join(df[df["label"] == 0]["headline"])
        wc = WordCloud(width=300, height=200, background_color="white").generate(text)
        st.image(wc.to_array())
    with col2:
        st.markdown("*Sarcastic 🌀*")
        text = " ".join(df[df["label"] == 1]["headline"])
        wc = WordCloud(width=300, height=200, background_color="white").generate(text)
        st.image(wc.to_array())

    # --- Ambiguous Examples ---
    st.subheader("Ambiguous or Noisy examples 🤔")
    st.markdown("These are headlines where the model was least confident (close to 0.5). They may be difficult even for humans.")
    
    model, tokenizer = load_lstm()
    seqs = tokenizer.texts_to_sequences(df["headline"])
    padded = pad_sequences(seqs, maxlen=50, padding='post', truncating='post')
    probs = model.predict(padded).flatten()
    
    df["pred_confidence"] = probs
    df["ambiguity_score"] = np.abs(probs - 0.5)
    ambiguous = df.sort_values("ambiguity_score").head(5)
    ambiguous["pred_label"] = np.where(ambiguous["pred_confidence"] > 0.5, "Sarcastic", "Not Sarcastic")
    ambiguous["true_label"] = ambiguous["label"].replace({0: "Not Sarcastic", 1: "Sarcastic"})
    st.dataframe(ambiguous[["headline", "true_label", "pred_label", "pred_confidence"]])

    # --- Average Headline Length ---
    st.subheader("Average Headline Length per Class 📐")
    avg_lengths = df.groupby("label")["length"].mean()
    st.bar_chart(avg_lengths.rename(index={0: "Not Sarcastic", 1: "Sarcastic"}))

# ========== TUNING PAGE ==========
def tuning():
    st.header("Hyperparameter Tuning ⚙️")
    df = pd.read_csv("optuna_results.csv")
    st.subheader("Top Trials 📋")
    top_trials = df.sort_values(by="val_accuracy", ascending=False).head(5)
    st.dataframe(top_trials)

    st.subheader("Validation accuracy per trial 📈")
    fig, ax = plt.subplots()
    sns.lineplot(x="trial", y="val_accuracy", data=df, marker="o", ax=ax)
    ax.set_ylabel("Validation Accuracy")
    ax.set_xlabel("Trial")
    st.pyplot(fig)

    st.subheader("Best Hyperparameters 🏆")
    best = df.loc[df["val_accuracy"].idxmax()]
    st.markdown(f"""
    - *embedding_dim:* {int(best['embedding_dim'])}  
    - *lstm_units:* {int(best['lstm_units'])}  
    - *dropout:* {best['dropout']:.4f}  
    - *learning_rate:* {best['learning_rate']:.4f}  
    - *batch_size:* {int(best['batch_size'])}  
    - *validation_accuracy:* {best['val_accuracy']:.2%}
    """)

# ========== JUSTIFICATION PAGE ==========
def justification():
    st.header("Model Analysis and Justification 🧐")
    st.subheader("Classification Reports")
    st.image("clasification_report.png", caption="LSTM", use_container_width=True)
    st.image("bert_report.png", caption="BERT", use_container_width=True)
    st.image("deberta_report.png", caption="DeBERTa", use_container_width=True)

    st.subheader("Confusion Matrices")
    st.image("confusion_matrix.png", caption="LSTM", use_container_width=True)
    st.image("bert_confusion_matrix.jpeg", caption="BERT", use_container_width=True)
    st.image("deberta_confusion_matrix.jpeg", caption="DeBERTa", use_container_width=True)

    st.subheader("Error Analysis")
    st.markdown("""
    False positives often occur when headlines use exaggerated language without actual sarcasm.
    False negatives include headlines that are subtle or require external context.

    *DeBERTa* performed slightly better in recall for both classes due to its deeper syntactic understanding.
    """)

# ========== LAYOUT ==========
page = st.selectbox("Select one:", [
    "Inference Interface", 
    "Dataset Visualization", 
    "Hyperparameter Tuning", 
    "Model Analysis and Justification"])

if page == "Inference Interface":
    inference_interface()
elif page == "Dataset Visualization":
    visualization()
elif page == "Hyperparameter Tuning":
    tuning()
elif page == "Model Analysis and Justification":
    justification()

