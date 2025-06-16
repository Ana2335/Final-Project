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
        st.markdown("Not Sarcastic 🚫")
        text = " ".join(df[df["label"] == 0]["headline"])
        wc = WordCloud(width=300, height=200, background_color="white").generate(text)
        st.image(wc.to_array())
    with col2:
        st.markdown("Sarcastic 🌀")
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

    model_option = st.selectbox("Select a model to view tuning details:", ["LSTM", "BERT", "DeBERTa"])

    if model_option == "LSTM":
        st.markdown("""
        We used Optuna to tune hyperparameters for the LSTM model with the goal of maximizing validation accuracy.
        The search space included:

        - embedding_dim 📏
        - lstm_units 🔄 
        - dropout 🌧 
        - learning_rate 🧠 
        - batch_size 📦 
        """)

        # Load and display results
        df = pd.read_csv("optuna_results.csv")

        st.subheader("Top Trials 📋")
        top_trials = df.sort_values(by="val_accuracy", ascending=False).head(5)
        st.dataframe(top_trials)

        st.subheader("Validation Accuracy Over Trials 📈")
        fig, ax = plt.subplots()
        sns.lineplot(x="trial", y="val_accuracy", data=df, marker="o", ax=ax, color="lightblue")
        ax.set_ylabel("Validation Accuracy")
        ax.set_xlabel("Trial Number")
        ax.set_title("Optuna Optimization Progress")
        st.pyplot(fig)

        st.subheader("Best Hyperparameters 🏆")
        best = df.loc[df["val_accuracy"].idxmax()]
        st.markdown(f"""
        **Best Trial Summary**

        - embedding_dim 📏: {int(best['embedding_dim'])} 
        - lstm_units 🔄: {int(best['lstm_units'])}
        - dropout 🌧: {best['dropout']:.4f}
        - learning_rate 🧠: {best['learning_rate']:.5f} 
        - batch_size 📦: {int(best['batch_size'])}  
        - validation_accuracy ✅: {best['val_accuracy']:.2%}
        """)

    elif model_option == "BERT":
        st.markdown("""
        We fine-tuned a custom BERT model using PyTorch and bert-base-uncased.  
        Although we did not use Optuna for automated tuning due to time and compute constraints,
        we manually set the following commonly recommended hyperparameters:
    
        - learning_rate 🧠: 2e-5
        - batch_size 📦: 16
        - epochs 🔁: 3
    
        These values were selected based on popular setups in BERT fine-tuning literature.
        """)
    
        st.info("Due to resource and time constraints, automated tuning was not performed for BERT. The configuration was selected manually based on established fine-tuning practices.")

    elif model_option == "DeBERTa":
        st.markdown("""
        Our DeBERTa model was fine-tuned using microsoft/deberta-base and PyTorch.  
        As with BERT, we did not perform automated tuning. Instead, we used the following parameters:
    
        - learning_rate 🧠: 2e-5
        - batch_size 📦: 16
        - epochs 🔁: 3
        - dropout 🌧: 0.3
    
        This configuration was chosen to mirror typical transformer fine-tuning setups.
        """)
        
        st.info("As with BERT, automated tuning was not applied to DeBERTa. Hyperparameters were selected manually following standard fine-tuning recommendations for transformer-based models.")

# ========== JUSTIFICATION PAGE ==========
def justification():
    st.header("Model Analysis and Justification 🧐")

    # === 1. Dataset Analysis ===
    st.subheader("What Makes This Dataset Challenging? 💪🏽")
    st.markdown("""
    The Sarcasm Headlines Dataset is challenging due to several factors:
    
    - ⚖️ Subtle class imbalance: Slightly more non-sarcastic headlines.
    - 🔊 Noisy language: Headlines are short, informal, and often ambiguous.
    - 📜 Context-dependency**: Sarcasm often depends on cultural or external knowledge.
    - 🧐 Label ambiguity: Some headlines may be sarcastic only in certain contexts.

    These challenges require models with strong linguistic and contextual understanding.
    """)

    # === 2. Model Justification ===
    st.subheader("Model Selection and Justification 🧩")
    st.markdown("""
    - 1️⃣ LSTM: Used as a baseline. Handles sequence data and learns word dependencies, but struggles with long-range context and syntactic nuance.
    - 2️⃣ BERT: Pretrained transformer with contextual embeddings. Strong at general sarcasm detection, though limited by its original English corpus.
    - 3️⃣ DeBERTa: Builds on BERT with disentangled attention and improved encoding of syntactic information. It handled subtle cues better and showed higher F1 in both classes.
    
    The dataset used in this project was obtained from [Hugging Face Datasets Hub](https://huggingface.co/datasets/SarcasmHeadlines) and is based on the original [Sarcasm Headlines Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) published by Misra (2018). 

    It consists of over 26,000 headlines labeled as sarcastic or not. The dataset is challenging due to short text length, subtle cues, and lack of context. These characteristics make it ideal for evaluating models that capture deep linguistic and contextual patterns, such as transformers.
    """)
    
    # === 3. Classification Reports ===
    st.subheader("Classification Reports 📄")
    st.write("1️⃣ LSTM")
    st.image("clasification_report.png", caption="LSTM", use_container_width=True)
    st.write("2️⃣ BERT")
    st.image("bert_report.png", caption="BERT", use_container_width=True)
    st.write("3️⃣ DeBERTa")
    st.image("deberta_report.png", caption="DeBERTa", use_container_width=True)
    st.markdown("""
    DeBERTa achieved the highest F1-score and balanced performance between classes, especially in identifying sarcastic content, which tends to be more contextually complex.
    """)

    # === 4. Confusion Matrices ===
    st.subheader("Confusion Matrices 📊")
    st.write("1️⃣ LSTM")
    st.image("confusion_matrix.png", caption="LSTM", use_container_width=True)
    st.write("2️⃣ BERT")
    st.image("bert_confusion_matrix.jpeg", caption="BERT", use_container_width=True)
    st.write("3️⃣ DeBERTa")
    st.image("deberta_confusion_matrix.jpeg", caption="DeBERTa", use_container_width=True)
    st.markdown("""
    - LSTM tended to overpredict the majority class (non-sarcastic).
    - BERT showed better class separation but missed subtle sarcasm.
    - DeBERTa balanced precision and recall more effectively.
    """)

    # === 5. Error Analysis ===
    st.subheader("Error Analysis 🚫")
    st.markdown("""
    **False Positives**:  
    Headlines with exaggeration or negativity, but not true sarcasm.  
    Example: "World ends tomorrow, women and minorities hardest hit"

    **False Negatives**:  
    Subtle sarcasm, often requiring world knowledge or tone.  
    Example: "Oh great, another Monday morning meeting"

    **Common patterns**:
    - Named entities (politicians, celebrities, etc.)
    - Negations and irony
    - Short ambiguous headlines

    **Suggestions for Improvement**:
    - Include external context (like news topic or article body)
    - Apply data augmentation (back translation or adversarial examples)
    - Use ensemble models (DeBERTa + CNN or LSTM voting)
    - Improve labeling with crowdsourced validation
    """)

    st.success("Overall, DeBERTa demonstrated the best understanding of syntactic cues and contextual subtleties in this sarcasm detection task.")


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

