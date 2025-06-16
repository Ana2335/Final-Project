import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix

# Cargar modelo y tokenizer
@st.cache_resource
def load_artifacts():
    model = load_model("sarcasm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# Parámetros del modelo
vocab_size = 10000
maxlen = 50

# Cargar dataset 
@st.cache_data
def load_data():
    df = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
    df = df.rename(columns={"is_sarcastic": "label"})
    return df
df = load_data()

st.title("Sarcasm Detection Classification Model 🗯️")

# Inference Interface
def main():
    st.header("Sarcasm Detection 🌀🔍")
    user_input = st.text_area("Write a text to predict if it contains sarcasm:", "Yeah, because that totally worked last time")
    
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please write a text to analyze")
        else:
            # Preprocesar y predecir
            sequence = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')
            pred = model.predict(padded)[0][0]

            # Mostrar resultado con estilo
            if pred > 0.5:
                st.success(f"Sarcastic 🌀: Trus of  {pred:.2%}")
            else:
                st.error(f"Not sarcastic 🚫: Trust of  {1 - pred:.2%}")

def visualization():
    st.header("Data visualization 📊📈")
    #st.markdown("Análisis exploratorio del conjunto de datos utilizado para entrenar el modelo de detección de sarcasmo.")

    # --- Distribución de clases ---
    st.subheader("Class distribution 📊")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="label", ax=ax1, palette="coolwarm")
    ax1.set_xticklabels(["Not sarcastic 🚫", "Sarcastic 🌀"])
    ax1.set_ylabel("Amount")
    st.pyplot(fig1)

    # --- Longitud de los titulares ---
    st.subheader("Token length histograms 📈")
    df["length"] = df["headline"].apply(lambda x: len(x.split()))
    fig2, ax2 = plt.subplots()
    sns.histplot(df["length"], bins=20, kde=True, ax=ax2)
    ax2.set_xlabel("Number of words")
    st.pyplot(fig2)

    # --- WordCloud opcional ---
    st.subheader("Word clouds ☁️")

    # Nube de palabras para cada clase
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Not Sarcastic**")
        text = " ".join(df[df["label"] == 0]["headline"])
        wc = WordCloud(width=300, height=200, background_color="white").generate(text)
        st.image(wc.to_array())

    with col2:
        st.markdown("**Sarcastic**")
        text = " ".join(df[df["label"] == 1]["headline"])
        wc = WordCloud(width=300, height=200, background_color="white").generate(text)
        st.image(wc.to_array())

    # --- Ejemplos de casos ambiguos ---
    st.subheader("Ambigous examples 🤔")

    # Predecimos todas las probabilidades
    sequences = tokenizer.texts_to_sequences(df["headline"])
    padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
    probs = model.predict(padded).flatten()

    df["pred_confidence"] = probs
    df["ambiguity_score"] = np.abs(df["pred_confidence"] - 0.5)  # valores cerca de 0.5 = más ambiguos

    # Seleccionar los más ambiguos
    ambiguous = df.sort_values("ambiguity_score").head(5)
    ambiguous["pred_label"] = np.where(ambiguous["pred_confidence"] > 0.5, "Sarcastic", "Not Sarcastic")
    ambiguous["true_label"] = ambiguous["label"].replace({0: "Not Sarcastic", 1: "Sarcastic"})

    st.dataframe(ambiguous[["headline", "true_label", "pred_label", "pred_confidence"]])

    
def tuning():
    st.header("Hyperparameter Tuning ⚙️")
    st.markdown("Results of the tuning process using Optun")

    # --- Cargar datos reales ---
    @st.cache_data
    def load_results():
        df = pd.read_csv("optuna_results.csv")  # Asegúrate de mover el archivo ahí si no está
        return df

    df = load_results()

    # --- Mostrar tabla con mejores resultados ---
    st.subheader("Top Trials 📋")
    top_trials = df.sort_values(by="val_accuracy", ascending=False).head(5)
    st.dataframe(top_trials)

    # --- Gráfica de desempeño ---
    st.subheader("Validation accuracy per trial 📈")
    fig, ax = plt.subplots()
    sns.lineplot(x="trial", y="val_accuracy", data=df, marker="o", ax=ax)
    ax.set_ylabel("Validation Accuracy")
    ax.set_xlabel("Trial")
    st.pyplot(fig)

    # --- Hiperparámetros ganadores ---
    st.subheader("Best Hyperparameters 🏆")
    best = df.loc[df["val_accuracy"].idxmax()]
    st.markdown(f"""
    - **embedding_dim:** {int(best['embedding_dim'])}  
    - **lstm_units:** {int(best['lstm_units'])}  
    - **dropout:** {best['dropout']:.4f}  
    - **learning_rate:** {best['learning_rate']:.4f}  
    - **batch_size:** {int(best['batch_size'])}  
    - **validation_accuracy:** {best['val_accuracy']:.2%}
    """)


def justification():
    st.header("Justification 🧐")

    # --- Preprocesamiento ---
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        df["headline"], df["label"], test_size=0.2, stratify=df["label"]
    )

    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_val_pad = pad_sequences(X_val_seq, maxlen=50, padding='post', truncating='post')

    # --- Predicción ---
    y_pred_prob = model.predict(X_val_pad)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # --- Classification report ---
    st.subheader("Classification Report 📋")
    st.image("clasification_report.png", caption="Classification Report", use_column_width=True)

    # --- Confusion matrix ---
    st.subheader("Confusion Matrix 📊")
    st.image("confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)

    # --- Análisis de errores ---
    st.subheader("Examples of model errors ⚠️")

    X_val_reset = X_val.reset_index(drop=True)
    y_val_reset = y_val.reset_index(drop=True)
    y_pred_flat = y_pred.flatten()

    errors = pd.DataFrame({
        "headline": X_val_reset,
        "true_label": y_val_reset,
        "predicted": y_pred_flat,
        "confidence": y_pred_prob.flatten()
    })

    false_pos = errors[(errors["true_label"] == 0) & (errors["predicted"] == 1)]
    false_neg = errors[(errors["true_label"] == 1) & (errors["predicted"] == 0)]

    st.markdown("#### False Positives 🚫 (predicted sarcasm but it wasn't)")
    st.table(false_pos[["headline", "confidence"]].head(3))

    st.markdown("#### False Negatives 🚫 (didn't predict sarcasm but it was)")
    st.table(false_neg[["headline", "confidence"]].head(3))

    # --- Justificación del modelo (opcional texto) ---
    st.subheader("Justification of the Model chosen")
    st.markdown("""An LSTM model was chosen because it is effective in sequential text classification tasks like this one. 
                The model achieved a validation accuracy of 85%, and Optuna helped find an optimal hyperparameter combination. 
                Although some errors are due to language ambiguity, the overall performance is solid.""")
#st.sidebar.title("Navigation")
#pagina = st.sidebar.selectbox

pagina = st.selectbox("Select one:", ["Inference Interface", 
                                             "Dataset Visualization", 
                                             "Hyperparameter Tuning", 
                                             "Model Analysis and Justification"])

if pagina == "Inference Interface":
    main()
elif pagina == "Dataset Visualization":
    visualization()
elif pagina == "Hyperparameter Tuning":
    tuning()
elif pagina == "Model Analysis and Justification":
    justification()
