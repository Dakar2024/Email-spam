import joblib
import pandas as pd
import gradio as gr

# Chemins vers les modèles sauvegardés
NAIVES_BAYES_MODEL_PATH = './models/naives_bayes_model.pkl'
TFIDF_VECTORIZER_PATH = './models/tfidf_vectorizer.pkl'

# Chargement du modèle et du vectorizer
vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
model = joblib.load(NAIVES_BAYES_MODEL_PATH)

def process(email_text: str) -> str:
    """
    Prend un texte (SMS ou email), le vectorise et retourne une interprétation :
    - "normal" si le modèle prédit 0
    - "spam" si le modèle prédit 1
    """
    if not email_text.strip():
        return "❌ Veuillez entrer un message."
    text_series = pd.Series([email_text])
    vectorized_text = vectorizer.transform(text_series)
    prediction = model.predict(vectorized_text)[0]
    return "✅ NORMAL" if prediction == 0 else "⚠️ SPAM"

# Liste d'exemples prédéfinis
examples = [
    ["Félicitations ! Vous avez gagné un iPhone. Cliquez ici !"],
    ["Salut, tu viens ce soir ?"],
    ["URGENT: Votre compte sera suspendu. Mettez à jour vos infos."]
]

# Interface Gradio
demo = gr.Interface(
    fn=process,
    inputs=gr.Textbox(label="Entrez un SMS ou Email à analyser", placeholder="Tapez ici..."),
    outputs=gr.Textbox(label="Résultat"),
    title="📩 Détection de SPAM dans les SMS",
    description="Analysez un message pour détecter s’il s’agit d’un SPAM ou non.",
    examples=examples
)

if __name__ == "__main__":
    demo.launch()
