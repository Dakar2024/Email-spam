import joblib
import pandas as pd

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
    text_series = pd.Series([email_text])
    vectorized_text = vectorizer.transform(text_series)
    prediction = model.predict(vectorized_text)[0]
    
    return "normal" if prediction == 0 else "spam"

if __name__ == "__main__":
    print("=== Détection de SPAM dans les SMS ===\n")
    print("Exemples de messages à tester :")
    print("  - 'Félicitations ! Vous avez gagné un iPhone. Cliquez ici !'")
    print("  - 'Salut, tu viens ce soir ?'")
    print("  - 'URGENT: Votre compte sera suspendu. Mettez à jour vos infos.'\n")

    user_input = input("Entrez un message à analyser :\n> ")

    if not user_input.strip():
        print("\nAucun message entré. Veuillez réessayer.")
    else:
        result = process(user_input)
        print(f"\nRésultat : Ce message est probablement : {result.upper()}")
