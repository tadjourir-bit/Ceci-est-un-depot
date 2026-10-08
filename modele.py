from html import escape
from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


def entrainer():
    messages = [
        "gagne un cadeau gratuit",
        "clique pour gagner un cadeau",
        "offre gratuite pour gagner",
        "recois ton cadeau gratuit",
        "gagne des prix gratuits",
        "clique ici pour ton prix",
        "reunion de travail demain",
        "bonjour voici le compte rendu",
        "on se retrouve pour le projet",
        "merci pour le document de travail",
        "la reunion est prevue lundi",
        "peux tu envoyer le compte rendu",
    ]

    categories = ["spam"] * 6 + ["normal"] * 6

    modele = make_pipeline(CountVectorizer(), MultinomialNB())
    modele.fit(messages, categories)
    return modele


if __name__ == "__main__":
    modele = entrainer()

    message = "cadeau gratuit"
    categorie = modele.predict([message])[0]
    print(f"Message : {message} → {categorie}")

    dump(modele, "modele.joblib")

    Path("site").mkdir(exist_ok=True)
    page = f"""<!doctype html>
<html lang="fr">
<meta charset="utf-8">
<title>Classification de messages</title>
<h1>Classification de messages</h1>
<p>Message : {escape(message)}</p>
<p>Prédiction du modèle : {escape(categorie)}</p>
"""
    Path("site/index.html").write_text(page, encoding="utf-8")
