import azure.functions as func
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import TFAutoModelForSequenceClassification
import numpy as np
import json
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CommentClassifier:
    def __init__(self):
        self.MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.rating_percentages = {
        '1 étoile': 10,   # 1 star: 10%
        '2 étoiles': 25,  # 2 stars: 25%
        '3 étoiles': 40,  # 3 stars: 40%
        '4 étoiles': 70,  # 4 stars: 70%
        '5 étoiles': 100, # 5 stars: 100%

        }
        self.rules = {
    "Facturation et coûts": [
        r"\b(coût|coûts|tarif|tarifs|prix|facture|factures|assurance|assurances|remboursement|remboursements|"
        r"facturation|tarification|cher(?:e|s)?|trop\s+cher(?:e|s)?|mensualité|mensualités|"
        r"réduction|réductions|honoraires|contribution|contributions|paiement|paiements|"
        r"financement|financements|dépassement|dépassements|frais|exorbitant(?:e|s)?|euros?|payer|argent|€|"
        r"reste\s+à\s+charge|devis|conventionné(?:e|s)?|secteur|surcoût|surcoûts|"
        r"sécurité\s+sociale|débourser|conventionnement|mutuelle|mutuelles|fric|cash|billets?)\b",
        r"carte vitale",r'carte de crédit',r'montant',r'cash',
    ],

    "Stationnement": [
        r"\b(stationnement|stationnements|parking|parkings|gar(?:er|é|ée|és|ées)?|stationner|"
        r"stationné(?:e|es|ée|ées)?|parc|parcs|voiture|voitures|véhicule|véhicules|"
        r"autopartage|zones\s+bleues)\b"
    ],

    "Transports en commun": [
    r"\b(transport|transports|RER|gare|gares|arrêt|arrêts|circulation|trafic|mobilité|"
    r"métro|bus|tramway|train|lignes\s+de\s+transport|transport\s+public|navette|"
    r"téléphérique|accessibilité\s+transport\s+public|taxis|véhicule\s+collectif|"
    r"autobus|autocar|transports\s+en\s+commun)\b"
],


    "Confidentialité et intimité": [
        r"\b(confidentialité|intimité|vie\s+privée|discrétion|secret(?:s)?|"
        r"secret\s+médical|secrets\s+médicaux|privé(?:e|s|ée|ées)?|pudeur|divulguer|"
        r"divulgation|anonymat|anonyme|anonymes|protection\s+des\s+données)\b"
    ],

    "Propreté et hygiène": [
        r"\b(propreté|hygiène|nettoyage|nettoyages|sanitaire(?:s)?|propre(?:s|ée|ées)?|sale(?:s)?|"
        r"entretien|entretiens|désinfecté(?:e|s|ée|ées)?|désinfection|désinfections|"
        r"local|locaux|assainissement|assainissements|douche|douches|"
        r"toilette(?:s)?|ordonné(?:e|s|ée|ées)?|désinfectant(?:s|ée|ées)?|sol|sols|alimentation|"
        r"alimentaire|gel|gels|cabine|cabines|stérilis(?:é|ée|és|ées|ation)|"
        r"jetable(?:s)?|poubelle(?:s)?|lavage\s+des\s+main(?:s)?|serviette(?:s)?|masque\s+chirurgical)\b"
    ],

    "Facilité de prise de RDV": [
        r"\b(prise|prises|prendre|disponibilité|disponibilités|"
        r"réservation|réservations|annulation|annulations|"
        r"report|reports|"
        r"planning|plannings|planification|planifications|créneau|créneaux)\b"
    ],
   "Facilités pour les PMR": [
        r"\b(rampe(?:s)?|ascenseur(?:s)?|personne(?:s)?\s+à\s+mobilité\s+réduite|pmr|handicap(?:é|ée|és|ées|é|ées)?|"
        r"adapté(?:e|s|ée|ées)?|accessibilité|invalid(?:ité|e|es)?|mobilité\s+réduite|facilité(?:s)?\s+d'accès)\b"
    ],


    "Amabilité du personnel d’accueil": [
        r"\b(accueil|personnel|secrétaire(?:s)?|hôte(?:sse|sses)?|accompagnement|"
        r"guichet(?:s)?|réception(?:niste|nistes)?|secrétariat(?:s)?|enregistrement|"
        r"assistant(?:e|s)?)\b"
    ],

    "Rapidité": [
        r"\b(rapide(?:s|ment)?|efficac(?:ité|e)?|vite|accéléré(?:e|s)?|instantané(?:e|s)?|"
        r"immédiat(?:e|s)?|réactivité|expéditif(?:ve|s)?|patienter|express|court(?:e|s)?)\b"
    ],

    "Ponctualité et attente": [
        r"\b(ponctuel(?:le|s)?|avance(?:s)?|attente(?:s)?|retard(?:s)?|ponctualité|"
        r"temps|horaire(?:s)?|délai(?:s)?|long(?:ue|s|ues)?)\b"
    ],

    "Accessibilité des résultats": [
        r"\b(résultat(?:s)?|en\s+ligne|télécharg(?:er|é|ée|és|ées|ement|ements)?|"
        r"plateforme(?:s)?|compte\s+rendu(?:s)?|comptes\s+rendus?|"
        r"télétransmission(?:s)?|automatique(?:s)?|informatique(?:s)?|"
        r"réseau(?:x)?|diffusion(?:s)?|internet|site(?:s)?|courrier(?:s)?|"
        r"média(?:s)?|données|dossier(?:s)?|resultat(?:s)?|application(?:s)?|"
        r"accès(?:s)?|portail(?:s)?|recherche|requêtes|consultation(?:s)?|"
        r"document(?:s)?|fichier(?:s)?|cliché(?:s)?|web|plate-forme(?:s)?)\b"
    ],

    "Explications et clarté": [
        r"\b(expliquer|explication(?:s)?|interprétation(?:s)?|"
        r"clarifier|clair(?:e|s)?)\b",
        r"diagnostic|diagnostique|expliquant(?:s|e|es)?|expliqué(?:s|es)?|interprét(?:é|ée|ées)?\b"
    ],


    "Compétence technique": [
        r"\b(dr|radiologue(?:s)?|praticien(?:ne|s)?|chirurgien(?:ne|s)?|"
        r"professionnel(?:le|s)?|équipe(?:s)?|manipulateur(?:trice|s)?|"
        r"prescripteur(?:s)?|docteur(?:s)?|échographe(?:s)?)|échographiste(?:s)|compétence(?:s)|expert(?:e|s|es|ise)?|compétent(?:e|s|es|ence)?\b",
        r"manip|manipulation(?:s)?|préparat(?:rice|eur)?|intervenant(?:s|e|es)?|analyse(?:s)?|pratique(?:s)?|qualifi(?:é|ées|és|ée|cation)?\b"
    ],

    "Infrastructure": [
        r"\b(machine(?:s)?|infrastructure(?:s)?|équipement(?:s)?|appareil(?:s)?|"
        r"dispositif(?:s)?|matériel(?:s)?|outil(?:s)?|technologie(?:s)?|panne(?:s)?)\b"
        r"automate(?:s)|réparation|maintenance|outils|outil|numérique(?:s)?|installation(?:s)?|borne(?:s)|matériel(?:s)?\b"
    ],

    "Urgence": [
        r"\b(urgence(?:s)?|urgent(?:e|s)?)\b"
        ],
    "Conseils et recommandations": [
        r"\b(recommandation(?:s)?|conseil(?:s)?)\b"
    ]

}

    def classify_comment(self, comment):
        labels = []
        comment = comment.lower()  # On normalise le commentaire en minuscules
        for label, patterns in self.rules.items():
            if any(re.search(pattern, comment) for pattern in patterns):
                labels.append(label)
        return labels

    def predict_sentiment(self,comment):
        inputs = self.tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = outputs.logits.softmax(dim=1).tolist()[0]
        labels = ["1 étoile", "2 étoiles", "3 étoiles", "4 étoiles", "5 étoiles",]
        sentiment = labels[np.argmax(scores)]
        return sentiment

    def extraire_segments(self,texte,sub_rules):
          rules_extracted = {cle: self.rules[cle] for cle in sub_rules if cle in self.rules}
          # Fonction pour segmenter le texte en phrases
          def segmenter_texte(texte):
              return re.split(r'(?<=[.!?])\s+', texte)

          # Dictionnaire pour stocker les segments extraits
          extracted_segments = {key: [] for key in rules_extracted}
          phrases = segmenter_texte(texte)
          # Extraire les segments pour chaque catégorie en fonction des règles

          for category, patterns in rules_extracted.items():
              for pattern in patterns:
                     for phrase in phrases:
                      if re.search(pattern, phrase, flags=re.IGNORECASE):
                          extracted_segments[category].append(phrase.strip())

          resultats = []
          for category, segments in extracted_segments.items():
              if segments:
                  resultats.append({category: segments})
              else:
                  resultats.append({category: []})

          return resultats

    def get_stats(self, comment):
        stats = []  # Liste pour stocker les résultats
        labels = self.classify_comment(comment)
        global_score = 0
        count = 0  # Compteur pour éviter la division par zéro

        if labels:  # Si des catégories sont trouvées
            sub_comments = self.extraire_segments(comment, labels)

            for com in sub_comments:
                for key, value in com.items():
                    sub_comment = str(value) if value else comment  # Prendre le premier sous-commentaire ou le commentaire complet

                    sentiment = self.predict_sentiment(sub_comment)
                    rating = self.rating_percentages.get(sentiment, 0)  # Par défaut 0 si le sentiment n'est pas trouvé

                    global_score += rating
                    count += 1  # Incrémenter le compteur si une note est ajoutée

                    stats.append({
                        'categorie': key,
                        'sub_comment': sub_comment,
                        'sentiment': sentiment,
                        'note': rating
                    })

            moyenne_score = global_score / count if count > 0 else 0  # Éviter la division par zéro
            return stats, moyenne_score

        else:  # Si aucun label n'est détecté
            sentiment = self.predict_sentiment(comment)
            rating = self.rating_percentages.get(sentiment, 0)  # Par défaut 0

            stats.append({
                'categorie': 'AUTRE',
                'sub_comment': comment,
                'sentiment': sentiment,
                'note': rating
            })

            return stats, rating

c=CommentClassifier()

def compute_stats(data_json_format):
    logging.info("Début de l'exécution de compute_stats")
    
    # Vérifier si le dictionnaire est vide ou mal formé
    if not data_json_format or "Commentaire" not in data_json_format:
        logging.warning("Données invalides : data_json_format est vide ou ne contient pas 'Commentaire'.")
        return {}, None, []

    commentaires = data_json_format["Commentaire"]
    
    # Vérifier si la liste de commentaires est vide
    if not commentaires or not all(isinstance(x, str) for x in commentaires):
        logging.warning("Données invalides : 'Commentaire' est vide ou contient des éléments non textuels.")
        return {}, None, []

    try:
        df = pd.DataFrame({"Commentaire": commentaires})
        logging.info(f"DataFrame créé avec {len(df)} commentaires.")

        # Appliquer get_stats en gérant les cas où cela retourne None
        df[['stats', 'note']] = df['Commentaire'].apply(
            lambda x: pd.Series(c.get_stats(x) if c.get_stats(x) else ({}, None))
        )
        logging.info("Extraction des statistiques terminée.")

        # Calcul de la moyenne des notes
        moyenne_note = df['note'].mean() if not df['note'].isnull().all() else None
        logging.info(f"Moyenne des notes calculée : {moyenne_note}")

        # Gestion des stats
        df_exploded = df.explode('stats')
        df_stats = pd.json_normalize(df_exploded['stats'])

        # Calcul du score par catégorie si df_stats n'est pas vide
        if not df_stats.empty:
            score_par_categorie = (
                df_stats.groupby("categorie")["note"].mean().reset_index().to_dict(orient="records")
            )
            logging.info(f"Score par catégorie calculé avec {len(score_par_categorie)} catégories.")
        else:
            score_par_categorie = []
            logging.warning("Aucune statistique extraite (df_stats est vide).")

        # Convertir le DataFrame en JSON
        json_output = df.to_json(orient="columns")
        json_dict = json.loads(json_output)
        logging.info("Conversion en JSON terminée.")

        logging.info("Fin de l'exécution de compute_stats")
        return json_dict, moyenne_note, score_par_categorie

    except Exception as e:
        logging.error(f"Erreur lors de l'exécution de compute_stats : {e}", exc_info=True)
        return {}, None, []



def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        query = req_body.get('data')

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "No data provided in request body"}),
                mimetype="application/json",
                status_code=400
            )
        
        dff, note_moy , note_cat = compute_stats(query)

        return func.HttpResponse(
            json.dumps({"stats": dff , "note moyenne" :note_moy , "note par categorie" : note_cat}),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500









