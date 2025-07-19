# Titanic-app ML App 🚢
Une application Streamlit permettant de prédire la survie de passagers du Titanic à partir d’un modèle de machine learning.

## Installation
```bash
git clone https://github.com/Julien302/Titanic-app.git
cd Titanic-app
pip install -r requirements.txt
# Entraînez d'abord votre modèle
python TrainModel.py

# Puis lancez l'app
streamlit run app.py


- `app.py` : l'application Streamlit
- `TrainModel.py` : script d'entraînement du modèle
- `model/model.joblib` : modèle ML sauvegardé
- `Data/train.csv` : données d'entraînement
- `Description.txt` : description du projet
- `requirements.txt` : dépendances Python
