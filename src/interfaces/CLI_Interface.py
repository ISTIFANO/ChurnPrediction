import streamlit as st
import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import logging

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Path du modèle
# -------------------------------
def get_model_path():
    """Retourne le chemin absolu vers le modèle"""
    current_dir = Path(__file__).parent  # src/interfaces/
    src_dir = current_dir.parent  # src/
    model_path = src_dir / "domain" / "models" / "best_pipeline_model"
    return model_path.resolve()  # absolu

# -------------------------------
# Setup environnement PySpark
# -------------------------------
def setup_environment():
    python_exe = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe

# -------------------------------
# Initialisation Spark
# -------------------------------
@st.cache_resource
def initialize_spark():
    try:
        spark = SparkSession.builder \
            .appName("ChurnPredictionApp") \
            .master("local[1]") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        logger.info("✅ Spark session initialized")
        return spark
    except Exception as e:
        logger.error(f"Erreur Spark: {str(e)}")
        st.error(f"❌ Erreur Spark: {str(e)}")
        return None

# -------------------------------
# Charger le modèle
# -------------------------------
@st.cache_resource
def load_model(_spark, model_path):
    try:
        model_path = Path(model_path)
        metadata_file = model_path / "metadata"

        if not model_path.exists():
            raise FileNotFoundError(f"❌ Dossier modèle introuvable: {model_path}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"❌ Fichier metadata manquant dans: {model_path}")

        model = PipelineModel.load(str(model_path))
        logger.info(f"✅ Modèle chargé depuis: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {str(e)}")
        raise

# -------------------------------
# Validation inputs
# -------------------------------
def validate_inputs(credit_score, age, tenure, balance, num_products, salary):
    errors = []
    if credit_score < 300 or credit_score > 900:
        errors.append("⚠️ Credit Score entre 300 et 900")
    if age < 18 or age > 100:
        errors.append("⚠️ Age entre 18 et 100 ans")
    if tenure < 0 or tenure > 10:
        errors.append("⚠️ Ancienneté entre 0 et 10 ans")
    if balance < 0:
        errors.append("⚠️ Solde >= 0")
    if num_products < 1 or num_products > 5:
        errors.append("⚠️ Nombre de produits entre 1 et 5")
    if salary < 0:
        errors.append("⚠️ Salaire >= 0")
    return errors

# -------------------------------
# Prédiction
# -------------------------------
def make_prediction(spark, model, input_dict):
    try:
        input_data = [[
            input_dict["CreditScore"],
            input_dict["Age"],
            input_dict["Tenure"],
            input_dict["Balance"],
            input_dict["NumOfProducts"],
            input_dict["HasCrCard"],
            input_dict["IsActiveMember"],
            input_dict["EstimatedSalary"],
            input_dict["Gender"],
            input_dict["Geography"]
        ]]
        columns = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                   "HasCrCard", "IsActiveMember", "EstimatedSalary", "Gender", "Geography"]
        input_df = spark.createDataFrame(input_data, columns)
        result_df = model.transform(input_df)
        result = result_df.collect()[0]
        prediction = int(result["prediction"])
        probability = float(result["probability"][prediction])
        return prediction, probability, None
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None, None, str(e)

# -------------------------------
# Streamlit Main
# -------------------------------
def main():
    st.set_page_config(page_title="Prédiction Attrition Bancaire", layout="wide")
    st.title("🏦 Prédiction d'Attrition Bancaire")

    setup_environment()
    spark = initialize_spark()
    if spark is None:
        st.stop()

    # Chemin modèle
    model_path = get_model_path()
    st.sidebar.subheader("🔍 Debug Chemin Modèle")
    st.sidebar.text(str(model_path))
    if model_path.exists():
        st.sidebar.success("✅ Dossier modèle trouvé")
        files = list(model_path.glob("*"))
        st.sidebar.info(f"📁 {len(files)} fichiers dans le modèle")
    else:
        st.sidebar.error("❌ Dossier modèle introuvable")
        st.stop()

    # Charger modèle
    try:
        model = load_model(spark, model_path)
        st.sidebar.success("✅ Modèle chargé")
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle:\n{str(e)}")
        st.stop()

    # Inputs utilisateur
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        balance = st.number_input("Solde (€)", 0.0, 300000.0, 50000.0, step=1000.0)
        salary = st.number_input("Salaire (€)", 0.0, 200000.0, 50000.0, step=1000.0)
        num_products = st.number_input("Nombre de Produits", 1, 5, 2)
    with col2:
        age = st.number_input("Âge", 18, 100, 35)
        tenure = st.number_input("Ancienneté", 0, 10, 3)
        gender = st.selectbox("Genre", ["Male", "Female"])
        geography = st.selectbox("Pays", ["France", "Germany", "Spain"])

    col3, col4 = st.columns(2)
    with col3:
        has_card = st.selectbox("Possède Carte", [0,1])
    with col4:
        is_active = st.selectbox("Membre Actif", [0,1])

    if st.button("🔮 Prédire"):
        errors = validate_inputs(credit_score, age, tenure, balance, num_products, salary)
        if errors:
            for e in errors:
                st.error(e)
            return

        input_dict = {
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary,
            "Gender": gender,
            "Geography": geography
        }

        with st.spinner("🔄 Analyse..."):
            prediction, probability, error = make_prediction(spark, model, input_dict)

        if error:
            st.error(f"❌ Erreur prédiction: {error}")
            return

        st.subheader("📈 Résultat")
        if prediction == 1:
            st.error(f"⚠️ Risque Élevé - Probabilité: {probability:.1%}")
        else:
            st.success(f"✅ Client Fidèle - Probabilité: {probability:.1%}")

if __name__ == "__main__":
    main()
