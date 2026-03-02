import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Prediction de risque de rechute d'une addiction",
    layout="wide"
)

model = joblib.load("relapse_model.pkl")

calculer = False

st.markdown("""
<style>
.card {
    background-color: #f9fafb;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}
.big-number {
    font-size: 60px;
    font-weight: 700;
}
.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("Prediction de risque de rechute d'une addiction")
st.markdown("Suivez votre etat mental journalier et estimer le risque de rechute")

left_col, right_col = st.columns([1, 1.5])

with left_col:

    st.subheader("Glissez a quel point vous ressentez ces emotions")

    mood = st.slider("Humeur (1-10)", 1, 10, 5)
    stress = st.slider("Stress (1-10)", 1, 10, 5)
    sleep_hours = st.slider("Heures de sommeil (1-10)", 1, 10, 5)
    urge_level = st.slider("Envie (1-10)", 1, 10, 5)

    day_of_week = st.selectbox(
        "Jour de la semaine",
        ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    )

    day_mapping = {
        "Lundi": 0,
        "Mardi": 1,
        "Mercredi": 2,
        "Jeudi": 3,
        "Vendredi": 4,
        "Samedi": 5,
        "Dimanche": 6
    }

    calculer = st.button("Calculer le risque de rechute")


with right_col:

    if calculer:

        features = np.array([[mood, stress, sleep_hours, urge_level, day_mapping[day_of_week]]])
        risk = model.predict_proba(features)[0][1]
        percentage = int(risk * 100)

        st.subheader("Analyse des risques")

        if risk < 0.3:
            color = "#16a34a"
            label = "Risque Faible"
        elif risk < 0.6:
            color = "#f59e0b"
            label = "Risque Modere"
        else:
            color = "#dc2626"
            label = "Risque Eleve"

        st.markdown(f"""
        <div class="center">
            <div class="big-number" style="color:{color};">
                {percentage}%
            </div>
            <h3 style="color:{color};">{label}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.progress(risk)

        if risk < 0.3:
            st.success("Risque faible de rechute - Continue d'avancer et d'etre le meilleur dans ce que tu fais !")
            st.info("Continue de maintenir ces bonnes habitudes et si tu le peux essaie de faire plus.")
            st.balloons()
        elif risk < 0.6:
            st.warning("Risque modere de rechute - Sois plus gentil avec ton corps et ton esprit.")
            st.info("Essaie de te reposer, de parler a quelqu'un, de faire du sport ou de la meditation.")
        else:
            st.error("Risque eleve de rechute - Prends des actions immediatement.")
            st.info("Contacte un proche ou un professionnel de sante et priorise ta paix mentale et physique.")

        st.divider()

        st.subheader("Resume d'aujourd'hui")

        def niveau(variable, nom):
            if variable <= 3:
                st.write(f"Faible niveau de {nom} detecte.")
            elif variable <= 6:
                st.write(f"Niveau modere de {nom} detecte.")
            else:
                st.write(f"Niveau eleve de {nom} detecte.")

        niveau(stress, "stress")
        niveau(urge_level, "envie")
        niveau(sleep_hours, "sommeil")
        niveau(mood, "humeur")



st.markdown("------------")
st.caption("Modele de prediction de rechute d'addiction a fin non medicale - Ne remplace pas l'avis d'un professionnel de sante.")
st.caption("En cas de doute rejoignez l'hopital le plus proche ou parlez-en a vos proches.")
st.caption("Projet a but educatif - Modele entraine sur un dataset synthetique - Utilisant la regression logistique - Par Romy - 2026")