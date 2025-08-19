import streamlit as st
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
def garmin_auth():
    st.subheader("🔑 Authentification Garmin Connect")

    login = st.text_input("📧 Identifiant Garmin")
    password = st.text_input("🔒 Mot de passe", type="password")

    if st.button("Se connecter"):
        try:
            garmin = Garmin(login, password)
            garmin.login()
            
            # Sauvegarder la session si besoin
            st.session_state["garmin"] = garmin
            
            st.success(f"Bienvenue {garmin.display_name} ✅")

        except GarminConnectAuthenticationError:
            st.error("❌ Échec de l'authentification (login/mot de passe incorrect)")
        except GarminConnectConnectionError:
            st.error("🌐 Impossible de se connecter à Garmin Connect (problème réseau ?)")
        except Exception as e:
            st.error(f"⚠️ Erreur inattendue : {e}")
