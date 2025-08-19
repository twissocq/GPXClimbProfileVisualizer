import streamlit as st
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
def garmin_auth():
    st.subheader("🔑 Authentification Garmin Connect")

    with st.form("auth_form"):
        login = st.text_input("📧 Identifiant Garmin")
        password = st.text_input("🔒 Mot de passe", type="password")
        st.form_submit_button("Se connecter")

    if st.button("Se connecter"):
        try:
            garmin = Garmin(login, password)
            garmin.login()
            
            # Sauvegarder la session si besoin
            st.session_state["garmin"] = garmin
            
            st.success(f"Bienvenue {garmin.get_full_name()} ✅")

        except GarminConnectAuthenticationError:
            st.error("❌ Échec de l'authentification (login/mot de passe incorrect)")
        except GarminConnectConnectionError:
            st.error("🌐 Impossible de se connecter à Garmin Connect (problème réseau ?)")
        except Exception as e:
            st.error(f"⚠️ Erreur inattendue : {e}")

def garmin_logout():
    """Déconnecte l'utilisateur Garmin et supprime la session."""
    if "garmin" in st.session_state:
        del st.session_state["garmin"]
        st.success("✅ Déconnecté de Garmin Connect")
    else:
        st.info("ℹ️ Aucune session Garmin active")
