#app.py
import streamlit as st
from streamlit_navigation_bar import st_navbar
import streamlit_lottie as st_lottie
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
from pages.upload import show_upload
from pages.prepare import show_prepare
from pages.train import show_train
from pages.test import show_test
from pages.documentacion import show_documentacion

# Configuración inicial
st.set_page_config(initial_sidebar_state="collapsed", page_title="Machine Learning", page_icon="🤖",layout="wide")
load_dotenv()

# Función para cargar el archivo Lottie
def load_lottie_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

# Configuración del sidebar
with st.sidebar:
    # Cargar y mostrar el logo animado
    gemini_logo = load_lottie_file('gemini_logo.json')
    st_lottie.st_lottie(
        gemini_logo, 
        key='logo', 
        height=50,
        width=50,
        loop=True,
        quality="low"
    )
    
    # Sección de API Keys
    st.markdown("### Configuración de APIs")
    
    # Gemini API
    st.markdown('''
        [Consigue tu API Key de Google AI Studio](https://aistudio.google.com/app/apikey)
    ''')
    genai_api_key = st.text_input(
        "Gemini API Key",
        type="password", 
        placeholder="Ingresa tu API Key de Gemini",
        key='gemini_api_key'
    )

    # Supabase API
    st.markdown('''
        [Consigue tus credenciales de Supabase](https://supabase.com/dashboard/project/_/settings/api)
    ''')
    supabase_url = st.text_input(
        "Supabase URL",
        type="password",
        placeholder="Ingresa tu Supabase URL",
        key='supabase_url'
    )
    
    supabase_key = st.text_input(
        "Supabase Key",
        type="password",
        placeholder="Ingresa tu Supabase Key",
        key='supabase_key'
    )

    # Validación de credenciales
    if not all([genai_api_key, supabase_url, supabase_key]):
        st.warning("Por favor ingresa todas las credenciales necesarias.")
    else:
        genai.configure(api_key=genai_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.success("✅ Credenciales configuradas correctamente")

    st.sidebar.markdown(
        f'''
        <div style="text-align: center; margin-bottom: 20px;">
            Elaborado por 
            <a href="https://www.linkedin.com/in/jersonalvr" target="_blank" style="text-decoration: none; color: inherit;">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn" width="20" style="vertical-align: middle; margin-right: 5px;"/>
                Jerson Ruiz Alva
            </a>
        </div>
        ''',
        unsafe_allow_html=True
    )

# Resto del código de navegación
pages = ["Upload", "Prepare", "Training", "Test", "Documentacion"]
styles = {
    "nav": {
        "background-color": "rgb(33, 216, 160)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Upload"

# Modified navigation bar call without default_page parameter
selected_page = st_navbar(pages, styles=styles)

if selected_page:
    st.session_state.current_page = selected_page

if st.session_state.current_page == "Upload":
    show_upload()
elif st.session_state.current_page == "Prepare":
    show_prepare()
elif st.session_state.current_page == "Training":
    show_train()
elif st.session_state.current_page == "Test":
    show_test()
elif st.session_state.current_page == "Documentacion":
    show_documentacion()
