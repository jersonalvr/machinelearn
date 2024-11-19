import streamlit as st
import streamlit.components.v1 as components

def show_documentacion():
    # Título de la sección
    st.markdown("### Repositorio de GitHub")
    
    # URL del PDF
    pdf_url = "https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"
    
    # URL para visualizar el PDF usando Google Docs Viewer
    embed_url = f"https://docs.google.com/viewer?url={pdf_url}&embedded=true"
    
    # HTML para el iframe responsivo
    github_iframe = f"""
    <iframe src="{embed_url}" width="100%" height="600px" style="border: none;"></iframe>
    """
    
    # Renderizar el HTML responsivo
    components.html(github_iframe, height=600)
    
    # Enlace directo como respaldo
    st.markdown(
        "[Abrir repositorio en una nueva pestaña ↗](https://github.com/jersonalvr/machinelearn)",
        unsafe_allow_html=True
    )
