import streamlit as st
import streamlit.components.v1 as components

def show_github():
    # Título de la sección
    st.markdown("### Repositorio de GitHub")
    
    # HTML y CSS para un iframe responsivo
    github_iframe = """
    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
        <iframe 
            src="https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf" 
            style="position: absolute; top:0; left: 0; width: 100%; height: 100%;" 
            frameborder="0" 
            scrolling="yes">
        </iframe>
    </div>
    """
    
    # Renderizar el HTML responsivo
    components.html(github_iframe, height=600, scrolling=True)
    
    # Enlace directo como respaldo
    st.markdown(
        "[Abrir repositorio en una nueva pestaña ↗](https://github.com/jersonalvr/machinelearn)",
        unsafe_allow_html=True
    )
