import streamlit as st

def show_github():
    # Redirect to GitHub repository using HTML
    st.markdown(
        '''
        <meta http-equiv="refresh" content="0;url=https://github.com/jersonalvr/machinelearning" target="_blank">
        <p>Redirigiendo a GitHub...</p>
        ''',
        unsafe_allow_html=True
    )
    
    # Alternative: Display a link with instructions
    st.markdown(
        '''
        ### Repositorio de GitHub
        
        Si no fuiste redirigido automáticamente, haz clic en el siguiente enlace:
        
        [Ver código fuente en GitHub ↗](https://github.com/jersonalvr/machinelearning)
        ''',
        unsafe_allow_html=True
    )