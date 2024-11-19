# Proyecto de Machine Learning
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://machinelearn.streamlit.app/)
Este proyecto es una aplicación web interactiva para cargar, preparar, entrenar y evaluar modelos de Machine Learning. Utiliza **Streamlit** como framework de desarrollo y permite a los usuarios interactuar con datos de manera sencilla.

## Características

- **Carga de Datos**: Soporta múltiples formatos de archivo (CSV, Excel, JSON, Parquet, etc.) y permite la carga desde Google Sheets y URLs.
- **Preparación de Datos**: Realiza análisis exploratorios, manejo de valores faltantes, codificación de variables categóricas y normalización.
- **Entrenamiento de Modelos**: Permite entrenar varios modelos de Machine Learning, incluyendo regresión y clasificación, con opciones de ajuste de hiperparámetros.
- **Evaluación de Modelos**: Evalúa el rendimiento del modelo con métricas relevantes y visualizaciones como matrices de confusión y gráficos de dispersión.
- **Explicaciones Automáticas**: Utiliza la API de Gemini para generar explicaciones automáticas de los resultados y parámetros de los modelos.

## Requisitos

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Google Generative AI
- Supabase
- Otras dependencias necesarias (ver `requirements.txt`)

## Instalación

1. **Clona el repositorio:**
    ```bash
    git clone https://github.com/jersonalvr/machinelearning.git
    cd machinelearning
    ```

2. **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Configura las credenciales necesarias:**
   
   - Obtén las API Keys de **Gemini** y **Supabase**.
   - Crea un archivo `.env` en la raíz del proyecto y agrega las claves de la siguiente manera:
     ```
     GEMINI_API_KEY=tu_api_key_de_gemini
     SUPABASE_URL=tu_url_de_supabase
     SUPABASE_KEY=tu_key_de_supabase
     ```
   
   - Asegúrate de que el archivo `.env` esté incluido en el archivo `.gitignore` para mantener tus credenciales seguras.

## Uso

1. **Ejecuta la aplicación:**
    ```bash
    streamlit run app.py
    ```

2. **Accede a la aplicación:**
   
   Abre tu navegador y navega a [http://localhost:8501](http://localhost:8501).

3. **Interacción:**
   
   Sigue las instrucciones en la interfaz para:
   - Cargar datos
   - Preparar el dataset
   - Entrenar modelos
   - Evaluar su rendimiento

## Contribuciones

¡Las contribuciones son bienvenidas! Para contribuir, sigue estos pasos:

1. **Haz un fork del proyecto.**
2. **Crea una nueva rama:**
    ```bash
    git checkout -b feature/nueva-caracteristica
    ```
3. **Realiza tus cambios y haz un commit:**
    ```bash
    git commit -m 'Añadir nueva característica'
    ```
4. **Sube tus cambios:**
    ```bash
    git push origin feature/nueva-caracteristica
    ```
5. **Abre un Pull Request.**

Por favor, asegúrate de que tus contribuciones sigan las [Buenas Prácticas de Código](LINK_A_TUS_PRACTICAS).

## Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE). Consulta el archivo `LICENSE` para más detalles.

## Autor

**Jerson Ruiz Alva**  
[Perfil de LinkedIn](https://www.linkedin.com/in/jersonalvr)

## Agradecimientos

Agradecemos a todas las bibliotecas y herramientas utilizadas en este proyecto:

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Plotly](https://plotly.com/)
- [Google Generative AI](https://ai.google/)
- [Supabase](https://supabase.com/)

---

¡Gracias por usar este proyecto! Si tienes alguna pregunta o sugerencia, no dudes en [contactarme](mailto:jersonruizalva@gmail.com).
