import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import io
import google.generativeai as genai
from time import sleep

def train_model(X_train, X_test, y_train, y_test, model_info, problem_type, col, model_name):
    with col:
        # Verificar si el modelo ya está entrenado
        if ('trained_models' in st.session_state and 
            model_name in st.session_state.trained_models and 
            not st.session_state.get('retrain_models', False)):
            return st.session_state.trained_models[model_name]['model']

        # Obtener el número de folds del estado de la sesión
        n_folds = st.session_state.get('n_folds', 5)

        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=n_folds,
            n_jobs=-1
        )

        start_time = time.time()
        
        # Training progress simulation
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Actual training
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Clear progress indicators
        progress_text.empty()
        progress_bar.empty()

        # Store results in session state
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
            
        st.session_state.trained_models[model_name] = {
            'model': grid_search.best_estimator_,
            'training_time': training_time,
            'best_params': grid_search.best_params_,
            'y_pred': grid_search.predict(X_test)
        }

        return grid_search.best_estimator_

def show_model_results(model_name, problem_type, y_test, col):
    with col:
        if model_name in st.session_state.trained_models:
            results = st.session_state.trained_models[model_name]
            
            st.success(f"¡Entrenamiento completado en {results['training_time']:.2f} segundos!")
            st.write("Mejores parámetros:", results['best_params'])
            
            # Metrics
            if problem_type == 'classification':
                st.write("Accuracy:", accuracy_score(y_test, results['y_pred']))
                st.text("Reporte de clasificación:")
                st.text(classification_report(y_test, results['y_pred']))
            else:
                st.write("R² Score:", r2_score(y_test, results['y_pred']))
                st.write("MSE:", mean_squared_error(y_test, results['y_pred']))
            
            # Parameters explanation section
            st.write("---")
            st.write("### Explicación de Parámetros")
            
            # Check for Gemini API key
            has_api_key = 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key
            
            if not has_api_key:
                st.warning("Configure su API key de Gemini en la sección superior izquierda para usar la explicación automática de los parámetros.")
            
            # Initialize explanations in session state if not exists
            if 'model_explanations' not in st.session_state:
                st.session_state.model_explanations = {}
            
            # Create a button to trigger the explanation
            explain_button = st.button(
                "Explicar Parámetros",
                disabled=not has_api_key,
                key=f"explain_{model_name}"
            )
            
            # Show existing explanation if available
            if model_name in st.session_state.model_explanations:
                st.markdown(st.session_state.model_explanations[model_name])
            
            if explain_button and has_api_key:
                try:
                    with st.spinner("Generando explicación..."):
                        # Configure Gemini
                        genai.configure(api_key=st.session_state.gemini_api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        # Prepare the prompt
                        params_text = "\n".join([f"- {k}: {v}" for k, v in results['best_params'].items()])
                        prompt = f"""Explica de manera clara y concisa los siguientes parámetros del modelo {model_name} y sus valores seleccionados:

                        {params_text}

                        La explicación debe ser técnicamente precisa pero comprensible para alguien con conocimientos básicos de machine learning.
                        Incluye:
                        1. Qué hace cada parámetro
                        2. Por qué el valor seleccionado podría ser beneficioso
                        3. Posibles trade-offs de estos valores"""
                        
                        # Generate explanation
                        response = model.generate_content(prompt)
                        
                        # Store explanation in session state
                        st.session_state.model_explanations[model_name] = response.text
                        
                        # Display explanation
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error al generar la explicación: {str(e)}")
            
            # Download section
            st.write("---")
            st.write("### Descarga del modelo")
            
            # Input for model name
            model_file_key = f"model_file_{model_name}"
            if model_file_key not in st.session_state:
                st.session_state[model_file_key] = f"{model_name.lower().replace(' ', '_')}_{int(time.time())}.pkl"
            
            model_name_input = st.text_input(
                "Nombre del archivo:",
                value=st.session_state[model_file_key],
                key=f"name_input_{model_name}"
            )
            
            # Create download button with a unique key
            model_buffer = io.BytesIO()
            pickle.dump(results['model'], model_buffer)
            model_buffer.seek(0)
            
            download_key = f"download_{model_name}"
            st.download_button(
                label="Descargar Modelo",
                data=model_buffer,
                file_name=model_name_input,
                mime="application/octet-stream",
                key=download_key
            )

def show_train():
    st.title("Desarrollo de Modelos")
    
    # Create a container for status messages
    status_container = st.empty()
    
    # Check if session state has required data
    if 'prepared_data' not in st.session_state:
        status_container.warning("⚠️ No hay datos preparados en la sesión. Por favor, carga y prepara los datos primero.")
        return
        
    if st.session_state.prepared_data is None:
        status_container.warning("⚠️ Los datos preparados están vacíos. Por favor, verifica la preparación de datos.")
        return
            
    train = st.session_state.prepared_data
    
    try:
        # Modificar esta parte para incluir todas las columnas como posibles predictores
        numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        st.subheader("Configuración del Modelo")
        
        # Mantener las selecciones en session_state
        if 'feature_cols' not in st.session_state:
            st.session_state.feature_cols = []
            
        feature_cols = st.multiselect(
            "Selecciona las variables predictoras (X):",
            numeric_cols,
            default=st.session_state.feature_cols
        )
        st.session_state.feature_cols = feature_cols
        
        # Obtener TODAS las columnas disponibles para target, incluyendo categóricas
        all_cols = train.columns.tolist()
        available_targets = [col for col in all_cols if col not in feature_cols]
        
        # Manejar el caso cuando no hay columnas disponibles para target
        if not available_targets:
            st.warning("Por favor, deselecciona algunas variables predictoras para poder seleccionar la variable objetivo.")
            return
            
        # Inicializar target_col en session_state si no existe o si ya no está en available_targets
        if ('target_col' not in st.session_state or 
            st.session_state.target_col not in available_targets):
            st.session_state.target_col = available_targets[0]
        
        target_col = st.selectbox(
            "Selecciona la variable objetivo (y):",
            available_targets,
            index=available_targets.index(st.session_state.target_col)
        )
        st.session_state.target_col = target_col
        
        if not (feature_cols and target_col):
            status_container.warning("Por favor selecciona variables predictoras y objetivo.")
            return
            
        X = train[feature_cols]
        y = train[target_col]
        
        # Verificar valores nulos
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            st.error("Hay valores nulos en los datos. Por favor, vuelve a la página de preparación y maneja los valores faltantes.")
            return
        
        # Determinar tipo de problema (modificado para manejar categóricas)
        is_categorical = y.dtype == 'object' or (y.dtype.name.startswith(('int', 'float')) and y.nunique() <= 10)
        problem_type = 'classification' if is_categorical else 'regression'
        st.write(f"Tipo de problema identificado: **{problem_type}**")

        # Visualización de distribución para clasificación
        if problem_type == 'classification':
            # Crear DataFrame con la distribución de clases
            class_dist = pd.DataFrame({
                'Clase': y.value_counts().index,
                'Cantidad': y.value_counts().values
            })
            
            # Crear gráfico de barras
            fig = px.bar(
                class_dist,
                x='Clase',
                y='Cantidad',
                title=f'Distribución de clases - {target_col}'
            )
            st.plotly_chart(fig)

            # Opciones de balanceo
            if y.value_counts().min() / y.value_counts().max() < 0.5:
                st.write("⚠️ Se detectó desbalanceo en las clases")
                balance_method = st.selectbox(
                    "Técnica de balanceo:",
                    ["Ninguno", "Submuestreo", "Sobremuestreo", "SMOTE"]
                )
                
                if balance_method != "Ninguno":
                    with st.spinner("Aplicando técnica de balanceo..."):
                        if balance_method == "Submuestreo":
                            min_class_size = y.value_counts().min()
                            X, y = resample(X, y, n_samples=min_class_size*2, stratify=y)
                        elif balance_method == "Sobremuestreo":
                            max_class_size = y.value_counts().max()
                            X, y = resample(X, y, n_samples=max_class_size*2, stratify=y)
                        else:  # SMOTE
                            smote = SMOTE(random_state=42)
                            X, y = smote.fit_resample(X, y)
                    st.success("Balanceo completado!")

        # Model definitions
        if problem_type == 'regression':
            model_options = {
                'Regresión Lineal': {
                    'model': Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', LinearRegression())
                    ]),
                    'params': {
                        'regressor__fit_intercept': [True, False],
                        'regressor__copy_X': [True],
                        'regressor__positive': [True, False],
                        'scaler__with_mean': [True, False],
                        'scaler__with_std': [True, False]
                    }
                },
                'Árbol de Decisión': {
                    'model': DecisionTreeRegressor(),
                    'params': {
                        'max_depth': [3, 5, 7, 10, 15, None],
                        'min_samples_split': [2, 5, 10, 20],
                        'min_samples_leaf': [1, 2, 4, 8],
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        'splitter': ['best', 'random'],
                        'max_features': ['sqrt', 'log2', None]
                    }
                },
                'Random Forest': {
                    'model': RandomForestRegressor(),
                    'params': {
                        'n_estimators': [100, 200, 300, 500],
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_split': [2, 5, 10, 20],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None],
                        'bootstrap': [True, False],
                        'criterion': ['squared_error', 'absolute_error', 'poisson'],
                        'oob_score': [True, False]
                    }
                }
            }
        else:
            model_options = {
                'Regresión Logística': {
                    'model': LogisticRegression(max_iter=1000),
                    'params': {
                        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga'],
                        'class_weight': [None, 'balanced'],
                        'warm_start': [True, False],
                        'random_state': [42],
                        'tol': [1e-4, 1e-3, 1e-2]
                    }
                },
                'Árbol de Decisión': {
                    'model': DecisionTreeClassifier(),
                    'params': {
                        'max_depth': [3, 5, 7, 10, 15, None],
                        'min_samples_split': [2, 5, 10, 20],
                        'min_samples_leaf': [1, 2, 4, 8],
                        'criterion': ['gini', 'entropy', 'log_loss'],
                        'splitter': ['best', 'random'],
                        'max_features': ['sqrt', 'log2', None],
                        'class_weight': [None, 'balanced'],
                        'ccp_alpha': [0.0, 0.1, 0.2]
                    }
                },
                'Random Forest': {
                    'model': RandomForestClassifier(),
                    'params': {
                        'n_estimators': [100, 200, 300, 500],
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_split': [2, 5, 10, 20],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None],
                        'bootstrap': [True, False],
                        'criterion': ['gini', 'entropy', 'log_loss'],
                        'class_weight': [None, 'balanced', 'balanced_subsample'],
                        'oob_score': [True, False],
                        'warm_start': [True, False]
                    }
                },
                'XGBoost': {
                    'model': xgb.XGBClassifier(),
                    'params': {
                        'max_depth': [3, 5, 7, 9],
                        'learning_rate': [0.01, 0.05, 0.1, 0.3],
                        'n_estimators': [100, 200, 300, 500],
                        'min_child_weight': [1, 3, 5],
                        'gamma': [0, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'reg_alpha': [0, 0.1, 0.5],
                        'reg_lambda': [0.1, 1.0, 5.0],
                        'scale_pos_weight': [1, 3, 5]
                    }
                }
            }

        # Mantener modelos seleccionados en session_state
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = []

        # Configuración de división de datos y validación cruzada
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.2)
        with col2:
            random_state = st.number_input("Random State:", min_value=0, value=42)
        with col3:
            n_folds = st.number_input("Número de folds para validación cruzada:", min_value=2, max_value=10, value=5)
            st.session_state.n_folds = n_folds

        selected_models = st.multiselect(
            "Selecciona los modelos a entrenar:",
            list(model_options.keys()),
            default=st.session_state.selected_models
        )
        st.session_state.selected_models = selected_models

        if not selected_models:
            st.warning("Por favor selecciona al menos un modelo para entrenar.")
            return

        # Botón para reentrenar modelos
        if st.button("Reentrenar Modelos"):
            st.session_state.retrain_models = True
        else:
            st.session_state.retrain_models = False

        # Entrenar/mostrar modelos
        try:
            # Split data once for all models
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if problem_type == 'classification' else None
            )

            # Create columns for each model
            cols = st.columns(len(selected_models))
            
            # Train models and show results
            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    st.write(f"### {model_name}")
                    train_model(
                        X_train, X_test, y_train, y_test,
                        model_options[model_name],
                        problem_type,
                        cols[i],
                        model_name
                    )
                    show_model_results(model_name, problem_type, y_test, cols[i])

        except Exception as e:
            st.error(f"Error durante el entrenamiento: {str(e)}")
                
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")