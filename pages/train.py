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
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder
import shap
from streamlit_shap import st_shap

def create_xgb_model(random_state):
    return xgb.XGBRegressor(
        enable_categorical=True,
        random_state=random_state
    )

def create_xgb_classifier(random_state):
    return xgb.XGBClassifier(
        enable_categorical=True,
        eval_metric='logloss',
        random_state=random_state
    )
    
def train_model(X_train, X_test, y_train, y_test, model_info, problem_type, col, model_name, random_state):
    with col:
        # Verificar si el modelo ya está entrenado
        if ('trained_models' in st.session_state and 
            model_name in st.session_state.trained_models and 
            not st.session_state.get('retrain_models', False)):
            return st.session_state.trained_models[model_name]['model']

        # Obtener el número de folds del estado de la sesión
        n_folds = st.session_state.get('n_folds', 5)
        # Si el modelo es una función, llamarla con random_state
        if callable(model_info['model']):
            model = model_info['model'](random_state)
        else:
            model = model_info['model']
        grid_search = GridSearchCV(
            model,
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

            # Add SHAP Analysis section
            st.write("---")
            
            if st.button("Mostrar Análisis SHAP", key=f"shap_button_{model_name}"):
                try:
                    with st.spinner("Calculando valores SHAP..."):
                        model = results['model']
                        X = st.session_state.prepared_data[st.session_state.feature_cols]
                        
                        # Para modelos Pipeline, extraemos el modelo real y aplicamos el escalado a X
                        if hasattr(model, 'named_steps'):
                            if 'scaler' in model.named_steps:
                                X_transformed = pd.DataFrame(
                                    model.named_steps['scaler'].transform(X),
                                    columns=X.columns,
                                    index=X.index
                                )
                            else:
                                X_transformed = X
                            
                            # Obtener el modelo real (regressor o classifier)
                            if 'regressor' in model.named_steps:
                                actual_model = model.named_steps['regressor']
                            elif 'classifier' in model.named_steps:
                                actual_model = model.named_steps['classifier']
                            else:
                                actual_model = model
                        else:
                            X_transformed = X
                            actual_model = model
                        
                        # Configurar el explainer según el tipo de modelo
                        if isinstance(actual_model, (LinearRegression, LogisticRegression)):
                            # Para modelos lineales, usamos un enfoque específico
                            if isinstance(actual_model, LinearRegression):
                                # Para regresión lineal, usamos LinearExplainer
                                explainer = shap.LinearExplainer(actual_model, X_transformed)
                                shap_values = explainer(X_transformed)
                            else:
                                # Para regresión logística
                                explainer = shap.LinearExplainer(actual_model, X_transformed)
                                shap_values = explainer(X_transformed)
                        elif 'XGBoost' in model_name:
                            explainer = shap.TreeExplainer(actual_model)
                            shap_values = explainer(X_transformed[:100])
                        elif any(name in model_name for name in ['Random Forest', 'Árbol de Decisión']):
                            explainer = shap.TreeExplainer(actual_model)
                            shap_values = explainer(X_transformed[:100])
                        else:
                            # Para modelos SVM y otros
                            background = shap.sample(X_transformed, 100, random_state=random_state)
                            explainer = shap.KernelExplainer(
                                actual_model.predict, 
                                background,
                                feature_names=X_transformed.columns.tolist()
                            )
                            shap_values = explainer.shap_values(X_transformed[:100])
                            if isinstance(shap_values, list):
                                shap_values = shap_values[0]

                        # Contenedor para controles interactivos
                        control_col1, control_col2 = st.columns(2)
                        
                        with control_col1:
                            num_features = len(X.columns)
                            if num_features == 1:
                                max_display = 1
                                st.write("Número de características a mostrar: 1")
                            else:
                                max_display = st.slider(
                                    "Número de características a mostrar:",
                                    min_value=1,
                                    max_value=num_features,
                                    value=min(10, num_features),
                                    key=f"max_display_{model_name}"
                                )
                        
                        with control_col2:
                            sample_size = min(100, len(X))
                            sample_index = st.selectbox(
                                "Seleccionar muestra para análisis detallado:",
                                range(sample_size),
                                format_func=lambda x: f"Muestra {x + 1}",
                                key=f"sample_index_{model_name}"
                            )

                        # Contenedor para los gráficos SHAP
                        plot_container = st.container()
                        with plot_container:
                            plot_width = 380
                            # Resumen de Impacto de Variables (Beeswarm plot)
                            st.write("#### Resumen de Impacto de Variables")
                            if isinstance(shap_values, np.ndarray):
                                # For regular shap values array
                                st_shap(shap.plots.beeswarm(shap_values))
                            else:
                                # For Shap objects (like from TreeExplainer)
                                st_shap(shap.plots.beeswarm(shap_values))

                            # Waterfall plot para muestra específica
                            st.write(f"#### Análisis Detallado de Muestra {sample_index + 1}")
                            if isinstance(shap_values, np.ndarray):
                                explanation = shap.Explanation(
                                    values=shap_values[sample_index],
                                    base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                                    data=X_transformed.iloc[sample_index].values,
                                    feature_names=X_transformed.columns.tolist()
                                )
                                st_shap(shap.plots.waterfall(explanation))
                            else:
                                st_shap(shap.plots.waterfall(shap_values[sample_index]))

                            # Dependence plot para la variable más importante
                            st.write("#### Gráfico de Dependencia")
                            if isinstance(shap_values, np.ndarray):
                                most_important_feature = X_transformed.columns[np.abs(shap_values).mean(0).argmax()]
                                st_shap(shap.plots.scatter(shap_values[:, X_transformed.columns.get_loc(most_important_feature)]))
                            else:
                                feature_importance = np.abs(shap_values.values if hasattr(shap_values, 'values') else shap_values).mean(0)
                                most_important_feature = X_transformed.columns[feature_importance.argmax()]
                                st_shap(shap.plots.scatter(shap_values[:, X_transformed.columns.get_loc(most_important_feature)]))

                            # Force plot
                            st.write("#### Visualización de Fuerza (Force Plot)")
                            try:
                                if isinstance(shap_values, np.ndarray):
                                    expected_value = (
                                        explainer.expected_value[0] 
                                        if isinstance(explainer.expected_value, list) 
                                        else explainer.expected_value
                                    )
                                    st_shap(shap.plots.force(
                                        base_value=expected_value,
                                        shap_values=shap_values[sample_index],
                                        features=X_transformed.iloc[sample_index],
                                        feature_names=X_transformed.columns.tolist()
                                    ))
                                else:
                                    st_shap(shap.plots.force(shap_values[sample_index]))
                            except Exception as e:
                                st.warning(f"No se pudo generar el force plot: {str(e)}")

                            # Decision plot para modelos de árbol
                            if any(name in model_name for name in ['Random Forest', 'Árbol de Decisión', 'XGBoost']):
                                st.write("#### Gráfico de Decisión")
                                try:
                                    st_shap(shap.plots.decision(
                                        base_value=explainer.expected_value,
                                        shap_values=shap_values.values if hasattr(shap_values, 'values') else shap_values,
                                        features=X_transformed
                                    ))
                                except Exception as e:
                                    st.warning(f"No se pudo generar el decision plot: {str(e)}")

                            # Feature importance plot
                            st.write("#### Importancia de Variables")
                            if hasattr(shap_values, 'values'):
                                feature_importance = np.abs(shap_values.values).mean(0)
                            else:
                                feature_importance = np.abs(shap_values).mean(0)
                            
                            importance_df = pd.DataFrame({
                                'feature': X_transformed.columns,
                                'importance': feature_importance
                            }).sort_values('importance', ascending=False)
                            
                            fig = px.bar(
                                importance_df.head(max_display),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title='SHAP Feature Importance'
                            )
                            
                            fig.update_layout(
                                width=plot_width,
                                height=400,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error al calcular valores SHAP: {str(e)}")
                    st.exception(e)
            
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
                            smote = SMOTE(random_state=random_state)
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
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'Random Forest': {
                    'model': RandomForestRegressor(),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7, None],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'Support Vector Machine (Regressor)': {
                    'model': lambda rs: SVR(),
                    'params': {
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'C': [0.1, 1, 10, 100],
                        'epsilon': [0.1, 0.2, 0.5],
                        'gamma': ['scale', 'auto']
                    }
                },
                'XGBoost': {
                    'model': lambda rs: xgb.XGBRegressor(
                        tree_method='hist',
                        device='cuda:0',
                        enable_categorical=True,
                        random_state=rs
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300, 500],
                        'max_depth': [3, 5, 7, 9],
                        'learning_rate': [0.01, 0.05, 0.1, 0.3],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'min_child_weight': [1, 3, 5],
                        'gamma': [0, 0.1, 0.2],
                        'reg_alpha': [0, 0.1, 0.5],
                        'reg_lambda': [0.1, 1.0, 5.0]
                    }
                }
            }
        else:
            model_options = {
                'Regresión Logística': {
                    'model': LogisticRegression(max_iter=1000),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga']
                    }
                },
                'Árbol de Decisión': {
                    'model': DecisionTreeClassifier(),
                    'params': {
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'Random Forest': {
                    'model': RandomForestClassifier(),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'criterion': ['gini', 'entropy']
                    }
                },
                'Support Vector Machine (Classifier)': {
                    'model': lambda rs: SVC(random_state=rs),
                    'params': {
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'C': [0.1, 1, 10, 100],
                        'gamma': ['scale', 'auto'],
                        'class_weight': [None, 'balanced']
                    }
                },
                'XGBoost': {
                    'model': lambda rs: xgb.XGBClassifier(
                        tree_method='hist',
                        device='cuda:0',
                        enable_categorical=True,
                        eval_metric='logloss',
                        random_state=rs
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300, 500],
                        'max_depth': [3, 5, 7, 9],
                        'learning_rate': [0.01, 0.05, 0.1, 0.3],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'min_child_weight': [1, 3, 5],
                        'gamma': [0, 0.1, 0.2],
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
                        model_name,
                        random_state
                    )
                    show_model_results(model_name, problem_type, y_test, cols[i])

        except Exception as e:
            st.error(f"Error durante el entrenamiento: {str(e)}")
                
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
