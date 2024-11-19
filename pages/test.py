import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error,
    accuracy_score, 
    confusion_matrix, 
    classification_report
)

def load_model(uploaded_file):
    """Load and return the pickled model."""
    try:
        model = pickle.load(uploaded_file)
        # Check if it's a GridSearchCV object
        if hasattr(model, 'best_estimator_'):
            return model.best_estimator_
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def determine_problem_type(model):
    """Determine if the model is for classification or regression."""
    class_methods = ['predict_proba', 'classes_']
    return 'classification' if any(hasattr(model, method) for method in class_methods) else 'regression'

def generate_random_sample(X, y, sample_size=0.3):
    """Generate a random sample from the dataset."""
    n_samples = round(sample_size * len(X))
    indices = np.random.randint(0, len(X), n_samples)
    return X.iloc[indices], y.iloc[indices]

def evaluate_classification_model(model, X, y):
    """Evaluate a classification model and return metrics."""
    predictions = model.predict(X)
    
    return {
        'accuracy': accuracy_score(y, predictions),
        'confusion_matrix': confusion_matrix(y, predictions),
        'classification_report': classification_report(y, predictions),
        'predictions': predictions,
        'true_values': y
    }

def evaluate_regression_model(model, X, y):
    """Evaluate a regression model and return metrics."""
    predictions = model.predict(X)
    
    return {
        'mse': mean_squared_error(y, predictions),
        'r2': r2_score(y, predictions),
        'mape': mean_absolute_percentage_error(y, predictions) * 100,
        'predictions': predictions,
        'true_values': y
    }

def plot_regression_results(results):
    """Create visualization for regression results."""
    comparison_df = pd.DataFrame({
        'Actual': results['true_values'],
        'Predicted': results['predictions']
    })
    
    fig = px.scatter(comparison_df,
                    x='Actual',
                    y='Predicted',
                    title='Predicted vs Actual Values')
    
    # Add perfect prediction line
    min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
    max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val],
                  y=[min_val, max_val],
                  mode='lines',
                  name='Perfect Prediction',
                  line=dict(dash='dash', color='red'))
    )
    
    return fig

def plot_classification_results(results):
    """Create visualization for classification results."""
    cm = results['confusion_matrix']
    
    # Create heatmap of confusion matrix
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    aspect="auto",
                    title="Confusion Matrix")
    
    # Add text annotations manually
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            fig.add_annotation(
                x=j,
                y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
            )
    
    return fig

def run_experiments(model, X, y, problem_type, n_experiments=100):
    """Run multiple experiments and track performance metrics."""
    metrics_list = []
    
    for _ in range(n_experiments):
        # Generate random sample
        X_sample, y_sample = generate_random_sample(X, y)
        
        # Evaluate model
        if problem_type == 'classification':
            results = evaluate_classification_model(model, X_sample, y_sample)
            metrics_list.append(results['accuracy'])
        else:
            results = evaluate_regression_model(model, X_sample, y_sample)
            metrics_list.append(results['mape'])
    
    # Create experiment results visualization
    metric_name = 'Accuracy' if problem_type == 'classification' else 'MAPE'
    fig = px.line(y=metrics_list,
                  title=f'{metric_name} Across Experiments',
                  labels={'index': 'Experiment', 'value': metric_name})
    
    return fig, np.mean(metrics_list), np.std(metrics_list)

def get_results_explanation(problem_type, results, model_name):
    """Generate explanation of model results using Gemini."""
    try:
        # Configure Gemini
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if problem_type == 'classification':
            prompt = f"""Analiza los siguientes resultados del modelo {model_name} de clasificación:

            Accuracy: {results['accuracy']:.4f}
            
            Classification Report:
            {results['classification_report']}

            Por favor proporciona:
            1. Una interpretación clara de la accuracy y el reporte de clasificación
            2. Posibles áreas de mejora basadas en estos resultados
            3. Recomendaciones específicas para mejorar el rendimiento del modelo"""
        else:
            prompt = f"""Analiza los siguientes resultados del modelo {model_name} de regresión:

            R² Score: {results['r2']:.4f}
            MAPE: {results['mape']:.2f}%
            MSE: {results['mse']:.4f}

            Por favor proporciona:
            1. Una interpretación clara de las métricas (R², MAPE y MSE)
            2. Evaluación de la calidad del modelo basada en estos resultados
            3. Recomendaciones específicas para mejorar el rendimiento del modelo"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al generar la explicación: {str(e)}"

def show_test():
    st.title("Test Model Performance")
    
    # Model upload section
    uploaded_file = st.file_uploader("Cargar modelo entrenado (.pkl file)", type=['pkl'])
    if not uploaded_file:
        st.info("Cargue un modelo entrenado para comenzar las pruebas.")
        return
    
    # Load model
    model = load_model(uploaded_file)
    if model is None:
        return
    
    # Determine problem type
    problem_type = determine_problem_type(model)
    st.write(f"Tipo de modelo detectado: **{problem_type}**")
    
    # Data selection
    if 'prepared_data' not in st.session_state or st.session_state.prepared_data is None:
        st.warning("⚠️ No prepared data found. Please prepare your data first.")
        return
    
    data = st.session_state.prepared_data
    
    # Feature selection
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    feature_cols = st.multiselect(
        "Seleccionar variables predictoras (X):",
        numeric_cols,
        default=st.session_state.get('feature_cols', [])
    )
    
    available_targets = [col for col in data.columns if col not in feature_cols]
    target_col = st.selectbox(
        "Seleccionar variable objetivo (y):",
        available_targets,
        index=available_targets.index(st.session_state.get('target_col', available_targets[0]))
        if st.session_state.get('target_col') in available_targets else 0
    )
    
    if not (feature_cols and target_col):
        st.warning("Please select both predictor and target variables.")
        return
    
    X = data[feature_cols]
    y = data[target_col]
    
    # Testing options
    st.subheader("Testing Options")
    use_random_sample = st.checkbox("Usar random sampling", value=True)
    
    if use_random_sample:
        sample_size = st.slider("Sample size (% of data)", 0.1, 1.0, 0.3)
        X_test, y_test = generate_random_sample(X, y, sample_size)
    else:
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Evaluation section
    st.subheader("Model Evaluation")
    
    if problem_type == 'classification':
        results = evaluate_classification_model(model, X_test, y_test)
        st.write(f"Accuracy: {results['accuracy']:.4f}")
        st.text("Classification Report:")
        st.text(results['classification_report'])
        
        fig = plot_classification_results(results)
        st.plotly_chart(fig)
    else:
        results = evaluate_regression_model(model, X_test, y_test)
        st.write(f"R² Score: {results['r2']:.4f}")
        st.write(f"MAPE: {results['mape']:.2f}%")
        st.write(f"MSE: {results['mse']:.4f}")
        
        fig = plot_regression_results(results)
        st.plotly_chart(fig)
    
    # Results explanation section
    st.subheader("Análisis de Resultados")
    
    # Check for Gemini API key
    has_api_key = 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key
    
    if not has_api_key:
        st.warning("Configure su API key de Gemini en la sección superior izquierda para usar la explicación automática de los resultados.")
    
    # Create a button to trigger the explanation
    explain_button = st.button(
        "Explicar Resultados",
        disabled=not has_api_key
    )
    
    # Initialize explanations in session state if not exists
    if 'test_results_explanation' not in st.session_state:
        st.session_state.test_results_explanation = None
    
    # Show existing explanation if available
    if st.session_state.test_results_explanation:
        st.markdown(st.session_state.test_results_explanation)
    
    if explain_button and has_api_key:
        with st.spinner("Generando explicación..."):
            explanation = get_results_explanation(
                problem_type,
                results,
                uploaded_file.name.split('.')[0]
            )
            st.session_state.test_results_explanation = explanation
            st.markdown(explanation)
    
    # Multiple experiments section
    st.subheader("Multiple Experiments")
    n_experiments = st.number_input("Number of experiments", min_value=10, max_value=1000, value=100)
    
    if st.button("Run Experiments"):
        with st.spinner("Running experiments..."):
            fig, avg_metric, std_metric = run_experiments(
                model, X, y, problem_type, n_experiments
            )
            
            metric_name = 'Accuracy' if problem_type == 'classification' else 'MAPE'
            st.write(f"Average {metric_name}: {avg_metric:.4f}")
            st.write(f"{metric_name} Standard Deviation: {std_metric:.4f}")
            st.plotly_chart(fig)
    
    # Display sample predictions
    st.subheader("Sample Predictions")
    n_samples = st.slider("Number of samples to show", 5, 50, 10)
    
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    sample_X = X_test.iloc[sample_indices]
    sample_y = y_test.iloc[sample_indices]
    sample_pred = model.predict(sample_X)
    
    comparison_df = pd.DataFrame({
        'Actual': sample_y,
        'Predicted': sample_pred,
        'Difference': np.abs(sample_y - sample_pred) if problem_type == 'regression' else sample_y != sample_pred
    })
    
    st.dataframe(comparison_df)