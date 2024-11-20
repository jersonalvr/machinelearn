import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

def show_prepare():
    # Crear un contenedor para mensajes de estado
    status_container = st.empty()
    
    # Verificar si hay datos cargados
    if 'er_data' not in st.session_state:
        status_container.warning("⚠️ No hay datos cargados. Por favor, carga un dataset en la página Upload primero.")
        return
    
    try:
        # Usar los datos preparados si existen, si no, usar los datos originales
        if 'temp_prepared_data' in st.session_state:
            prepare = st.session_state.temp_prepared_data.copy()
        else:
            # Si no hay datos temporales, intentar usar datos preparados permanentes
            if 'prepared_data' in st.session_state:
                prepare = st.session_state.prepared_data.copy()
            else:
                prepare = st.session_state.er_data.copy()
            st.session_state.temp_prepared_data = prepare.copy()
    except AttributeError:
        status_container.warning("⚠️ No hay datos cargados o los datos son inválidos.")
        return
        
    # Análisis de valores únicos por columna
    st.markdown("### Análisis de Valores Únicos por Columna")

    # Selección de columnas para analizar - sin selección por defecto
    all_columns = prepare.columns.tolist()
    selected_columns = st.multiselect(
        "Seleccionar columnas para analizar",
        all_columns,
        default=[],  # Sin selección por defecto
        key="unique_values_columns",
        help="Selecciona las columnas que deseas analizar"
    )

    if not selected_columns:
        st.info("👆 Selecciona una o más columnas para ver su análisis detallado.")
    else:
        # Controles para número de valores a mostrar
        col1, col2 = st.columns(2)
        with col1:
            n_first = st.number_input(
                "Número de primeros valores a mostrar",
                min_value=1,
                max_value=20,
                value=5,
                key="n_first_values"
            )
        with col2:
            n_last = st.number_input(
                "Número de últimos valores a mostrar",
                min_value=1,
                max_value=20,
                value=5,
                key="n_last_values"
            )
        
        # Crear tabs para cada columna seleccionada
        tabs = st.tabs([f"📊 {col}" for col in selected_columns])
        
        # Análisis por cada columna seleccionada
        for tab, col in zip(tabs, selected_columns):
            with tab:
                try:
                    st.markdown(f"### Análisis de {col} ({prepare[col].dtype})")
                    
                    # Safely convert column to handle mixed types
                    column_data = prepare[col].fillna('Sin valor').astype(str)
                    
                    valores_unicos = column_data.unique()
                    n_valores = len(valores_unicos)
                    
                    # Información general en columnas
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"Total de valores únicos: {n_valores}")
                    with col2:
                        st.write(f"Valores nulos: {prepare[col].isnull().sum()}")
                    with col3:
                        st.write(f"% nulos: {(prepare[col].isnull().sum() / len(prepare) * 100).round(2)}%")
                    
                    st.markdown("---")
                    
                    # Visualización de valores únicos
                    if n_valores > (n_first + n_last):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("🔼 Primeros valores:")
                            for valor in valores_unicos[:n_first]:
                                st.write(f"• {str(valor)}")
                        
                        with col2:
                            st.write("🔽 Últimos valores:")
                            for valor in valores_unicos[-n_last:]:
                                st.write(f"• {str(valor)}")
                    else:
                        st.write("📝 Todos los valores únicos:")
                        for valor in valores_unicos:
                            st.write(f"• '{str(valor)}'")
                    
                    st.markdown("---")
                    
                    # Distribución de frecuencias
                    value_counts = column_data.value_counts()
                    
                    # Gráfico de barras con conversión segura
                    fig_bar = {
                        'data': [{
                            'type': 'bar',
                            'x': value_counts.index,
                            'y': value_counts.values,
                            'name': 'Frecuencia'
                        }],
                        'layout': {
                            'title': f'Distribución de valores en {col}',
                            'xaxis': {'title': 'Valor'},
                            'yaxis': {'title': 'Frecuencia'},
                            'height': 400,
                            'showlegend': False
                        }
                    }
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Tabla de frecuencias
                    freq_df = pd.DataFrame({
                        'Valor': value_counts.index,
                        'Frecuencia': value_counts.values,
                        'Porcentaje': (value_counts.values / len(prepare) * 100).round(2)
                    })
                    st.dataframe(freq_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error al procesar la columna {col}")
                    st.error(str(e))
                    st.write(f"Detalles técnicos del error en la columna {col}:", e)

    st.subheader("Preparación de Datos")

    st.subheader("Eliminación de Columnas")
    
    # Verificar el estado actual de valores nulos
    current_null_count = prepare.isnull().sum().sum()
    
    all_columns = prepare.columns.tolist()
    columns_to_drop = st.multiselect(
        "Seleccionar columnas a eliminar",
        all_columns,
        key="columns_to_drop"
    )
    
    if columns_to_drop:
        if st.button("Eliminar columnas seleccionadas", key="drop_columns_button"):
            try:
                # Crear una copia temporal antes de eliminar columnas
                temp_prepare = prepare.copy()
                
                # Verificar que las columnas existen antes de eliminarlas
                missing_cols = [col for col in columns_to_drop if col not in temp_prepare.columns]
                if missing_cols:
                    st.error(f"❌ Las siguientes columnas no existen: {', '.join(missing_cols)}")
                    return
                
                # Eliminar las columnas
                temp_prepare = temp_prepare.drop(columns=columns_to_drop)
                
                # Verificar valores nulos después de la eliminación
                new_null_count = temp_prepare.isnull().sum().sum()
                
                if new_null_count <= current_null_count:  # Permitir igual o menor cantidad de nulos
                    # Actualizar el DataFrame y el estado
                    prepare = temp_prepare
                    st.session_state.temp_prepared_data = prepare.copy()
                    
                    # Mostrar mensaje de éxito
                    st.success(f"✅ Columnas eliminadas exitosamente: {', '.join(columns_to_drop)}")
                    
                    # Actualizar información sobre valores nulos
                    if new_null_count == 0:
                        st.success("✅ No hay valores nulos en los datos.")
                    else:
                        st.warning(f"⚠️ Hay {new_null_count} valores nulos en los datos.")
                        
                    # Mostrar resumen de las columnas restantes
                    st.write(f"Columnas restantes: {len(prepare.columns)}")
                    if st.checkbox("Ver lista de columnas restantes", key="remaining_columns_checkbox"):
                        st.write(prepare.columns.tolist())
                else:
                    st.error(f"❌ La operación incrementaría los valores nulos de {current_null_count} a {new_null_count}. Operación cancelada.")
                    
            except Exception as e:
                st.error(f"Error durante la eliminación de columnas: {str(e)}")
                st.exception(e)

    # Manejo de valores faltantes
    st.subheader("Manejo de Valores Faltantes")
    missing_values = prepare.isnull().sum()
    missing_percentages = (missing_values / len(prepare) * 100).round(2)
    
    # Crear DataFrame y ordenar por número de valores faltantes de mayor a menor
    missing_df = pd.DataFrame({
        'Columna': missing_values.index,
        'Valores Faltantes': missing_values.values,
        'Porcentaje': missing_percentages.values,
        'Tipo': prepare[missing_values.index].dtypes
    })
    missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values('Valores Faltantes', ascending=False)
    
    if not missing_df.empty:
        # Mostrar advertencia si hay columnas de tipo object con valores faltantes
        object_cols = missing_df[missing_df['Tipo'] == 'object']
        if not object_cols.empty:
            st.warning("⚠️ Se detectaron columnas de tipo texto/categórico (object) con valores faltantes. " 
                      "Se recomienda revisar estos casos con especial atención ya que el método de imputación "
                      "podría afectar significativamente el análisis.")
            
            st.write("Columnas de tipo texto/categórico con valores faltantes:")
            st.dataframe(object_cols)
    
        st.write("Valores faltantes por columna (ordenados de mayor a menor):")
        st.dataframe(missing_df)

        # Checkbox para manejo especial de columnas object
        handle_objects = st.checkbox("Especificar valor de reemplazo para columnas de texto",
                                   help="Marca esta opción para especificar un valor personalizado para rellenar "
                                        "los valores faltantes en columnas de texto/categóricas")
        
        object_replacement = None
        if handle_objects:
            object_replacement = st.text_input("Valor de reemplazo para columnas de texto:",
                                             value="MISSING",
                                             help="Este valor se usará para rellenar los valores faltantes "
                                                  "en todas las columnas de texto/categóricas")

        missing_strategy = st.radio(
            "Selecciona estrategia para valores faltantes:",
            ["Eliminar filas", "Rellenar con media", "Rellenar con mediana", "Rellenar con moda"]
        )

        if st.button("Aplicar estrategia de valores faltantes", key="apply_missing_strategy_button"):
            try:
                # Guardar el estado anterior de prepare para verificación
                nulls_before = prepare.isnull().sum().sum()
                
                if missing_strategy == "Eliminar filas":
                    # Guardar el número de filas antes
                    rows_before = len(prepare)
                    
                    # Crear una copia para no modificar el original
                    prepare_cleaned = prepare.copy()
                    
                    # Eliminar filas con valores nulos
                    prepare_cleaned = prepare_cleaned.dropna(how='any')
                    
                    # Verificar que no queden valores nulos
                    if prepare_cleaned.isnull().sum().sum() == 0:
                        prepare = prepare_cleaned  # Actualizar prepare solo si la limpieza fue exitosa
                        st.session_state.temp_prepared_data = prepare.copy()  # Actualizar el estado temporal
                        rows_removed = rows_before - len(prepare)
                        st.success(f"Se eliminaron {rows_removed} filas con valores faltantes. No quedan valores nulos.")
                    else:
                        st.error(f"Error: Aún quedan {prepare_cleaned.isnull().sum().sum()} valores faltantes después de la eliminación.")
                        return
                else:
                    # Separar columnas numéricas y no numéricas
                    numeric_cols = prepare.select_dtypes(include=['int64', 'float64']).columns
                    non_numeric_cols = prepare.select_dtypes(exclude=['int64', 'float64']).columns
                    
                    # Manejar columnas object primero si se especificó un valor de reemplazo
                    if handle_objects and object_replacement is not None:
                        object_cols = prepare.select_dtypes(include=['object']).columns
                        for col in object_cols:
                            prepare[col] = prepare[col].fillna(object_replacement)
                    
                    if missing_strategy == "Rellenar con media":
                        # Para columnas numéricas usar media
                        if len(numeric_cols) > 0:
                            prepare[numeric_cols] = prepare[numeric_cols].fillna(prepare[numeric_cols].mean())
                        # Para columnas no numéricas sin valor especificado usar moda
                        if len(non_numeric_cols) > 0 and not handle_objects:
                            for col in non_numeric_cols:
                                prepare[col] = prepare[col].fillna(prepare[col].mode()[0] if not prepare[col].mode().empty else 'NA')
                            
                    elif missing_strategy == "Rellenar con mediana":
                        # Para columnas numéricas usar mediana
                        if len(numeric_cols) > 0:
                            prepare[numeric_cols] = prepare[numeric_cols].fillna(prepare[numeric_cols].median())
                        # Para columnas no numéricas sin valor especificado usar moda
                        if len(non_numeric_cols) > 0 and not handle_objects:
                            for col in non_numeric_cols:
                                prepare[col] = prepare[col].fillna(prepare[col].mode()[0] if not prepare[col].mode().empty else 'NA')
                            
                    else:  # Rellenar con moda
                         # Usar moda para todas las columnas que no son object o no tienen valor especificado
                        for col in prepare.columns:
                            if prepare[col].dtype in ['object']:
                                if handle_objects:
                                    continue  # Ya se manejaron las columnas object
                            mode_value = prepare[col].mode()
                            prepare[col] = prepare[col].fillna(mode_value[0] if not mode_value.empty else ('NA' if col in non_numeric_cols else 0))

                    # Actualizar el estado temporal después de la imputación
                    st.session_state.temp_prepared_data = prepare.copy()
                
                # Verificar los cambios
                nulls_after = prepare.isnull().sum().sum()
                values_filled = nulls_before - nulls_after
                
                if missing_strategy == "Eliminar filas":
                    st.success(f"Se eliminaron {values_filled} filas con valores faltantes")
                else:
                    st.success(f"Se rellenaron {values_filled} valores faltantes")
                
                # Verificar si quedan valores nulos
                remaining_nulls = prepare.isnull().sum()
                remaining_nulls = remaining_nulls[remaining_nulls > 0]
                
                if not remaining_nulls.empty:
                    st.error("⚠️ Error: Aún quedan valores faltantes en las siguientes columnas:")
                    for col in remaining_nulls.index:
                        st.write(f"- {col}: {remaining_nulls[col]} valores faltantes")
                    st.write("Por favor, contacta al equipo de desarrollo para revisar este error.")
    
                # Mostrar un resumen de los datos actualizados
                st.write("\n### Resumen después del procesamiento:")
                st.write(f"- Total de filas: {len(prepare)}")
                st.write(f"- Total de columnas: {len(prepare.columns)}")
                st.write(f"- Valores faltantes totales: {prepare.isnull().sum().sum()}")
                    
                # Verificación final de valores nulos
                final_null_check = prepare.isnull().sum().sum()
                if final_null_check == 0:
                    st.success("✅ ¡No quedan valores faltantes en el dataset!")
                else:
                    st.error(f"⚠️ Aún quedan {final_null_check} valores nulos en el dataset.")
                    return

                # Actualizar la sesión solo si no hay valores nulos
                if final_null_check == 0:
                    st.session_state.prepared_data = prepare.copy()
                    st.session_state.temp_prepared_data = prepare.copy()
                    # No sobrescribir 'er_data'
            
            except Exception as e:
                st.error(f"Error al procesar valores faltantes: {str(e)}")
                st.error("Detalles técnicos del error:")
                st.code(str(e))
        
    # Manejo de fechas
    st.subheader("Manejo de Fechas")
    with st.expander("Procesamiento de Fechas"):
        date_columns = st.multiselect(
            "Seleccionar columnas de fecha",
            prepare.columns,
            key="date_columns"
        )
        
        if date_columns:
            date_format = st.selectbox(
                "Formato de fecha",
                [
                    "yyyy-mm-dd", 
                    "dd-mm-yyyy", 
                    "mm-dd-yyyy", 
                    "yyyy-mm-dd hh:mm",
                    "dd-mm-yyyy hh:mm",
                    "mm-dd-yyyy hh:mm",
                    "yyyy-mm-dd hh:mm:ss",
                    "dd-mm-yyyy hh:mm:ss",
                    "mm-dd-yyyy hh:mm:ss",
                    "hh:mm",
                    "hh:mm:ss"
                ],
                help="Selecciona el formato que coincida con tus datos de fecha/hora."
            )
            
            time_format = st.radio(
                "Formato de hora",
                ["24 horas", "12 horas (AM/PM)"],
                help="Selecciona si el formato de hora está en 12 o 24 horas"
            )
            
            # Ajustar las características disponibles según el formato
            if date_format in ["hh:mm", "hh:mm:ss"]:
                available_features = ["Hora del día", "Periodo del día", "Minutos", "Segundos"]
            else:
                if "hh:mm:ss" in date_format:
                    available_features = [
                        "Año", "Mes", "Día", "Día de la semana", "Trimestre", "Estación", 
                        "Es fin de semana", "Hora del día", "Periodo del día", "Minutos", "Segundos"
                    ]
                else:
                    available_features = [
                        "Año", "Mes", "Día", "Día de la semana", "Trimestre", "Estación", 
                        "Es fin de semana", "Hora del día", "Periodo del día", "Minutos"
                    ]
            
            date_features = st.multiselect(
                "Seleccionar características a extraer",
                available_features
            )
            
            if st.button("Procesar fechas", key="process_dates_button"):
                for col in date_columns:
                    try:
                        if date_format in ["hh:mm", "hh:mm:ss"]:
                            # Procesar solo tiempo
                            if date_format == "hh:mm:ss":
                                time_parse_format = '%I:%M:%S %p' if time_format == "12 horas (AM/PM)" else '%H:%M:%S'
                                time_with_seconds = True
                            else:
                                time_parse_format = '%I:%M %p' if time_format == "12 horas (AM/PM)" else '%H:%M'
                                time_with_seconds = False
                            
                            def convert_time(time_str):
                                try:
                                    time_obj = datetime.strptime(time_str.strip(), time_parse_format)
                                    if time_with_seconds:
                                        return time_obj.hour, time_obj.minute, time_obj.second
                                    else:
                                        return time_obj.hour, time_obj.minute, None
                                except ValueError:
                                    st.warning(f"⚠️ Formato de hora inesperado en {col}: '{time_str}'")
                                    return None, None, None if time_with_seconds else None
                            
                            # Aplicar la conversión y crear nuevas columnas
                            hours_minutes_seconds = prepare[col].apply(convert_time)
                            
                            # Depuración: Mostrar una vista previa de la conversión
                            st.write(f"Vista previa de la conversión de tiempo para la columna {col}:")
                            st.write(hours_minutes_seconds.head())
                            
                            if "Hora del día" in date_features:
                                prepare[f'{col}_hora'] = hours_minutes_seconds.apply(lambda x: x[0] if x and x[0] is not None else None)
                            if "Minutos" in date_features:
                                prepare[f'{col}_minutos'] = hours_minutes_seconds.apply(lambda x: x[1] if x and x[1] is not None else None)
                            if "Segundos" in date_features and time_with_seconds:
                                prepare[f'{col}_segundos'] = hours_minutes_seconds.apply(lambda x: x[2] if x and x[2] is not None else None)
                            
                            # Agregar periodo del día si se seleccionó
                            if "Periodo del día" in date_features:
                                def get_period(hour):
                                    if hour is None:
                                        return None
                                    if 5 <= hour < 12:
                                        return 'Mañana'
                                    elif 12 <= hour < 17:
                                        return 'Tarde'
                                    elif 17 <= hour < 21:
                                        return 'Noche'
                                    else:
                                        return 'Madrugada'
                                prepare[f'{col}_periodo'] = prepare[f'{col}_hora'].apply(get_period)
                            
                        else:
                            # Definir el formato de parsing según la selección
                            if date_format == "yyyy-mm-dd":
                                date_parse_format = '%Y-%m-%d'
                            elif date_format == "dd-mm-yyyy":
                                date_parse_format = '%d-%m-%Y'
                            elif date_format == "mm-dd-yyyy":
                                date_parse_format = '%m-%d-%Y'
                            elif date_format == "yyyy-mm-dd hh:mm":
                                date_parse_format = '%Y-%m-%d %H:%M' if time_format == "24 horas" else '%Y-%m-%d %I:%M %p'
                            elif date_format == "dd-mm-yyyy hh:mm":
                                date_parse_format = '%d-%m-%Y %H:%M' if time_format == "24 horas" else '%d-%m-%Y %I:%M %p'
                            elif date_format == "mm-dd-yyyy hh:mm":
                                date_parse_format = '%m-%d-%Y %H:%M' if time_format == "24 horas" else '%m-%d-%Y %I:%M %p'
                            elif date_format == "yyyy-mm-dd hh:mm:ss":
                                date_parse_format = '%Y-%m-%d %H:%M:%S' if time_format == "24 horas" else '%Y-%m-%d %I:%M:%S %p'
                            elif date_format == "dd-mm-yyyy hh:mm:ss":
                                date_parse_format = '%d-%m-%Y %H:%M:%S' if time_format == "24 horas" else '%d-%m-%Y %I:%M:%S %p'
                            elif date_format == "mm-dd-yyyy hh:mm:ss":
                                date_parse_format = '%m-%d-%Y %H:%M:%S' if time_format == "24 horas" else '%m-%d-%Y %I:%M:%S %p'
                            else:
                                st.error(f"Formato de fecha no reconocido: {date_format}")
                                continue
                            
                            # Convertir a datetime con manejo de errores
                            temp_dates = pd.to_datetime(prepare[col], format=date_parse_format, errors='coerce')
                            
                            # Depuración: Mostrar una vista previa de las fechas parseadas
                            st.write(f"Vista previa de las fechas parseadas para la columna {col}:")
                            st.write(temp_dates.head())
                            
                            # Manejo de valores que no se pudieron parsear
                            if temp_dates.isnull().any():
                                st.warning(f"⚠️ Algunas fechas en la columna {col} no pudieron ser parseadas y se asignaron como NaT.")
                            
                            # Extraer características según selección
                            if "Año" in date_features:
                                prepare[f'{col}_año'] = temp_dates.dt.year
                            if "Mes" in date_features:
                                prepare[f'{col}_mes'] = temp_dates.dt.month
                            if "Día" in date_features:
                                prepare[f'{col}_dia'] = temp_dates.dt.day
                            if "Día de la semana" in date_features:
                                prepare[f'{col}_dia_semana'] = temp_dates.dt.dayofweek + 1
                            if "Trimestre" in date_features:
                                prepare[f'{col}_trimestre'] = temp_dates.dt.quarter
                            if "Es fin de semana" in date_features:
                                prepare[f'{col}_fin_semana'] = temp_dates.dt.dayofweek.isin([5, 6]).astype(int)
                            if "Estación" in date_features:
                                def get_season(month):
                                    if month in [12, 1, 2]:
                                        return 'Invierno'
                                    elif month in [3, 4, 5]:
                                        return 'Primavera'
                                    elif month in [6, 7, 8]:
                                        return 'Verano'
                                    else:
                                        return 'Otoño'
                                prepare[f'{col}_estacion'] = temp_dates.dt.month.apply(get_season)
                            if "Hora del día" in date_features and any(sub in date_format for sub in ["hh:mm", "hh:mm:ss"]):
                                prepare[f'{col}_hora'] = temp_dates.dt.hour
                            if "Minutos" in date_features and any(sub in date_format for sub in ["hh:mm", "hh:mm:ss"]):
                                prepare[f'{col}_minutos'] = temp_dates.dt.minute
                            if "Segundos" in date_features and "hh:mm:ss" in date_format:
                                prepare[f'{col}_segundos'] = temp_dates.dt.second
                            if "Periodo del día" in date_features and any(sub in date_format for sub in ["hh:mm", "hh:mm:ss"]):
                                def get_period(hour):
                                    if hour is None:
                                        return None
                                    if 5 <= hour < 12:
                                        return 'Mañana'
                                    elif 12 <= hour < 17:
                                        return 'Tarde'
                                    elif 17 <= hour < 21:
                                        return 'Noche'
                                    else:
                                        return 'Madrugada'
                                prepare[f'{col}_periodo'] = temp_dates.dt.hour.apply(get_period)
                        
                        # Eliminar la columna original de fecha
                        prepare = prepare.drop(columns=[col])
                        st.success(f"Columna {col} procesada exitosamente")
                    
                    except Exception as e:
                        st.error(f"Error procesando {col}: {str(e)}")
                        st.exception(e)
        
        # Actualizar el estado temporal después de procesar fechas
        st.session_state.temp_prepared_data = prepare.copy()
        
    # Codificación de variables categóricas
    st.subheader("Codificación de Variables Categóricas")
    categorical_columns = prepare.select_dtypes(include=['object']).columns
    
    if len(categorical_columns) > 0:
        encoding_method = st.radio(
            "Método de codificación:",
            ["Label Encoding", "One-Hot Encoding"]
        )
        
        cols_to_encode = st.multiselect(
            "Seleccionar columnas para codificar",
            categorical_columns,
            key="cols_to_encode"
        )
        
        if st.button("Aplicar codificación", key="apply_encoding_button"):
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                for col in cols_to_encode:
                    try:
                        prepare[col] = le.fit_transform(prepare[col].astype(str))
                        st.success(f"✅ Label Encoding aplicado a la columna '{col}'")
                    except Exception as e:
                        st.error(f"Error al codificar la columna {col} con Label Encoding: {str(e)}")
                # Actualizar el estado temporal después de la codificación
                st.session_state.temp_prepared_data = prepare.copy()
            else:  # One-Hot Encoding
                try:
                    prepare = pd.get_dummies(prepare, columns=cols_to_encode)
                    st.success("✅ One-Hot Encoding aplicado")
                    # Actualizar el estado temporal después de la codificación
                    st.session_state.temp_prepared_data = prepare.copy()
                except Exception as e:
                    st.error(f"Error al aplicar One-Hot Encoding: {str(e)}")
    
    # Normalización de variables numéricas
    st.subheader("Normalización de Variables Numéricas")
    numeric_columns = prepare.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_columns) > 0:
        cols_to_normalize = st.multiselect(
            "Seleccionar columnas para normalizar",
            numeric_columns,
            key="cols_to_normalize"
        )
        
        if cols_to_normalize and st.button("Aplicar normalización", key="apply_normalization_button"):
            try:
                scaler = StandardScaler()
                prepare[cols_to_normalize] = scaler.fit_transform(prepare[cols_to_normalize])
                st.success("✅ Normalización aplicada")
                # Actualizar el estado temporal después de la normalización
                st.session_state.temp_prepared_data = prepare.copy()
            except Exception as e:
                st.error(f"Error al aplicar normalización: {str(e)}")
    
    # Guardar datos preparados y mostrar matriz de correlación
    st.write("### Vista previa de los datos:")
    st.dataframe(prepare.head())
    
    # Información sobre valores nulos
    null_count = prepare.isnull().sum().sum()
    if null_count > 0:
        st.warning(f"⚠️ Hay {null_count} valores nulos en los datos.")
    else:
        st.success("✅ No hay valores nulos en los datos.")

    # Matriz de correlación
    st.subheader("Matriz de Correlación")
    numerical_columns = prepare.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numerical_columns) > 1:
        corr_variables = st.multiselect(
            "Selecciona las variables para incluir en la matriz de correlación",
            options=numerical_columns,
            default=numerical_columns[:min(5, len(numerical_columns))]  # Seleccionar hasta 5 columnas por defecto
        )
        
        if corr_variables:
            try:
                # Calcular y mostrar la matriz de correlación
                corr_matrix = prepare[corr_variables].corr(method='pearson')
                
                # Mapa de calor de correlación
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title='Matriz de Correlación de Pearson'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Botón de descarga
                csv_corr = corr_matrix.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="Descargar Matriz de Correlación como CSV",
                    data=csv_corr,
                    file_name='matriz_correlacion.csv',
                    mime='text/csv',
                )
                
                # Análisis de correlaciones significativas
                st.write("### Análisis de Correlaciones Significativas")
                threshold = st.slider(
                    "Selecciona el umbral mínimo de correlación para considerar significativa",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05
                )
                
                # Obtener y mostrar correlaciones significativas
                corr_pairs = corr_matrix.unstack()
                significant_corr = corr_pairs[
                    (abs(corr_pairs) >= threshold) & 
                    (abs(corr_pairs) < 1)
                ].drop_duplicates().sort_values(ascending=False)
                
                if not significant_corr.empty:
                    st.write(f"Correlaciones significativas (|correlación| ≥ {threshold}):")
                    for (var1, var2), corr_value in significant_corr.items():
                        st.write(f"- **{var1}** y **{var2}**: correlación de **{corr_value:.2f}**")
                else:
                    st.write("No se encontraron correlaciones significativas con el umbral seleccionado.")
                
            except Exception as e:
                st.error(f"Error al calcular la matriz de correlación: {str(e)}")
        else:
            st.warning("Por favor, selecciona al menos una variable para mostrar la matriz de correlación.")
    else:
        st.warning("No hay suficientes variables numéricas para calcular correlaciones.")

    # Botón de guardar al final
    if st.button("Guardar datos preparados", key="save_prepared_data_button"):
        try:
            null_count = prepare.isnull().sum().sum()
            if null_count == 0:
                st.session_state.prepared_data = prepare.copy()
                # Actualizar también los datos temporales
                st.session_state.temp_prepared_data = prepare.copy()
                st.success("✅ Datos preparados guardados exitosamente")
            else:
                st.error(f"❌ No se pueden guardar los datos. Aún hay {null_count} valores nulos.")
                st.warning("Por favor, aplica una estrategia de manejo de valores faltantes antes de guardar.")
        except Exception as e:
            st.error(f"Error al guardar los datos preparados: {str(e)}")
    st.info("👆 No te olvides de guardar los datos preparados antes de continuar con el análisis en la página Training o Test.")
