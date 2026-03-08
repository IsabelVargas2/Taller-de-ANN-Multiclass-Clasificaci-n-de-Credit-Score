# 💳 Taller ANN Multiclass — Clasificación de Credit Score

Aplicación de Red Neuronal Artificial (ANN) multiclase para predecir el puntaje crediticio de un cliente en tres categorías: **Poor**, **Standard** y **Good**.

## 🔗 Enlaces

- **Aplicación Streamlit:** [https://taller-de-ann-multiclass-clasificaci-n-de-credit-score-ssreefs.streamlit.app](https://taller-de-ann-multiclass-clasificaci-n-de-credit-score-ssreefs.streamlit.app)
- **Repositorio GitHub:** [https://github.com/IsabelVargas2/Taller-de-ANN-Multiclass-Clasificaci-n-de-Credit-Score](https://github.com/IsabelVargas2/Taller-de-ANN-Multiclass-Clasificaci-n-de-Credit-Score)

## 📋 Descripción

Este proyecto implementa una ANN multiclase entrenada sobre el dataset de riesgo crediticio, con las siguientes etapas:

- Preprocesamiento: limpieza, imputación, Label Encoding
- Reducción de dimensionalidad con PCA (95% de varianza → 13 componentes)
- Red neuronal con capas Dense + BatchNormalization + Dropout
- Evaluación con accuracy, reporte de clasificación y matriz de confusión
- Despliegue en Streamlit Cloud

## 🏗️ Arquitectura del Modelo

```
Entrada (13 componentes PCA)
→ Dense(128) + BatchNorm + Dropout(0.3)
→ Dense(64)  + BatchNorm + Dropout(0.2)
→ Dense(32)  + Dropout(0.2)
→ Dense(3, softmax)  ← Poor / Standard / Good
```

## 📁 Archivos

| Archivo | Descripción |
|---|---|
| `Taller_ANN_Multiclass_CreditScore.ipynb` | Notebook con entrenamiento completo |
| `app.py` | Aplicación Streamlit |
| `ann_credit_score.keras` | Modelo entrenado |
| `scaler.pkl` | StandardScaler ajustado |
| `pca.pkl` | PCA ajustado (13 componentes) |
| `requirements.txt` | Dependencias del proyecto |

## 🎯 Clases

| Clase | Etiqueta | Descripción |
|---|---|---|
| 0 | 🔴 Poor | Alto riesgo crediticio |
| 1 | 🟡 Standard | Riesgo crediticio medio |
| 2 | 🟢 Good | Bajo riesgo crediticio |

## 🛠️ Tecnologías

- Python 3.11
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- Pandas / NumPy
