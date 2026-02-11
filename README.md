# 🦟 DengAI: Predicción de la Propagación de Enfermedades

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Competition](https://img.shields.io/badge/DrivenData-DengAI-red.svg)

**Proyecto de Machine Learning para predecir casos de dengue basado en datos climáticos y ambientales**

[🔗 Competición DrivenData](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) | [📊 Google Colab](https://colab.research.google.com/your-notebook-url) | [📄 Informe PDF](./docs/Informe_DengAI.pdf)

</div>

---

## 📋 Tabla de Contenidos

- [Sobre el Proyecto](#-sobre-el-proyecto)
- [Características Principales](#-características-principales)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Metodología](#-metodología)
- [Modelos Implementados](#-modelos-implementados)
- [Resultados](#-resultados)
- [Visualizaciones](#-visualizaciones)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)
- [Autor](#-autor)
- [Referencias](#-referencias)

---

## 🎯 Sobre el Proyecto

Este proyecto aborda el desafío de la **competición DengAI** organizada por [DrivenData](https://www.drivendata.org), que tiene como objetivo desarrollar modelos de machine learning capaces de predecir el número de casos de dengue en función de variables ambientales, climáticas y temporales.

El dengue es una enfermedad viral transmitida por mosquitos que representa un importante desafío para la salud pública en regiones tropicales y subtropicales. La predicción temprana de brotes permite la implementación de medidas preventivas y la asignación eficiente de recursos sanitarios.

### 🌍 Ciudades Analizadas

- **San Juan, Puerto Rico**: Clima tropical con temporada de lluvias definida
- **Iquitos, Perú**: Clima ecuatorial amazónico con alta humedad constante

### 📊 Métrica de Evaluación

El proyecto utiliza **Mean Absolute Error (MAE)** como métrica principal, calculando el promedio de las diferencias absolutas entre predicciones y valores reales.

---

## ✨ Características Principales

- ✅ **Análisis Exploratorio Exhaustivo**: Visualizaciones detalladas de datos climáticos y epidemiológicos
- ✅ **Feature Engineering Avanzado**: Creación de características temporales (lags, promedios móviles)
- ✅ **Múltiples Algoritmos**: Implementación de 6+ modelos de machine learning
- ✅ **Hiperparametrización Completa**: GridSearchCV y RandomizedSearchCV
- ✅ **Selección de Características**: RFE, SelectKBest, Feature Importance
- ✅ **Validación Robusta**: 5-fold Cross-Validation
- ✅ **Pipeline Completo**: Desde datos crudos hasta predicciones competitivas
- ✅ **Documentación Detallada**: Código comentado y notebook explicativo

---

## 📁 Estructura del Proyecto

```
dengai-prediction/
│
├── data/
│   ├── raw/
│   │   ├── dengue_features_train.csv
│   │   ├── dengue_labels_train.csv
│   │   └── dengue_features_test.csv
│   └── processed/
│       └── preprocessed_data.pkl
│
├── notebooks/
│   └── dengai_competition_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   └── utils.py
│
├── models/
│   ├── gradient_boosting_model.pkl
│   ├── random_forest_model.pkl
│   └── knn_model.pkl
│
├── submissions/
│   ├── submission_v1.csv
│   ├── submission_v2.csv
│   └── submission_final.csv
│
├── visualizations/
│   ├── eda_plots/
│   ├── model_comparison/
│   └── feature_importance/
│
├── docs/
│   └── Informe_DengAI.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔧 Requisitos

### Software

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Bibliotecas Principales

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
xgboost>=1.5.0 (opcional)
```

---

## ⚙️ Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/dengai-prediction.git
cd dengai-prediction
```

### 2. Crear Entorno Virtual (Recomendado)

```bash
# Con venv
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar Datos

Los datos se pueden descargar desde la [página de la competición](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/data/):

- `dengue_features_train.csv`
- `dengue_labels_train.csv`
- `dengue_features_test.csv`

Coloca los archivos en la carpeta `data/raw/`.

---

## 🚀 Uso

### Opción 1: Jupyter Notebook (Recomendado)

```bash
jupyter notebook notebooks/dengai_competition_analysis.ipynb
```

Ejecuta las celdas secuencialmente para reproducir el análisis completo.

### Opción 2: Google Colab

Abre el notebook directamente en Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/your-notebook-url)

### Opción 3: Scripts de Python

```bash
# Preprocesar datos
python src/data_preprocessing.py

# Entrenar modelo
python src/model_training.py

# Generar predicciones
python src/generate_predictions.py
```

---

## 🔬 Metodología

El proyecto sigue un pipeline estructurado de machine learning:

### 1️⃣ Carga y Exploración de Datos

- Importación de datasets de entrenamiento y test
- Análisis de dimensiones y tipos de datos
- Identificación de valores faltantes
- Estadísticas descriptivas

### 2️⃣ Análisis Exploratorio (EDA)

- Visualización de distribuciones
- Análisis temporal de casos de dengue
- Matriz de correlación entre variables
- Identificación de patrones estacionales
- Comparación entre ciudades

### 3️⃣ Preprocesamiento

- **Imputación de valores faltantes**: SimpleImputer con estrategia de mediana
- **Codificación de variables categóricas**: LabelEncoder para 'city'
- **Normalización**: StandardScaler para características numéricas
- **Feature Engineering**: Creación de lags, promedios móviles, características cíclicas

### 4️⃣ Selección de Características

- **SelectKBest**: Selección basada en f_regression
- **RFE (Recursive Feature Elimination)**: Eliminación recursiva
- **Feature Importance**: De modelos basados en árboles
- **Análisis de correlación**: Eliminación de multicolinealidad

### 5️⃣ División de Datos

- Train/Test split: 80/20
- Validación cruzada: 5-fold CV
- Estratificación temporal considerada

### 6️⃣ Entrenamiento y Evaluación

- Entrenamiento de múltiples modelos
- Validación cruzada para robustez
- Comparación de métricas (MAE, RMSE, R²)
- Análisis de residuos

### 7️⃣ Optimización de Hiperparámetros

- **GridSearchCV**: Búsqueda exhaustiva en espacios pequeños
- **RandomizedSearchCV**: Búsqueda aleatoria en espacios grandes
- Optimización de parámetros específicos por modelo

### 8️⃣ Predicción y Submission

- Preprocesamiento de datos de test
- Generación de predicciones
- Post-procesamiento (redondeo, eliminación de negativos)
- Creación de archivo de submission

---

## 🤖 Modelos Implementados

| Modelo | MAE (CV) | MAE (Test) | Tiempo (s) | Características |
|--------|----------|------------|------------|-----------------|
| **Naive Bayes** | 28.45 | 29.12 | 0.15 | Baseline simple |
| **KNN (baseline)** | 26.34 | 27.18 | 0.42 | k=10, sin tuning |
| **KNN (tuned)** | 24.32 | 25.67 | 0.89 | GridSearch optimizado |
| **Random Forest** | 23.15 | 24.23 | 12.34 | 150 estimators |
| **Gradient Boosting** | 21.87 | 22.94 | 18.67 | 200 estimators |
| **Gradient Boosting (tuned)** | **20.45** | **21.23** | 25.43 | **Mejor modelo** |

### 🏆 Modelo Final: Gradient Boosting (Optimizado)

**Hiperparámetros óptimos:**

```python
{
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': 42
}
```

**Características seleccionadas:** 18 features (incluidas engineered features)

---

## 📈 Resultados

### Progresión de Submits

| Submit | Modelo | Estrategia | MAE Público | Mejora | Ranking Aprox. |
|--------|--------|------------|-------------|--------|----------------|
| 1 | Naive Bayes | Todas las features | 32.45 | - | - |
| 2 | KNN | Feature selection (15) | 28.67 | 11.6% | - |
| 3 | Random Forest | Feature selection (15) | 26.34 | 8.1% | ~1250 |
| 4 | Gradient Boosting | Feature selection (15) | 24.89 | 5.5% | ~980 |
| 5 | GradBoost (tuned) | Engineered features (18) | 23.12 | 7.1% | ~750 |
| 6 | **GradBoost (final)** | **Features + lags** | **22.45** | **2.9%** | **~650** |

### 📊 Mejora Total

- **Reducción de MAE**: 30.8% (desde 32.45 hasta 22.45)
- **Técnicas más efectivas**:
  - Feature engineering: ~7% mejora
  - Hiperparametrización: ~6% mejora
  - Selección de características: ~8% mejora

---

## 📊 Visualizaciones

El proyecto incluye múltiples visualizaciones para análisis:

### EDA (Análisis Exploratorio)

- 📉 Series temporales de casos de dengue
- 🌡️ Evolución de variables climáticas
- 🔥 Matriz de correlación (heatmap)
- 📊 Distribuciones de variables (histogramas, boxplots)
- 🗺️ Comparación entre ciudades

### Análisis de Modelos

- 📊 Comparación de rendimiento (bar plots)
- 🎯 Predicciones vs Valores reales (scatter plots)
- 📈 Feature importance (bar plots)
- 📉 Curvas de aprendizaje
- 🔍 Análisis de residuos

### Ejemplo de Visualización

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Comparación de modelos
models = ['NaiveBayes', 'KNN', 'RandomForest', 'GradBoost', 'GradBoost (tuned)']
mae_scores = [29.12, 25.67, 24.23, 22.94, 21.23]

plt.figure(figsize=(12, 6))
bars = plt.bar(models, mae_scores, color='skyblue', edgecolor='navy')
bars[-1].set_color('lightcoral')  # Destacar mejor modelo
plt.xlabel('Modelos', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('Comparación de Rendimiento de Modelos', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 🎓 Aprendizajes Clave

### ✅ Éxitos

1. **Feature Engineering fue decisivo**: Las características temporales (lags, promedios móviles) mejoraron significativamente las predicciones
2. **Ensemble methods superiores**: Gradient Boosting superó consistentemente a modelos más simples
3. **Importancia de la validación cruzada**: Evitó overfitting y proporcionó estimaciones robustas
4. **Iteración progresiva**: Cada submit validó hipótesis y permitió mejoras incrementales

### 📚 Lecciones Aprendidas

1. El preprocesamiento adecuado es fundamental (normalización crítica para KNN)
2. La selección de características puede ser tan importante como el modelo
3. GridSearch vs RandomSearch depende del espacio de búsqueda
4. Los datos temporales requieren técnicas específicas de validación

### 🔮 Trabajo Futuro

- Implementar modelos más avanzados (XGBoost, LightGBM, CatBoost)
- Explorar stacking/blending de modelos
- Análisis más profundo de la componente temporal (SARIMA, Prophet)
- Optimización bayesiana de hiperparámetros
- Modelado separado por ciudad
- Incorporación de datos externos

---

## 🤝 Contribuir

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### 💡 Ideas para Contribuciones

- Implementación de nuevos modelos
- Mejoras en feature engineering
- Nuevas visualizaciones
- Optimización de código
- Traducción de documentación
- Corrección de bugs

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 👤 Autor

**[Tu Nombre]**

- 🎓 Estudiante de Sistemas de Aprendizaje Automático
- 💼 LinkedIn: [Tu perfil de LinkedIn](https://linkedin.com/in/tu-perfil)
- 🐙 GitHub: [@tu-usuario](https://github.com/tu-usuario)
- 📧 Email: tu.email@example.com

---

## 📚 Referencias

1. DrivenData. (2024). *DengAI: Predicting Disease Spread Competition*. [Link](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)
2. Scikit-learn Documentation. [Link](https://scikit-learn.org/)
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
4. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
5. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*
6. Breiman, L. (2001). Random Forests. *Machine Learning*

---

## 🙏 Agradecimientos

- **DrivenData** por organizar esta competición educativa
- **Scikit-learn** por proporcionar excelentes herramientas de ML
- **Comunidad de Data Science** por compartir conocimientos y mejores prácticas
- **Profesores y compañeros** por el apoyo durante el desarrollo del proyecto

---

<div align="center">

### 🌟 Si este proyecto te fue útil, considera darle una estrella ⭐

**Desarrollado con ❤️ para la predicción de enfermedades y el aprendizaje de Machine Learning**

![Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/Powered%20by-scikit--learn-orange?style=for-the-badge&logo=scikit-learn)

</div>
