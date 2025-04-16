# %% [markdown]
# # Proyecto de regresión líneal

# %% [markdown]
# ## Cargar dataframe

# %%
# Importar librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# %%
# Leer el archivo CSV desde la URL y cargarlo en un DataFrame

df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")

df.head()

# %% [markdown]
# **Columna Descripción**
# 
# - `age`: Edad del beneficiario principal (numérico).
# - `sex`: Género del beneficiario principal (categórico).
# - `bmi`: índice de masa corporal (numérico).
# - `children`: Número de niños/dependientes cubiertos por un seguro médico (numérico).
# - `smoker`: ¿Es fumador? (categórico).
# - `region`: Área residencial del beneficiario en USA: noreste, sureste, suroeste, noroeste (categórico).
# - `charges`: Prima del seguro médico (numérico).
# 

# %%
# Guardar el dataframe en un archivo CSV en la carpeta raw

df.to_csv("/workspaces/efrainnalmeida-proyecto-regresion-lineal-final/data/raw/medical_insurance_cost.csv", index=False)

# %%
# Extraer de la carpeta raw el archivo CSV y cargarlo en un DataFrame

df_interim = pd.read_csv("/workspaces/efrainnalmeida-proyecto-regresion-lineal-final/data/raw/medical_insurance_cost.csv")

df_interim.head()

# %%
# Guardar el dataframe en un archivo CSV en la carpeta interim

df_interim.to_csv("/workspaces/efrainnalmeida-proyecto-regresion-lineal-final/data/interim/medical_insurance_cost.csv", index=False)

# %% [markdown]
# ## EDA

# %% [markdown]
# ### Información general

# %%
# Información general del DataFrame

print("\nInformación general:")
df_interim.info()

# %%
# Estadísticas descriptivas

print("\nEstadísticas descriptivas:")
df_interim.describe(include='all')

# %% [markdown]
# ### Análisis de valores faltantes

# %%
# Comprobación de valores nulos

print("\nValores nulos por columna:")
df_interim.isnull().sum()

# %% [markdown]
# ### Análisis de valores duplicados

# %%
# Detección de registros duplicados

print("\nNúmero de registros duplicados:")
df_interim.duplicated().sum()

# %%
# Eliminar el registro duplicado

df_interim = df_interim.drop_duplicates().reset_index(drop=True)

print("\nNúmero de registros duplicados después de eliminar:")
df_interim.duplicated().sum()

# %%
# Guardar el dataframe en un archivo CSV en la carpeta interim

df_interim.to_csv("/workspaces/efrainnalmeida-proyecto-regresion-lineal-final/data/interim/medical_insurance_cost.csv", index=False)

df_interim.head()

# %%
# Información general del DataFrame

print("\nInformación general:")
df_interim.info()

# %% [markdown]
# ### Análisis de outliers

# %%
# Outliers

# Lista de variables numéricas
numeric_vars = ["age", "bmi", "children", "charges"]

# Crear boxplots
plt.figure(figsize=(14, 10))
for i, var in enumerate(numeric_vars, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df_interim[var])
    plt.title(f"Boxplot de {var}")
plt.tight_layout()
plt.show()

# %%
# Crear histograma logarítmico para charges por fumador/no fumador

plt.figure(figsize=(10,6))
sns.histplot(data=df_interim, x=np.log(df_interim["charges"]), hue="smoker", kde=True, bins=50)
plt.title("Distribución logarítmica de charges según hábito de fumar")
plt.xlabel("Log(charges)")
plt.ylabel("Frecuencia")
plt.show()

# %% [markdown]
# ### Escalado de valores

# %%
# Seleccionar variables categóricas (tipo object)
categorical_vars = df_interim.select_dtypes(include=["object"]).columns.tolist()

# Seleccionar variables numéricas (excluyendo la variable original y la logarítmica de charges)
numeric_vars = df_interim.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_vars = [var for var in numeric_vars if var not in ["charges", "log_charges"]]

# Mostrar resultados
print("Variables categóricas:", categorical_vars)
print("Variables numéricas:", numeric_vars)

# %%
# Aplicar One-Hot Encoding a las variables categóricas
df_encoded = pd.get_dummies(df_interim, columns=categorical_vars, drop_first=True)

# Verificar las nuevas columnas
df_encoded.head()

# %%
# Separar variables predictoras (X) y variable objetivo (y)

X = df_encoded.drop(columns=["charges"])
y = df_encoded["charges"]

# Dividir en conjunto de entrenamiento y prueba (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar las dimensiones
print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de entrenamiento:", y_train.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)
print("Tamaño del conjunto de prueba:", y_test.shape)

# %%
# Instanciar el escalador
scaler = MinMaxScaler()

# Ajustar sobre train y transformar ambos datasets
X_train[numeric_vars] = scaler.fit_transform(X_train[numeric_vars])
X_test[numeric_vars] = scaler.transform(X_test[numeric_vars])

# %% [markdown]
# ### Selección de características

# %%
# Selección de las mejores k características (por ejemplo, 10)
k = 4  # puedes ajustar este número

# Inicializar selector con f_regression
selector = SelectKBest(score_func=f_regression, k=k)

# Ajustar selector sobre X_train y y_train
X_train_selected = selector.fit_transform(X_train, y_train)

# Aplicar la misma transformación a X_test
X_test_selected = selector.transform(X_test)

# Obtener nombres de las columnas seleccionadas
selected_features = X_train.columns[selector.get_support()].tolist()

# Mostrar las características seleccionadas
print("Características seleccionadas:")
for feature in selected_features:
    print("-", feature)

# %%
# Mostrar puntajes de cada variable
feature_scores = pd.DataFrame({
    "Feature": X_train.columns,
    "Score": selector.scores_
}).sort_values(by="Score", ascending=False)

print(feature_scores)

# %%
# Crear DataFrames alineados con las variables seleccionadas

X_train_kbest = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
X_test_kbest = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

# %%
# Añadir la variable objetivo

X_train_kbest["charges"] = y_train
X_test_kbest["charges"] = y_test

# %%
# Verificar

print("X_train_kbest:")
print(X_train_kbest.head())
print("\nX_test_kbest:")
print(X_test_kbest.head())

# %% [markdown]
# ### Guardar los datos limpios

# %%
X_train_kbest.to_csv("/workspaces/efrainnalmeida-proyecto-regresion-lineal-final/data/processed/clean_train.csv", index=False)
X_test_kbest.to_csv("/workspaces/efrainnalmeida-proyecto-regresion-lineal-final/data/processed/clean_test.csv", index=False)

# %% [markdown]
# ## Modelo de regresión líneal

# %% [markdown]
# ### Cargar clean_train.csv y clean_test.csv como df_train y df_test

# %%
# Leer los archivos CSV desde la carpeta processed y cargarlos en DataFrames

df_train = pd.read_csv("/workspaces/efrainnalmeida-proyecto-regresion-lineal-final/data/processed/clean_train.csv")
df_test = pd.read_csv("/workspaces/efrainnalmeida-proyecto-regresion-lineal-final/data/processed/clean_test.csv")

# %%
# Verificar el DataFrame de entrenamiento

print(df_train.shape)
df_train.head()


# %%
# Verificar el DataFrame de prueba

print(df_test.shape)
df_test.head()

# %% [markdown]
# ### Línea de regresión y Heatmap de correlación

# %% [markdown]
# #### Concatenar df_train y df_test

# %%
# Concatenar ambos DataFrames
df_total = pd.concat([df_train, df_test], ignore_index=True)

# Verificamos que charges esté presente
print(df_total.shape)
df_total.head()

# %% [markdown]
# #### Regplots: Variables independientes vs. `charges`

# %%
# Variables independientes
independent_vars = ["age", "bmi", "children", "smoker_yes"]

# Crear regplots
plt.figure(figsize=(16, 12))
for i, var in enumerate(independent_vars, 1):
    plt.subplot(2, 2, i)
    sns.regplot(x=var, y="charges", data=df_total, scatter_kws={"alpha":0.4})
    plt.title(f"Regplot: {var} vs. charges")
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Heatmap de correlación

# %%
# Variables para la matriz de correlación
corr_vars = independent_vars + ["charges"]

# Calcular matriz de correlación
correlation_matrix = df_total[corr_vars].corr()

# Graficar heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación con charges")
plt.show()

# %% [markdown]
# #### Establecer X_train, y_train, X_test, y_test

# %%
X_train = df_train.drop(["charges"], axis = 1)
y_train = df_train["charges"]
X_test = df_test.drop(["charges"], axis = 1)
y_test = df_test["charges"]

# %% [markdown]
# #### Modelo

# %% [markdown]
# ##### Entrenar el modelo

# %%
# Instanciar el modelo
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# %% [markdown]
# ##### Hacer predicciones

# %%
# Predicciones sobre el conjunto de prueba

y_pred = model.predict(X_test)

# %% [markdown]
# ##### Evaluar el modelo

# %%
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# %% [markdown]
# ##### Gráfico de dispersión `y_test` vs. `y_pred`

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # línea de referencia
plt.xlabel("Valor real (charges)")
plt.ylabel("Predicción (charges)")
plt.title("y_test vs y_pred")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ##### Coeficientes del modelo

# %%
# Obtener nombres de las variables
features = X_train.columns

# Obtener coeficientes
coefficients = model.coef_

# Crear DataFrame para visualizar
coef_df = pd.DataFrame({
    "Variable": features,
    "Coeficiente": coefficients
})

# Ordenar por magnitud del coeficiente
coef_df = coef_df.sort_values(by="Coeficiente", key=abs, ascending=False)

print(coef_df)

# %% [markdown]
# | Variable     | Coeficiente   | Interpretación |
# |--------------|---------------|----------------|
# | `smoker_yes` | 23,042.51     | Si una persona **fuma**, el modelo predice que pagará en promedio **23,043 USD más** por su seguro médico que una persona no fumadora, manteniendo las demás variables constantes. Es la variable más influyente. |
# | `age`        | 11,462.80     | Por cada **año adicional** de edad, el costo del seguro aumenta en promedio **11,463 USD**, si todo lo demás se mantiene constante. |
# | `bmi`        | 11,346.79     | Por cada **unidad de IMC (índice de masa corporal)** adicional, el costo del seguro aumenta en promedio **11,347 USD**, lo cual muestra un efecto importante del sobrepeso/obesidad. |
# | `children`   | 2,689.86      | Cada hijo adicional está asociado con un incremento promedio de **2,690 USD** en el seguro médico. Es la variable con menor impacto directo. |
# 

# %% [markdown]
# ## Hiperparametrización

# %% [markdown]
# ### Ajustar Ridge con GridSearchCV

# %%
# Definir el modelo base
ridge = Ridge()

# Definir los valores de alpha a evaluar
param_grid = {"alpha": [0.1, 1, 10, 100, 200, 500]}

# Configurar la búsqueda con validación cruzada
grid_search = GridSearchCV(
    estimator=ridge,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

# Ejecutar la búsqueda en los datos de entrenamiento
grid_search.fit(X_train, y_train)

# %% [markdown]
# ### Resultados: Mejor modelo y su rendimiento

# %%
# Mejor valor de alpha encontrado
best_alpha = grid_search.best_params_["alpha"]
print(f"Mejor alpha: {best_alpha}")

# Mejor score (negativo MSE)
best_score = grid_search.best_score_
print(f"Mejor MSE promedio (negativo): {best_score:.2f}")


# %% [markdown]
# ### Evaluar el modelo final en el conjunto de prueba

# %%
# Usar el mejor modelo ajustado
best_ridge_model = grid_search.best_estimator_

# Predicciones
y_pred_best_ridge = best_ridge_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_best_ridge)
mse = mean_squared_error(y_test, y_pred_best_ridge)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_best_ridge)

print(f"\nEvaluación del mejor modelo Ridge:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# %% [markdown]
# ### Coeficientes del modelo

# %%
# Obtener nombres de las variables
features = X_train.columns

# Obtener coeficientes del mejor modelo Ridge
coefficients = best_ridge_model.coef_

# Crear DataFrame para visualización
ridge_coef_df = pd.DataFrame({
    "Variable": features,
    "Coeficiente": coefficients
})

# Ordenar por valor absoluto del coeficiente
ridge_coef_df = ridge_coef_df.sort_values(by="Coeficiente", key=abs, ascending=False)

# Mostrar
print(ridge_coef_df)


