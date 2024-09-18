# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregando o dataset
data_path = 'C:\\Users\\pedro\\Downloads\\Mina_ou_Rocha\\Base_Dados\\connectionist\\sonar.all-data'

data = pd.read_csv(data_path, header=None)

# Análise Exploratória de Dados (EDA)
print("Primeiras linhas do dataset:")
print(data.head())
print("\nDescrição do dataset:")
print(data.describe())

# Visualização da distribuição das classes
plt.figure(figsize=(6, 4))
sns.countplot(data[60])
plt.title('Distribuição das Classes')
plt.show()

# Visualização da Matriz de Correlação
corr_matrix = data.iloc[:, :-1].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação das Features')
plt.show()

# Pré-processamento de Dados
# Verificando dados faltantes
print("Dados faltantes em cada coluna:")
print(data.isnull().sum())

# Normalização das features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.iloc[:, :-1])

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data[60], test_size=0.3, random_state=42, stratify=data[60])

# Seleção de Hiperparâmetros
# Grid Search para Acurácia
param_grid = {'n_neighbors': range(1, 21)}
grid_search_acc = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search_acc.fit(X_train, y_train)
print("Melhor número de vizinhos para acurácia:", grid_search_acc.best_params_)

# Grid Search para F1-score
grid_search_f1 = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_macro')
grid_search_f1.fit(X_train, y_train)
print("Melhor número de vizinhos para F1-score:", grid_search_f1.best_params_)

# Treinamento do Modelo
knn_best_acc = KNeighborsClassifier(n_neighbors=grid_search_acc.best_params_['n_neighbors'])
knn_best_acc.fit(X_train, y_train)
knn_best_f1 = KNeighborsClassifier(n_neighbors=grid_search_f1.best_params_['n_neighbors'])
knn_best_f1.fit(X_train, y_train)

# Avaliação do Modelo
y_pred_acc = knn_best_acc.predict(X_test)
y_pred_f1 = knn_best_f1.predict(X_test)
print("Relatório de Classificação para o Modelo Otimizado para Acurácia:")
print(classification_report(y_test, y_pred_acc))
print("Matriz de Confusão para Acurácia:")
print(confusion_matrix(y_test, y_pred_acc))

print("\nRelatório de Classificação para o Modelo Otimizado para F1-Score:")
print(classification_report(y_test, y_pred_f1))
print("Matriz de Confusão para F1-Score:")
print(confusion_matrix(y_test, y_pred_f1))
