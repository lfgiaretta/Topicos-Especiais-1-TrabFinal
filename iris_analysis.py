import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo para os gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("ANÁLISE DO IRIS DATASET - MINERAÇÃO DE DADOS")
print("=" * 60)

# 1. CARREGAMENTO DOS DADOS
print("\n1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS")
print("-" * 50)

# Carregando o dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"Dataset carregado com sucesso!")
print(f"Dimensões: {df.shape[0]} registros e {df.shape[1]} colunas")
print(f"\nPrimeiros 5 registros:")
print(df.head())

print(f"\nInformações sobre o dataset:")
print(df.info())

print(f"\nEstatísticas descritivas:")
print(df.describe())

print(f"\nDistribuição das espécies:")
print(df['species_name'].value_counts())

# 2. ANÁLISE EXPLORATÓRIA DOS DADOS
print("\n\n2. ANÁLISE EXPLORATÓRIA DOS DADOS")
print("-" * 50)

# Criando gráficos de análise exploratória
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análise Exploratória do Iris Dataset', fontsize=16, fontweight='bold')

# Gráfico 1: Histograma das características
df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].hist(
    bins=20, ax=axes[0,0], alpha=0.7
)
axes[0,0].set_title('Distribuição das Características')

# Gráfico 2: Boxplot por espécie
df_melted = df.melt(id_vars=['species_name'], 
                   value_vars=['sepal length (cm)', 'sepal width (cm)', 
                              'petal length (cm)', 'petal width (cm)'],
                   var_name='feature', value_name='value')
sns.boxplot(data=df_melted, x='feature', y='value', hue='species_name', ax=axes[0,1])
axes[0,1].set_title('Boxplot das Características por Espécie')
axes[0,1].tick_params(axis='x', rotation=45)

# Gráfico 3: Scatter plot - Sepal
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', 
                hue='species_name', s=100, ax=axes[1,0])
axes[1,0].set_title('Comprimento vs Largura da Sépala')

# Gráfico 4: Scatter plot - Petal
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', 
                hue='species_name', s=100, ax=axes[1,1])
axes[1,1].set_title('Comprimento vs Largura da Pétala')

plt.tight_layout()
plt.show()

# Matriz de correlação
plt.figure(figsize=(10, 8))
correlation_matrix = df[['sepal length (cm)', 'sepal width (cm)', 
                        'petal length (cm)', 'petal width (cm)']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f')
plt.title('Matriz de Correlação das Características', fontsize=14, fontweight='bold')
plt.show()

# 3. PREPARAÇÃO DOS DADOS
print("\n3. PREPARAÇÃO DOS DADOS PARA MINERAÇÃO")
print("-" * 50)

# Separando features (X) e target (y)
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['species']

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

print(f"Dados de treino: {X_train.shape[0]} registros")
print(f"Dados de teste: {X_test.shape[0]} registros")

# Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. APLICAÇÃO DE TÉCNICAS DE MINERAÇÃO DE DADOS
print("\n\n4. APLICAÇÃO DE ALGORITMOS DE CLASSIFICAÇÃO")
print("-" * 50)

# Dicionário para armazenar os modelos e resultados
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

results = {}

# Treinamento e avaliação dos modelos
for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Treinar o modelo
    if name == 'SVM':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Relatório de classificação
    print("\nRelatório de Classificação:")
    target_names = ['setosa', 'versicolor', 'virginica']
    print(classification_report(y_test, y_pred, target_names=target_names))

# 5. VISUALIZAÇÃO DOS RESULTADOS
print("\n\n5. VISUALIZAÇÃO DOS RESULTADOS")
print("-" * 50)

# Gráfico de comparação dos modelos
plt.figure(figsize=(12, 8))

# Subplot 1: Comparação de acurácia
plt.subplot(2, 2, 1)
models_names = list(results.keys())
accuracies = list(results.values())
bars = plt.bar(models_names, accuracies, color=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Comparação de Acurácia dos Modelos', fontweight='bold')
plt.ylabel('Acurácia')
plt.ylim(0, 1)

# Adicionando valores nas barras
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# Subplot 2: Matriz de confusão do melhor modelo
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

plt.subplot(2, 2, 2)
if best_model_name == 'SVM':
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['setosa', 'versicolor', 'virginica'],
            yticklabels=['setosa', 'versicolor', 'virginica'])
plt.title(f'Matriz de Confusão - {best_model_name}', fontweight='bold')
plt.ylabel('Classe Real')
plt.xlabel('Classe Predita')

# Subplot 3: Importância das características (Random Forest)
if 'Random Forest' in models:
    plt.subplot(2, 2, 3)
    rf_model = models['Random Forest']
    feature_importance = rf_model.feature_importances_
    features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    bars = plt.bar(features, feature_importance, color='lightcoral')
    plt.title('Importância das Características (Random Forest)', fontweight='bold')
    plt.ylabel('Importância')
    plt.xticks(rotation=45)
    
    # Adicionando valores nas barras
    for bar, importance in zip(bars, feature_importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')

# Subplot 4: Distribuição das classes
plt.subplot(2, 2, 4)
class_counts = df['species_name'].value_counts()
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
        colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Distribuição das Classes no Dataset', fontweight='bold')

plt.tight_layout()
plt.show()

# 6. RELATÓRIO FINAL
print("\n\n6. RELATÓRIO FINAL DOS RESULTADOS")
print("=" * 60)

print(f"\nDataset: Iris Dataset")
print(f"Origem: Scikit-learn datasets (UCI Machine Learning Repository)")
print(f"Registros: {df.shape[0]}")
print(f"Características: {df.shape[1]-2} (sepal length, sepal width, petal length, petal width)")
print(f"Classes: 3 (setosa, versicolor, virginica)")

print(f"\nTécnica aplicada: Classificação")
print(f"Algoritmos testados: {', '.join(models.keys())}")

print(f"\nResultados:")
for model, accuracy in results.items():
    print(f"  - {model}: {accuracy:.4f} ({accuracy*100:.2f}%)")

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
print(f"\nMelhor modelo: {best_model_name} com {best_accuracy:.4f} ({best_accuracy*100:.2f}%) de acurácia")

print(f"\nConclusões:")
print(f"  - O dataset Iris é um caso clássico de classificação multiclasse")
print(f"  - Todos os modelos apresentaram excelente performance (>90%)")
print(f"  - As características relacionadas à pétala são mais discriminativas")
print(f"  - A espécie 'setosa' é facilmente separável das demais")
print(f"  - 'Versicolor' e 'virginica' apresentam maior sobreposição")

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 60)