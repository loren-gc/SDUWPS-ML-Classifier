# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Anderson Goncalves e Lorenzo Grippo
# RA: 821675 e 823917
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def plot_auc_comparison(model_scores):
    """
    Plota um gráfico de barras comparando o AUC de todos os modelos testados.
    """
    results = []
    for name, data in model_scores.items():
        results.append({'Modelo': name, 'AUC': data['score']})

    df_res = pd.DataFrame(results).sort_values(by='AUC', ascending=False)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='AUC', y='Modelo', data=df_res, palette='viridis')

    for i in ax.containers:
        ax.bar_label(i, fmt='%.4f', padding=3)

    plt.title('Comparação de Performance (AUC Médio - Cross Validation)', fontsize=16)
    plt.xlabel('Área Sob a Curva (AUC)')
    plt.xlim(0, 1.1)  # Dá espaço para o texto
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_grid(model_scores, X, y, classes, cv=5):
    """
    Gera uma grade (grid) com as Matrizes de Confusão de TODOS os modelos.
    Usa cross_val_predict para gerar predições justas sobre o treino.
    """
    models = model_scores.keys()
    n_models = len(models)

    cols = 3
    rows = math.ceil(n_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    print("Gerando matrizes de confusão (pode demorar um pouco)...")

    for i, (name, data) in enumerate(model_scores.items()):
        ax = axes[i]
        model = data['estimator']

        # Gera predições usando CV (evita data leakage de testar no mesmo que treinou)
        try:
            y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
            cm = confusion_matrix(y, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                        xticklabels=classes, yticklabels=classes)

            ax.set_title(f'{name}\n(AUC: {data["score"]:.3f})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Real')
            ax.set_xlabel('Predito')
        except Exception as e:
            ax.text(0.5, 0.5, f"Erro ao gerar:\n{str(e)}", ha='center', va='center')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_feature_importance(best_model, feature_names, top_n=20):
    """
    Plota a importância das features para modelos baseados em árvore.
    """
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_

        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=df_imp, palette='magma')
        plt.title(f'Top {top_n} Features Mais Importantes ({best_model.__class__.__name__})', fontsize=15)
        plt.xlabel('Importância Relativa')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()

        return df_imp
    else:
        print(
            f"O modelo campeão ({best_model.__class__.__name__}) não fornece 'feature_importances_' nativo (ex: KNN, SVM).")
        return None


def plot_classification_report_heatmap(model, X, y, classes, cv=5):
    """
    Gera um heatmap colorido das métricas de Precision, Recall e F1-Score por classe.
    Facilita identificar qual classe é a 'mais difícil' para o modelo.
    """
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    # Gera dicionário de métricas
    report = classification_report(y, y_pred, target_names=classes, output_dict=True)

    # Converte para DataFrame e remove as linhas de média (accuracy, macro avg...)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.loc[classes, ['precision', 'recall', 'f1-score']]

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    plt.title(f'Performance Detalhada por Classe ({model.__class__.__name__})', fontsize=14)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_prediction_confidence(model, X, y, classes, cv=5):
    """
    Plota a distribuição de probabilidade atribuída à classe VERDADEIRA.
    Quanto mais à direita (perto de 1.0), mais confiante e correto o modelo está.
    """
    if not hasattr(model, "predict_proba"):
        print("Este modelo não suporta probabilidades (predict_proba).")
        return

    # Probabilidades de todas as classes
    y_probas = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)
    true_class_probs = y_probas[np.arange(len(y)), y]

    # Cria DataFrame para plotagem
    df_conf = pd.DataFrame({
        'Probabilidade Atribuída à Classe Real': true_class_probs,
        'Classe Real': [classes[i] for i in y]
    })

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=df_conf,
        x='Probabilidade Atribuída à Classe Real',
        hue='Classe Real',
        fill=True,
        palette='rocket',
        alpha=0.4,
        linewidth=2
    )
    plt.title(f'Confiança do Modelo nas Predições ({model.__class__.__name__})', fontsize=14)
    plt.xlabel('Probabilidade (Confiança)')
    plt.ylabel('Densidade')
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()