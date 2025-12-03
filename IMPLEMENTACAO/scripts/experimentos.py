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

# Arquivo com todas as funcoes e codigos referentes aos experimentos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Imports de Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb


def prepare_modeling_data(df_features, df_users, df_labels):
    """
    Junta as features extraídas com dados demográficos, faz o encoding dos labels
    e a normalização.
    """
    df_sensor = df_features.copy()
    if df_sensor.index.name == 'Id' or 'Id' not in df_sensor.columns:
        df_sensor.index.name = 'Id'
        df_sensor = df_sensor.reset_index()

    df_sensor['Id'] = df_sensor['Id'].astype(str).str.strip()
    df_users['Id'] = df_users['Id'].astype(str).str.strip()

    df_sensor['join_key'] = df_sensor['Id'].apply(lambda x: x.split('_w')[0])
    df_full = pd.merge(df_sensor, df_users, left_on='join_key', right_on='Id', suffixes=('', '_user'))

    print(f"Dataset montado. Dimensões: {df_full.shape}")

    if df_full.empty:
        print("ERRO CRÍTICO: Merge vazio.")
        print(f"Exemplo Feature ID: {df_sensor['join_key'].head(1).values}")
        print(f"Exemplo User ID:    {df_users['Id'].head(1).values}")
        raise ValueError("IDs não batem.")

    df_labels_clean = df_labels.copy()
    df_labels_clean['Id'] = df_labels_clean['Id'].astype(str).str.strip()
    label_map = df_labels_clean.set_index('Id')['Label'].to_dict()

    y_raw = df_full['join_key'].map(label_map)
    df_full = df_full[y_raw.notna()]
    y_raw = y_raw[y_raw.notna()]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = label_encoder.classes_

    cols_to_drop = ['Id', 'Id_user', 'join_key', 'original_Id', 'Label']
    X = df_full.drop(columns=[c for c in cols_to_drop if c in df_full.columns], errors='ignore')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, classes, scaler, label_encoder


def run_benchmark_models(X, y, cv=5, n_jobs=1):
    """
    Roda GridSearchCV para uma bateria de modelos clássicos e avançados.
    Retorna um dicionário com os resultados.
    """
    print(f"\nIniciando Benchmark de Modelos (CV={cv})...")
    model_scores = {}

    models_config = [
        ('k-NN', KNeighborsClassifier(),
         {
             'n_neighbors': [3, 5, 11, 21, 31],
             'weights': ['uniform', 'distance'],
             'metric': ['euclidean', 'manhattan'],
             'p': [1, 2]
         }),

        ('Naïve Bayes', GaussianNB(),
         {'var_smoothing': np.logspace(0, -9, num=20)}),

        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
         {'C': [0.1, 1, 10]}),

        ('MLP', MLPClassifier(random_state=42, max_iter=500),
         {
             'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 75), (200, 100)],
             'alpha': [0.0001, 0.001, 0.01],
             'learning_rate_init': [0.001, 0.01],
             'activation': ['relu', 'tanh']
         }),

        ('SVM', SVC(probability=True, random_state=42),
         {
             'C': [0.1, 1, 5, 10, 20],
             'kernel': ['rbf', 'poly'],
             'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
             'degree': [2, 3]  # só usado no kernel poly
         }),

        ('Random Forest', RandomForestClassifier(random_state=42),
         {
             'n_estimators': [100, 200, 300],
             'max_depth': [10, 20, None],
             'min_samples_split': [2, 5],
             'min_samples_leaf': [1, 2]
         }),

        ('XGBoost', xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42, verbosity=0),
         {
             'n_estimators': [100, 200],
             'learning_rate': [0.05, 0.1],
             'max_depth': [3, 5],
             'subsample': [0.9, 1.0],
             'colsample_bytree': [0.8, 1.0]
         }),

        ('LightGBM', lgb.LGBMClassifier(objective='multiclass', random_state=42, verbosity=-1),
         {
             'n_estimators': [200, 400],
             'learning_rate': [0.05, 0.1],
             'num_leaves': [31],
             'feature_fraction': [0.8, 1.0],
             'bagging_fraction': [0.8, 1.0]
         })
    ]

    for name, model, params in models_config:
        print(f"Otimizando {name}...")
        try:
            grid = GridSearchCV(model, params, cv=cv, scoring='roc_auc_ovr', n_jobs=n_jobs)
            grid.fit(X, y)
            model_scores[name] = {
                'score': grid.best_score_,
                'params': grid.best_params_,
                'estimator': grid.best_estimator_
            }
            print(f"  -> AUC: {grid.best_score_:.4f}")
        except Exception as e:
            print(f"  -> Erro no {name}: {e}")

    return model_scores


def plot_best_model_performance(model_scores, X, y, classes, cv=5):
    """
    Seleciona o melhor modelo, faz predições cross-validadas e plota
    Matriz de Confusão e Curvas ROC.
    """
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    best_name = sorted_models[0][0]
    best_model = model_scores[best_name]['estimator']

    print("\n" + "=" * 50)
    print(f"CAMPEÃO: {best_name} (AUC: {model_scores[best_name]['score']:.4f})")
    print("=" * 50)

    y_probas = cross_val_predict(best_model, X, y, cv=cv, method='predict_proba')
    y_pred = np.argmax(y_probas, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=classes, yticklabels=classes)
    axes[0].set_title(f'Matriz de Confusão - {best_name}', fontsize=14)
    axes[0].set_xlabel('Predito')
    axes[0].set_ylabel('Real')

    y_bin = label_binarize(y, classes=range(len(classes)))
    colors = plt.cm.get_cmap('tab10', len(classes))

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probas[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, lw=2, label=f'ROC {cls} (area = {roc_auc:.2f})')

    axes[1].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'Curvas ROC - {best_name}', fontsize=14)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    print("\nRelatório de Classificação:")
    print(classification_report(y, y_pred, target_names=classes))

    return best_model


def generate_submission_df(X_test_features, train_columns, model, scaler, label_encoder, sample_sub_path):
    """
    Alinha as features de teste com o treino, faz a predição e formata o DataFrame de submissão.
    """
    print("\n--- Gerando Submissão ---")
    X_aligned = pd.DataFrame(0, index=X_test_features.index, columns=train_columns)

    common_cols = list(set(X_test_features.columns) & set(train_columns))
    X_aligned[common_cols] = X_test_features[common_cols]

    X_aligned = X_aligned[train_columns]
    print(f"Dimensões alinhadas: {X_aligned.shape}")

    X_scaled = scaler.transform(X_aligned)
    y_probs = model.predict_proba(X_scaled)

    submission = pd.DataFrame()
    submission['Id'] = X_aligned.index

    classes = label_encoder.classes_
    try:
        idx_stress = list(classes).index('STRESS')
        idx_aerobic = list(classes).index('AEROBIC')
        idx_anaerobic = list(classes).index('ANAEROBIC')

        submission['Predicted_0'] = y_probs[:, idx_stress]  # Stress
        submission['Predicted_1'] = y_probs[:, idx_aerobic]  # Aerobic
        submission['Predicted_2'] = y_probs[:, idx_anaerobic]  # Anaerobic
    except ValueError:
        print("Aviso: Classes não encontradas. Usando ordem padrão (0, 1, 2).")
        submission['Predicted_0'] = y_probs[:, 2]
        submission['Predicted_1'] = y_probs[:, 0]
        submission['Predicted_2'] = y_probs[:, 1]

    df_sample = pd.read_csv(sample_sub_path)
    df_final = pd.merge(df_sample[['Id']], submission, on='Id', how='left')

    df_final.fillna(1 / 3, inplace=True)

    return df_final