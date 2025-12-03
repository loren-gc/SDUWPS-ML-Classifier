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

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import linregress, skew, kurtosis, entropy
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def preprocess_user_info(file_path: str):
    """
    Carrega, limpa e pré-processa o arquivo de informações demográficas.
    Trata caracteres especiais (*), imputa valores e codifica categorias.
    """
    try:
        df = pd.read_csv(
            file_path,
            na_values='-',
            skipfooter=10,
            engine='python',
            sep=','
        )
        print("Arquivo carregado. Dimensão inicial:", df.shape)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
        return None

    # Remove asteriscos (*) que indicam anotações de erro 'Yes****' vira 'Yes')
    df = df.replace(r'\*', '', regex=True)
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].str.strip()

    print("Limpeza de caracteres especiais (*) e espaços concluída.")

    cols_to_drop = ['Stress Inducement', 'Aerobic Exercise', 'Anaerobic Exercise']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Colunas estáticas removidas: {cols_to_drop}")

    numeric_cols = ['Age', 'Height (cm)', 'Weight (kg)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("\n--- Imputação ---")
    for col in df.columns:
        if col == 'Id': continue  # Não imputamos ID

        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                # Imputação pela média
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                print(f"'{col}': NaNs preenchidos com Média ({mean_val:.1f})")
            else:
                # Imputação pela moda
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"'{col}': NaNs preenchidos com Moda ('{mode_val}')")

    print("\n--- Codificação ---")

    target_col = 'Does physical activity regularly?'
    if target_col in df.columns:
        df['activity_regularly'] = df[target_col].map({'Yes': 1, 'No': 0})
        df['activity_regularly'] = df['activity_regularly'].fillna(0).astype(int)
        df.drop(columns=[target_col], inplace=True)

    # One-Hot Encoding para Gênero e Protocolo
    categorical_cols = ['Gender', 'Protocol']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    df.rename(columns={
        'Gender_m': 'is_male',
        'Protocol_V2': 'is_protocol_v2'
    }, inplace=True)

    print("Pré-processamento demográfico concluído.")
    return df


def load_and_format_sensor_file(file_path, sensor_name):
    """
    Carrega e padroniza as colunas dos sensores.
    """
    try:
        skip = 1 if sensor_name == 'IBI' else 0
        try:
            df = pd.read_csv(file_path, header=None, skiprows=skip)
        except pd.errors.ParserError:
            df = pd.read_csv(file_path, header=None, skiprows=skip, engine='python')

        if df.empty: return pd.DataFrame()

        cols = df.shape[1]

        if sensor_name == 'ACC':
            if cols == 3:
                df.columns = ['X', 'Y', 'Z']
            elif cols >= 4:
                df = df.iloc[:, -3:]
                df.columns = ['X', 'Y', 'Z']
            else:
                return pd.DataFrame()
        elif sensor_name == 'IBI':
            if cols >= 2:
                df = df.iloc[:, :2]
                df.columns = ['Timestamp', 'Interval']
            elif cols == 1:
                df.columns = ['Interval']
            else:
                return pd.DataFrame()
        else:
            if cols >= 1:
                df = df.iloc[:, -1:]
                df.columns = ['value']
            else:
                return pd.DataFrame()

        return df

    except Exception:
        return pd.DataFrame()


def preprocess_sensor_file(df_sensor, sensor_name: str):
    """
    Aplica um pipeline de pré-processamento robusto, incluindo o cálculo
    da magnitude do acelerômetro e tratamento de outliers com IQR.
    """
    if df_sensor.empty: return None
    processed_df = df_sensor.copy()

    if sensor_name == 'ACC':
        processed_df.columns = ['X', 'Y', 'Z']
        for col in ['X', 'Y', 'Z']:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    elif sensor_name == 'IBI':
        processed_df.columns = ['Timestamp', 'Interval']
        processed_df['Interval'] = pd.to_numeric(processed_df['Interval'], errors='coerce')
    else:  # HR, EDA, TEMP, BVP
        processed_df.columns = ['value']
        processed_df['value'] = pd.to_numeric(processed_df['value'], errors='coerce')

    processed_df = processed_df.infer_objects(copy=False)
    processed_df.interpolate(method='linear', limit_direction='both', inplace=True)
    processed_df.dropna(inplace=True)
    if processed_df.empty: return None

    # --- 3. Lógica Específica para Acelerômetro (Cálculo da Magnitude) ---
    if sensor_name == 'ACC':
        # Calcula a magnitude do vetor e cria uma nova coluna 'value'
        magnitude = np.sqrt(processed_df['X'] ** 2 + processed_df['Y'] ** 2 + processed_df['Z'] ** 2)
        # Cria um novo DataFrame apenas com a magnitude para padronizar a saída
        processed_df = pd.DataFrame({'value': magnitude})
        target_columns = ['value']
    elif sensor_name == 'IBI':
        target_columns = ['Interval']
    else:
        target_columns = ['value']

    # Tratamento de Outliers
    hard_limits = {'HR': (40, 220), 'TEMP': (20, 42), 'EDA': (0.01, 30), 'IBI': (0.27, 2.0), 'BVP': (-150, 150)}
    if sensor_name in hard_limits:
        lower_limit, upper_limit = hard_limits[sensor_name]
        col_to_clip = 'value' if sensor_name != 'IBI' else 'Interval'
        processed_df[col_to_clip] = processed_df[col_to_clip].clip(lower=lower_limit, upper=upper_limit)

    for col in target_columns:
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.0 * IQR
        upper_bound = Q3 + 2.0 * IQR
        processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)

    return processed_df


def load_data_for_tsfresh(base_path, df_labels):
    """
    Lê os arquivos CSV pré-processados de todos os usuários e sensores,
    formatando-os para o padrão exigido pelo tsfresh (Long Format).
    """
    all_sensor_data = []

    for index, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Carregando sensores"):
        user_id = str(row['Id']).strip()

        for sensor in ['HR', 'EDA', 'TEMP', 'BVP', 'ACC', 'IBI']:
            file_path = os.path.join(base_path, user_id, f'{sensor}.csv')

            if not os.path.exists(file_path):
                continue

            try:
                skip = 0 if sensor == 'IBI' else 1
                df_sensor = pd.read_csv(file_path, header=None, skiprows=skip)

                if df_sensor.empty: continue

                if sensor == 'ACC':
                    df_sensor.columns = ['value']
                elif sensor == 'IBI':
                    # IBI tem Timestamp e Interval, queremos Interval
                    col_idx = 1 if df_sensor.shape[1] >= 2 else 0
                    df_sensor = df_sensor.iloc[:, col_idx].to_frame(name='value')
                else:
                    df_sensor.columns = ['value']

                # Adiciona colunas obrigatórias do tsfresh
                df_sensor['kind'] = sensor
                df_sensor['Id'] = user_id
                df_sensor['time'] = range(len(df_sensor))

                df_sensor['value'] = pd.to_numeric(df_sensor['value'], errors='coerce')
                df_sensor.dropna(subset=['value'], inplace=True)

                all_sensor_data.append(df_sensor[['Id', 'time', 'kind', 'value']])

            except Exception:
                continue

    if not all_sensor_data:
        return pd.DataFrame()

    return pd.concat(all_sensor_data, ignore_index=True)


def plot_correlation_heatmap(df_features, n_top=15):
    """Gera o heatmap das top features mais correlacionadas."""
    top_features = df_features.iloc[:, :n_top]
    correlation_matrix = top_features.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        correlation_matrix,
        cmap='bwr', annot=True, fmt='.2f',
        linewidths=.5, vmin=-1, vmax=1
    )
    plt.title(f'Matriz de Correlação das Top {n_top} Features', fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def extract_advanced_tag_features(valid_ids, tags_base_dir):
    """
    Percorre os usuários e extrai ~20 características estatísticas avançadas
    dos arquivos tags.csv (Regularidade, Tendência, Distribuição).
    """
    tags_features_list = []

    for uid in valid_ids:
        tag_path = os.path.join(tags_base_dir, str(uid), 'tags.csv')

        feat = {
            'Id': str(uid),
            'tags_count': 0.0, 'tags_duration_total': 0.0, 'tags_density': 0.0, 'tags_start_hour': 0.0,
            'tags_int_mean': 0.0, 'tags_int_std': 0.0, 'tags_int_min': 0.0, 'tags_int_max': 0.0,
            'tags_int_median': 0.0, 'tags_int_iqr': 0.0,
            'tags_int_skewness': 0.0, 'tags_int_kurtosis': 0.0,
            'tags_int_cv': 0.0, 'tags_int_entropy': 0.0,
            'tags_int_slope': 0.0, 'tags_early_late_ratio': 0.0,
            'tags_ratio_max_min': 0.0, 'tags_double_clicks': 0.0
        }

        if os.path.exists(tag_path) and os.path.getsize(tag_path) > 0:
            try:
                df_tags = pd.read_csv(tag_path, header=None)
                timestamps = pd.to_datetime(df_tags[0]).sort_values()
                count = len(timestamps)

                feat['tags_count'] = float(count)
                feat['tags_start_hour'] = timestamps.iloc[0].hour

                if count > 1:
                    intervals = timestamps.diff().dt.total_seconds().dropna().values
                    duration = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()

                    feat['tags_duration_total'] = duration
                    if duration > 0: feat['tags_density'] = count / duration

                    feat['tags_int_mean'] = np.mean(intervals)
                    feat['tags_int_std'] = np.std(intervals, ddof=1)
                    feat['tags_int_min'] = np.min(intervals)
                    feat['tags_int_max'] = np.max(intervals)
                    feat['tags_int_median'] = np.median(intervals)
                    feat['tags_int_iqr'] = np.percentile(intervals, 75) - np.percentile(intervals, 25)

                    if feat['tags_int_mean'] > 0:
                        feat['tags_int_cv'] = feat['tags_int_std'] / feat['tags_int_mean']

                    if feat['tags_int_min'] > 0:
                        feat['tags_ratio_max_min'] = feat['tags_int_max'] / feat['tags_int_min']

                    hist_counts, _ = np.histogram(intervals, bins='auto')
                    feat['tags_int_entropy'] = entropy(hist_counts)

                    if len(intervals) > 2:
                        feat['tags_int_skewness'] = skew(intervals)
                        feat['tags_int_kurtosis'] = kurtosis(intervals)

                    if len(intervals) > 1:
                        slope, _, _, _, _ = linregress(range(len(intervals)), intervals)
                        feat['tags_int_slope'] = slope

                    mid_time = timestamps.iloc[0] + pd.Timedelta(seconds=duration / 2)
                    early = len(timestamps[timestamps < mid_time])
                    late = len(timestamps[timestamps >= mid_time])
                    if late > 0: feat['tags_early_late_ratio'] = early / late

                    feat['tags_double_clicks'] = np.sum(intervals < 2.0)
            except Exception:
                pass

        tags_features_list.append(feat)

    return pd.DataFrame(tags_features_list).set_index('Id')


def visualize_feature_separation(X, y):
    """Gera Boxplots das Top 12 features, PCA 2D e t-SNE 2D."""

    # 1. Boxplots
    try:
        f_values, p_values = f_classif(X, y)
        feat_imp = pd.DataFrame({'Feature': X.columns, 'F-value': f_values})
        top_features = feat_imp.sort_values(by='F-value', ascending=False)['Feature'].head(12).tolist()

        plt.figure(figsize=(24, 10))
        unique_labels = sorted(list(set(y)))
        for i, feature in enumerate(top_features):
            plt.subplot(2, 6, i + 1)
            sns.boxplot(x=y, y=X[feature], palette='viridis', order=unique_labels, showfliers=False)
            plt.title(feature[:25], fontsize=9)
            plt.ylabel('')
        plt.suptitle('Top 12 Features Mais Discriminantes', fontsize=16)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erro no Boxplot: {e}")

    # Preparação para PCA/t-SNE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_.sum() * 100

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='rocket', s=100, alpha=0.8)
    plt.title(f'PCA 2D (Variância: {var:.2f}%)', fontsize=15)
    plt.show()

    # 3. t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='rocket', s=100, alpha=0.8)
    plt.title('t-SNE (2D)', fontsize=15)
    plt.show()

def load_sensor_data_from_ids(user_ids, base_path):
    """
    Carrega dados de sensores para uma LISTA de IDs (usado no teste/submissão).
    Retorna o DataFrame formatado para o tsfresh.
    """
    all_sensor_data = []

    for user_id in tqdm(user_ids, desc="Carregando Sensores"):
        for sensor in ['HR', 'EDA', 'TEMP', 'BVP', 'ACC', 'IBI']:
            file_path = os.path.join(base_path, str(user_id), f'{sensor}.csv')

            if not os.path.exists(file_path): continue

            try:
                skip = 0 if sensor == 'IBI' else 1
                df = pd.read_csv(file_path, header=None, skiprows=skip)
                if df.empty: continue

                if sensor == 'ACC':
                    df.columns = ['value'];
                    df['kind'] = 'ACC'
                elif sensor == 'IBI':
                    # Tenta pegar a segunda coluna (Interval), se não, pega a primeira
                    col_idx = 1 if df.shape[1] >= 2 else 0
                    df = df.iloc[:, col_idx].to_frame(name='value')
                    df['kind'] = 'IBI'
                else:
                    df.columns = ['value'];
                    df['kind'] = sensor

                df['Id'] = user_id
                df['time'] = range(len(df))

                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df.dropna(inplace=True)

                all_sensor_data.append(df[['Id', 'time', 'kind', 'value']])
            except:
                continue

    if not all_sensor_data:
        return pd.DataFrame()

    return pd.concat(all_sensor_data, ignore_index=True)