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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

def check_data_integrity(
        df_labels: pd.DataFrame,
        base_dir: str = 'dataset/wearables',
        verbose: bool = True
) -> dict:
    """
    Verifica a integridade dos dados, procurando por arquivos de sensores ausentes,
    vazios ou com valores nulos internos para cada amostra no DataFrame de rótulos.

    Args:
        df_labels (pd.DataFrame): DataFrame contendo a coluna 'Id' das amostras a serem verificadas.
        base_dir (str): O diretório base onde as pastas dos usuários estão localizadas.
        verbose (bool): Se True, imprime o progresso e o relatório final no console.

    Returns:
        dict: Um dicionário contendo o relatório de problemas encontrados, indexado pelo user_id.
              Retorna um dicionário vazio se nenhum problema for encontrado.
    """
    sensors_to_check = ['HR', 'EDA', 'TEMP', 'BVP', 'ACC', 'IBI']
    missing_data_report = {}

    if verbose:
        print("Iniciando verificação de integridade dos arquivos de sensores...")

    # Usa tqdm para mostrar uma barra de progresso
    iterator = tqdm(df_labels['Id'], desc="Verificando Amostras") if verbose else df_labels['Id']

    for user_id in iterator:
        missing_files, empty_files, nan_in_files = [], [], {}

        for sensor in sensors_to_check:
            file_path = os.path.join(base_dir, user_id, f'{sensor}.csv')

            if not os.path.exists(file_path):
                missing_files.append(sensor)
                continue

            try:
                if os.path.getsize(file_path) < 2:
                    empty_files.append(sensor)
                    continue

                df_sensor = pd.read_csv(file_path, header=None)
                if df_sensor.isnull().sum().sum() > 0:
                    nan_in_files[sensor] = df_sensor.isnull().sum().sum()

            except pd.errors.EmptyDataError:
                empty_files.append(sensor)

        if missing_files or empty_files or nan_in_files:
            missing_data_report[user_id] = {
                'Arquivos Ausentes': missing_files,
                'Arquivos Vazios': empty_files,
                'Valores NaN Internos': nan_in_files
            }

    if verbose:
        print("Verificação concluída.")
        if not missing_data_report:
            print("Nenhuma amostra possui arquivos de sensores ausentes, vazios ou com valores NaN internos.")
        else:
            print("Foram encontrados problemas de dados ausentes em algumas amostras:")
            for user_id, report in missing_data_report.items():
                print(f"\n  Usuário: {user_id}")
                if report['Arquivos Ausentes']: print(f"    - Arquivos Ausentes: {report['Arquivos Ausentes']}")
                if report['Arquivos Vazios']: print(f"    - Arquivos Vazios: {report['Arquivos Vazios']}")
                if report['Valores NaN Internos']: print(
                    f"    - Valores NaN Internos: {report['Valores NaN Internos']}")

    return missing_data_report

def plot_all_sensors_for_random_sample(
        dataset: pd.DataFrame,
        base_dir: str = 'dataset/wearables'
):
    """
    Seleciona uma única amostra aleatória do dataset e plota todos os seus
    6 sinais de sensores em uma grade 2x3.

    Args:
        dataset (pd.DataFrame): DataFrame contendo a coluna 'Id' e 'Label'.
        base_dir (str): O diretório base onde as pastas dos usuários estão localizadas.
    """

    if dataset.empty:
        print("Dataset está vazio. Não é possível selecionar uma amostra.")
        return

    random_sample = dataset.sample(n=1).iloc[0]
    user_id = random_sample['Id']
    label = random_sample['Label']

    print(f"Selecionada amostra aleatória: {user_id} (Classe: {label})")

    sensors_to_plot = ['HR', 'EDA', 'TEMP', 'BVP', 'ACC', 'IBI']

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f'Visualização Completa dos Sensores para a Amostra: {user_id} (Classe: {label})', fontsize=24, y=1.0)

    axes = axes.flatten()

    print("\nGerando gráficos para cada sensor...")

    for i, sensor in enumerate(sensors_to_plot):
        ax = axes[i]
        file_path = os.path.join(base_dir, user_id, f'{sensor}.csv')

        try:
            if sensor == 'ACC':
                df_signal = pd.read_csv(file_path, header=None, skiprows=1, names=['X', 'Y', 'Z'])
                df_signal.plot(ax=ax, linewidth=1, title=f'Sensor: {sensor}')
                ax.legend(fontsize='medium')
            elif sensor == 'IBI':
                df_signal = pd.read_csv(file_path, header=None, names=['Timestamp', 'Interval'])
                df_signal['Interval'] = pd.to_numeric(df_signal['Interval'], errors='coerce')
                df_signal.dropna(inplace=True)
                ax.plot(df_signal['Timestamp'], df_signal['Interval'])
                ax.set_title(f'Sensor: {sensor}')
                ax.set_xlabel('Timestamp (s)')
            else:  # HR, EDA, TEMP, BVP
                df_signal = pd.read_csv(file_path, header=None, skiprows=1)
                ax.plot(df_signal.index, df_signal.values)
                ax.set_title(f'Sensor: {sensor}')
                ax.set_xlabel('Amostras (Tempo)')

            ax.grid(False)
            ax.set_ylabel('Valor do Sinal')

        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
            ax.text(0.5, 0.5, 'Dados Indisponíveis', ha='center', va='center', color='red')
            ax.set_title(f'Sensor: {sensor}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()



def plot_signal_distribution_for_class(
        dataset,
        target_class: str,
        sensor_name: str,
        base_dir: str = 'dataset/wearables',
        output_dir: str = 'figs/analise_atributos',
        cols: int = 5,
        save_fig: bool = False,
        show_fig: bool = False,
        verbose: bool = False,
):
    """
    Plota a distribuição de um sinal de sensor para todas as amostras de uma classe específica,
    tratando corretamente sensores de uma ou múltiplas colunas (ACC, IBI).
    """
    filtered_samples = dataset[dataset['Label'] == target_class]
    num_samples = len(filtered_samples)

    if num_samples == 0:
        print(f"Nenhuma amostra encontrada para a classe '{target_class}'.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(25, rows * 5), squeeze=False)
    fig.suptitle(f'Distribuição do Sinal {sensor_name.replace(".csv", "")} na Classe {target_class}',
                 fontsize=24, y=0.99)

    global_min, global_max = float('inf'), float('-inf')
    if verbose:
        print(f"Calculando escala global para {num_samples} amostras do sensor {sensor_name}...")
    for user_id in filtered_samples['Id']:
        file_path = os.path.join(base_dir, user_id, sensor_name)
        if not os.path.exists(file_path): continue
        try:
            if sensor_name == 'ACC.csv':
                df = pd.read_csv(file_path, header=None, skiprows=1, names=['X', 'Y', 'Z'])
                if not df.empty:
                    global_min = min(global_min, df.min().min())
                    global_max = max(global_max, df.max().max())
            elif sensor_name == 'IBI.csv':
                df = pd.read_csv(file_path, header=None, names=['Timestamp', 'Interval'])
                intervals = pd.to_numeric(df['Interval'], errors='coerce').dropna()
                if not intervals.empty:
                    global_min = min(global_min, intervals.min())
                    global_max = max(global_max, intervals.max())
            else:
                df = pd.read_csv(file_path, header=None, skiprows=1)
                if not df.empty:
                    global_min = min(global_min, df.iloc[:, 0].min())
                    global_max = max(global_max, df.iloc[:, 0].max())
        except (pd.errors.EmptyDataError, IndexError, ValueError):
            continue

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        print(f"Não foi possível determinar a escala para {sensor_name}. Verifique os arquivos.")
        plt.close(fig)
        return

    margin = (global_max - global_min) * 0.05 if (global_max - global_min) > 0 else 0.1
    global_min -= margin
    global_max += margin

    if verbose:
        print("Escala global definida. Plotando gráficos...")
    for idx, user_id in enumerate(filtered_samples['Id']):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        file_path = os.path.join(base_dir, user_id, sensor_name)
        ax.set_title(user_id, fontsize=12)
        try:
            if sensor_name == 'ACC.csv':
                df = pd.read_csv(file_path, header=None, skiprows=1, names=['X', 'Y', 'Z'])
                if df.empty: raise ValueError("Arquivo vazio após pular linha")
                df.plot(ax=ax, linewidth=1)
                ax.legend(fontsize='small')

            elif sensor_name == 'IBI.csv':
                df = pd.read_csv(file_path, header=None, names=['Timestamp', 'Interval'])
                df['Interval'] = pd.to_numeric(df['Interval'], errors='coerce')
                df.dropna(subset=['Interval'], inplace=True)
                if df.empty: raise ValueError("Nenhum dado numérico no IBI")

                # Plota a coluna 'Interval' usando o timestamp como eixo X para uma visualização mais correta
                ax.plot(df['Timestamp'], df['Interval'], linewidth=1)
                ax.set_xlabel('Timestamp (s)', fontsize=7)

            else:  # HR, EDA, TEMP, BVP
                df = pd.read_csv(file_path, header=None, skiprows=1)
                if df.empty: raise ValueError("Arquivo vazio após pular linha")
                df.iloc[:, 0].plot(ax=ax, linewidth=1, legend=False)

            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_ylim(global_min, global_max)
        except (FileNotFoundError, pd.errors.EmptyDataError, IndexError, ValueError):
            ax.text(0.5, 0.5, f'{user_id}\n(Arquivo ausente/inválido)',
                    ha='center', va='center', fontsize=10, color='red')
            ax.axis('off')

    for j in range(num_samples, rows * cols):
        row, col = j // cols, j % cols
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.join(output_dir,
                                   f"{sensor_name.replace('.csv', '')}_{target_class.lower()}_distribution.png")
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        print(f"Figura salva em '{output_name}'")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    if verbose: print(
        f"\n{num_samples} gráficos da classe '{target_class}' para o sensor '{sensor_name}' foram processados.")


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_boxplot_comparison_for_class(
        dataset,
        target_class: str,
        sensor_name: str,
        base_dir: str = 'dataset/wearables',
        output_dir: str = 'figs/boxplots',
        save_fig: bool = False,
        show_fig: bool = False,
        verbose: bool = False
):
    """
    Função UNIVERSAL para gerar boxplots.
    Corrige o problema de detecção do ACC processado.
    """

    filtered_samples = dataset[dataset['Label'] == target_class]

    if filtered_samples.empty:
        print(f"Nenhuma amostra encontrada para a classe '{target_class}'.")
        return

    all_sensor_data = []
    sample_ids = []

    if verbose:
        print(f"Processando '{sensor_name}' para classe '{target_class}' em: {base_dir}")

    for user_id in filtered_samples['Id']:
        file_path = os.path.join(base_dir, str(user_id), sensor_name)

        if not os.path.exists(file_path):
            continue

        try:
            # --- IBI ---
            if sensor_name == 'IBI.csv':
                try:
                    df = pd.read_csv(file_path, header=None)
                except pd.errors.EmptyDataError:
                    continue
                # Se tiver 2 colunas, pega a segunda (intervalo). Se 1, pega a primeira.
                col_idx = 1 if df.shape[1] >= 2 else 0
                signal_data = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna()

            # --- ACC (Acelerômetro) ---
            elif sensor_name == 'ACC.csv':
                df = pd.read_csv(file_path, header=None)

                # VERIFICAÇÃO INTELIGENTE:
                # Se tem 3 colunas, verificamos se as colunas 1 e 2 (Y e Z) têm dados reais.
                # Arquivos processados com cabeçalho sujo parecem ter 3 colunas, mas Y e Z são NaN.
                is_really_raw = False
                if df.shape[1] >= 3:
                    # Converte para numérico para testar
                    temp_df = df.apply(pd.to_numeric, errors='coerce')
                    # Se mais de 50% da coluna 1 (Y) for válida, é Raw.
                    if temp_df.iloc[:, 1].count() > (len(df) * 0.5):
                        is_really_raw = True

                if is_really_raw:
                    # Lógica Raw: Calcula Magnitude
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    magnitude = np.sqrt(df.iloc[:, 0] ** 2 + df.iloc[:, 1] ** 2 + df.iloc[:, 2] ** 2)
                    signal_data = magnitude
                else:
                    # Lógica Processed: Pega a coluna 0 (que é a magnitude)
                    # Usamos iloc[:, 0] porque se o pandas leu 3 colunas mas só a 1ª tem dados, é ela que queremos.
                    signal_data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()

            # --- OUTROS SENSORES ---
            else:
                df = pd.read_csv(file_path, header=None)
                # Pega a última coluna válida (geralmente a única)
                signal_data = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna()

            # --- FILTROS FINAIS ---
            if not signal_data.empty:
                # Remove zeros absolutos ou negativos espúrios em sensores biológicos
                if sensor_name in ['HR.csv', 'EDA.csv', 'TEMP.csv']:
                    signal_data = signal_data[signal_data > 0]

                if not signal_data.empty:
                    all_sensor_data.append(signal_data)
                    sample_ids.append(user_id)

        except Exception as e:
            if verbose: print(f"Erro em {user_id}: {e}")
            continue

    if not all_sensor_data:
        if verbose: print(f"--> Aviso: Nenhum dado válido extraído para {sensor_name} ({target_class}).")
        return

    # --- PLOTAGEM ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 10))

    box = plt.boxplot(all_sensor_data, labels=sample_ids, vert=True, patch_artist=True)

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_sensor_data)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    title_suffix = "(Magnitude)" if sensor_name == 'ACC.csv' else ""
    plt.title(f'Distribuição - {sensor_name.replace(".csv", "")} {title_suffix} - Classe: {target_class}', fontsize=18)
    plt.ylabel('Valor do Sinal')
    plt.xlabel('Usuários')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        # Identifica no nome se é dado processado ou bruto
        type_data = "raw" if "processed" not in base_dir else "proc"
        fname = f"{sensor_name.replace('.csv', '')}_{target_class}_{type_data}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        print(f"Salvo: {fname}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

def plot_demographic_analysis(df_train: pd.DataFrame, df_users: pd.DataFrame):
    """
    Realiza uma análise demográfica completa dos participantes do conjunto de treino,
    gerando uma grade de gráficos com distribuições, relações e análises cruzadas.

    Args:
        df_train (pd.DataFrame): DataFrame com os rótulos de treino (colunas 'Id', 'Label').
        df_users (pd.DataFrame): DataFrame com os dados demográficos limpos de todos os usuários.
    """

    df_users_train = df_users[df_users['Id'].isin(df_train['Id'])].copy()
    df_merged = pd.merge(df_train, df_users_train, on='Id')

    if 'Height (cm)' in df_merged.columns and 'Weight (kg)' in df_merged.columns:
        df_merged['Height (m)'] = df_merged['Height (cm)'] / 100
        df_merged['IMC'] = df_merged['Weight (kg)'] / (df_merged['Height (m)'] ** 2)

    print(f"Análise demográfica será realizada em {len(df_users_train)} participantes do conjunto de treino.")

    fig, axes = plt.subplots(4, 3, figsize=(22, 24))
    fig.suptitle('Análise Demográfica Completa dos Participantes do Conjunto de Treino', fontsize=28, y=1.0)
    axes = axes.flatten()

    sns.countplot(ax=axes[0], data=df_users_train, x='Gender', palette='viridis', hue='Gender', legend=False)
    axes[0].set_title('Distribuição de Gênero', fontsize=14)
    for c in axes[0].containers: axes[0].bar_label(c)

    sns.histplot(ax=axes[1], data=df_users_train, x='Age', kde=True, bins=10, color='skyblue')
    axes[1].set_title('Distribuição de Idade', fontsize=14)

    sns.histplot(ax=axes[2], data=df_users_train, x='Height (cm)', kde=True, bins=10, color='salmon')
    axes[2].set_title('Distribuição de Altura (cm)', fontsize=14)

    sns.histplot(ax=axes[3], data=df_users_train, x='Weight (kg)', kde=True, bins=10, color='lightgreen')
    axes[3].set_title('Distribuição de Peso (kg)', fontsize=14)

    ax4 = sns.countplot(ax=axes[4], data=df_users_train, x='Does physical activity regularly?', palette='magma',
                        hue='Does physical activity regularly?', legend=False)
    axes[4].set_title('Praticantes de Atividade Física', fontsize=14)
    axes[4].set_xlabel('Pratica Atividade Regularmente?')
    for c in ax4.containers: ax4.bar_label(c)

    ax5 = sns.countplot(ax=axes[5], data=df_users_train, x='Protocol', palette='plasma', hue='Protocol', legend=False)
    axes[5].set_title('Distribuição de Protocolo', fontsize=14)
    for c in ax5.containers: ax5.bar_label(c)

    sns.scatterplot(ax=axes[6], data=df_users_train, x='Age', y='Height (cm)', alpha=0.8, hue='Gender')
    axes[6].set_title('Relação Idade vs. Altura', fontsize=14)
    axes[6].spines[['top', 'right']].set_visible(False)

    sns.scatterplot(ax=axes[7], data=df_users_train, x='Height (cm)', y='Weight (kg)', alpha=0.8, hue='Gender')
    axes[7].set_title('Relação Altura vs. Peso', fontsize=14)
    axes[7].spines[['top', 'right']].set_visible(False)

    sns.countplot(ax=axes[8], data=df_merged, x='Label', hue='Gender', palette='viridis',
                  order=['STRESS', 'AEROBIC', 'ANAEROBIC'])
    axes[8].set_title('Distribuição de Classes por Gênero', fontsize=14)

    sns.countplot(ax=axes[9], data=df_merged, x='Label', hue='Does physical activity regularly?', palette='magma',
                  order=['STRESS', 'AEROBIC', 'ANAEROBIC'])
    axes[9].set_title('Classes por Prática de Atividade Física', fontsize=14)

    axes[10].axis('off')
    axes[11].axis('off')

    # --- FINALIZAÇÃO ---
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def analyze_raw_signal_stats(df_train: pd.DataFrame, base_dir: str = 'dataset/wearables'):
    """
    Calcula estatísticas descritivas dos sinais brutos para cada amostra,
    agrega os resultados por classe e plota uma matriz de correlação das médias.

    Args:
        df_train (pd.DataFrame): DataFrame com os rótulos de treino (colunas 'Id', 'Label').
        base_dir (str): O diretório base onde as pastas dos usuários estão localizadas.
    """
    sensors = ['HR', 'EDA', 'TEMP', 'BVP', 'ACC', 'IBI']
    all_sample_stats = []

    print("Calculando estatísticas descritivas dos sinais brutos para cada amostra...")

    for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
        user_id = row['Id']
        label = row['Label']
        sample_stats = {'Id': user_id, 'Label': label}

        for sensor in sensors:
            file_path = os.path.join(base_dir, user_id, f'{sensor}.csv')
            try:
                signal_data = None
                prefix = sensor

                if sensor == 'ACC':
                    df_sensor = pd.read_csv(file_path, header=None, skiprows=1, names=['X', 'Y', 'Z'])
                    signal_data = np.sqrt(df_sensor['X'] ** 2 + df_sensor['Y'] ** 2 + df_sensor['Z'] ** 2)
                    prefix = 'ACC_magnitude'
                elif sensor == 'IBI':
                    df_sensor = pd.read_csv(file_path, header=None, names=['Timestamp', 'Interval'])
                    signal_data = pd.to_numeric(df_sensor['Interval'], errors='coerce').dropna()
                    prefix = 'IBI_interval'
                else:
                    df_sensor = pd.read_csv(file_path, header=None, skiprows=1, names=['value'])
                    signal_data = df_sensor['value']

                if signal_data is not None and not signal_data.empty:
                    stats = signal_data.describe()
                    for stat_name, value in stats.items():
                        sample_stats[f'{prefix}_{stat_name}'] = value
                    sample_stats[f'{prefix}_skew'] = signal_data.skew()
                    sample_stats[f'{prefix}_kurtosis'] = signal_data.kurtosis()

            except (FileNotFoundError, pd.errors.EmptyDataError, IndexError, ValueError):
                continue

        all_sample_stats.append(sample_stats)

    df_stats_raw = pd.DataFrame(all_sample_stats)

    print("\n--- Medidas Descritivas Agregadas por Classe (Dados Brutos) ---")

    cols_to_display = [col for col in df_stats_raw.columns if
                       any(stat in col for stat in ['mean', 'std', 'skew', 'kurtosis'])]
    summary_stats = df_stats_raw.groupby('Label')[cols_to_display].mean()

    print("\n--- Tabela de Médias das Estatísticas por Classe ---")
    display(summary_stats.round(2).T)

    print("\n\n--- Matriz de Correlação entre as Médias dos Sinais ---")

    mean_cols_for_corr = [col for col in df_stats_raw.columns if 'mean' in col]
    if not mean_cols_for_corr:
        print("Não foi possível gerar a matriz de correlação (nenhuma coluna 'mean' encontrada).")
        return

    correlation_df = df_stats_raw[mean_cols_for_corr].copy()
    correlation_df.columns = [col.replace('_mean', '').replace('_magnitude', ' (Mag)').replace('_interval', '') for col
                              in mean_cols_for_corr]
    correlation_matrix = correlation_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=.5,
        vmin=-1, vmax=1
    )
    plt.title('Matriz de Correlação entre as Médias dos Sinais', fontsize=16)
    plt.show()