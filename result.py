import os
import matplotlib.pyplot as plt
from collections import Counter

def count_labels(label_dir):
    """
    Conta as classes nos arquivos de rótulos de uma pasta.

    Args:
        label_dir (str): Caminho para a pasta com os arquivos de rótulos (.txt).

    Returns:
        Counter: Um contador com a quantidade de cada classe.
    """
    class_counter = Counter()
    num_files = 0

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            num_files += 1
            with open(os.path.join(label_dir, label_file), "r") as f:
                for line in f:
                    class_id = line.split()[0]
                    class_counter[class_id] += 1

    return class_counter, num_files

def plot_class_distribution(train_counts, val_counts):
    """
    Gera um gráfico de barras para a distribuição de classes nos conjuntos train e val.

    Args:
        train_counts (Counter): Contagem de classes no conjunto de treino.
        val_counts (Counter): Contagem de classes no conjunto de validação.
    """
    classes = sorted(set(train_counts.keys()).union(val_counts.keys()))
    train_values = [train_counts.get(cls, 0) for cls in classes]
    val_values = [val_counts.get(cls, 0) for cls in classes]

    x = range(len(classes))

    plt.figure(figsize=(10, 6))
    plt.bar(x, train_values, width=0.4, label='Train', align='center')
    plt.bar([i + 0.4 for i in x], val_values, width=0.4, label='Val', align='center')

    plt.xlabel('Class ID')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution in Train and Val Datasets')
    plt.xticks([i + 0.2 for i in x], classes)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main():
    train_label_dir = "dataset/labels/train"  # Substitua pelo caminho correto da pasta train/labels
    val_label_dir = "dataset/labels/val"      # Substitua pelo caminho correto da pasta val/labels

    # Contar os rótulos nas pastas
    train_counts, train_files = count_labels(train_label_dir)
    val_counts, val_files = count_labels(val_label_dir)

    print(f"Número total de arquivos de treino: {train_files}")
    print(f"Número total de arquivos de validação: {val_files}")

    # Exibir contagem de classes
    print(f"Contagem de classes (train): {train_counts}")
    print(f"Contagem de classes (val): {val_counts}")

    # Gerar gráfico de distribuição de classes
    plot_class_distribution(train_counts, val_counts)

if __name__ == "__main__":
    main()
