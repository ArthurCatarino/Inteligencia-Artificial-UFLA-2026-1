import pandas as pd
import math
import random
from sklearn.metrics import silhouette_score

class Ponto:
    def __init__(self, id, sepalLengthCm, sepalWidthCm, petalLengthCm, petalWidthCm):
        self.id = id
        self.coords = [sepalLengthCm, sepalWidthCm, petalLengthCm, petalWidthCm]

    def calculaDistancia(self, centroide_coords):
        # Distância Euclidiana
        soma = sum((a - b) ** 2 for a, b in zip(self.coords, centroide_coords))
        return math.sqrt(soma)

def k_means_custom(df_pontos, k, max_iter=100):
    # Inicialização: seleciona K pontos aleatórios como centroides iniciais
    indices_iniciais = random.sample(range(len(df_pontos)), k)
    centroides = [df_pontos[i].coords[:] for i in indices_iniciais]
    
    labels = [0] * len(df_pontos)
    
    for _ in range(max_iter):
        # 1. Atribuição: cada ponto vai para o centroide mais próximo
        mudou = False
        for i, p in enumerate(df_pontos):
            menor_dist = math.inf
            nova_classe = -1
            
            for idx_c, c_coords in enumerate(centroides):
                dist = p.calculaDistancia(c_coords)
                if dist < menor_dist:
                    menor_dist = dist
                    nova_classe = idx_c
            
            if labels[i] != nova_classe:
                labels[i] = nova_classe
                mudou = True
        
        if not mudou: # Convergência
            break
            
        # recalcula a média dos pontos de cada cluster
        for idx_c in range(k):
            pontos_no_cluster = [df_pontos[i].coords for i in range(len(df_pontos)) if labels[i] == idx_c]
            
            if pontos_no_cluster:
                novas_coords = []
                for d in range(4):
                    media_dim = sum(p[d] for p in pontos_no_cluster) / len(pontos_no_cluster)
                    novas_coords.append(media_dim)
                centroides[idx_c] = novas_coords
                
    return labels

try:
    df = pd.read_csv("./datasets/iris.csv")
except:
    # Fallback caso o arquivo não esteja no local para o exemplo rodar
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

lista_pontos = []
for i, row in df.iterrows():
    lista_pontos.append(Ponto(i, row[0], row[1], row[2], row[3]))

# Matriz de dados para o silhouette_score
X = [p.coords for p in lista_pontos]

# --- Experimentos ---
for k in [3, 5]:
    labels_obtidos = k_means_custom(lista_pontos, k)
    score = silhouette_score(X, labels_obtidos)
    print(f"--- Experimento com K = {k} ---")
    print(f"Silhouette Score: {score:.4f}\n")