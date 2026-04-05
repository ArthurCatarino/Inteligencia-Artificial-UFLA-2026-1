import math
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

caminhoDados = r"M:\Programacao\Inteligencia Artificial\datasets\Iris.csv"

class Ponto:
    def __init__(self, id, sepalLengthCm, sepalWidthCm, petalLengthCm, petalWidthCm, species):
        self.id = id
        self.sepalLengthCm = sepalLengthCm
        self.sepalWidthCm = sepalWidthCm
        self.petalLengthCm = petalLengthCm
        self.petalWidthCm = petalWidthCm
        self.species = species

    def calculaDistancia(self, ponto2):
        return math.sqrt((self.sepalLengthCm - ponto2.sepalLengthCm)**2 + 
                         (self.sepalWidthCm - ponto2.sepalWidthCm)**2 + 
                         (self.petalLengthCm - ponto2.petalLengthCm)**2 + 
                         (self.petalWidthCm - ponto2.petalWidthCm)**2)

def carregaTodosDados(caminho):
    pontos = []
    df = pd.read_csv(caminho)
    for index, linha in df.iterrows():
        novoPonto = Ponto(linha["Id"], linha["SepalLengthCm"], linha["SepalWidthCm"], 
                          linha["PetalLengthCm"], linha["PetalWidthCm"], linha["Species"])
        pontos.append(novoPonto)
    return pontos

def embaralhaEDivide(pontos, proporcao_treino=0.7):
    pontos_embaralhados = pontos.copy()
    random.shuffle(pontos_embaralhados)
    
    limite = int(len(pontos_embaralhados) * proporcao_treino)
    dadosTreino = pontos_embaralhados[:limite]
    dadosTeste = pontos_embaralhados[limite:]
    
    return dadosTreino, dadosTeste

def desempateKnn(vizinhos):
    especies = [v.species for v in vizinhos]
    contagem = Counter(especies)
    return contagem.most_common(1)[0][0]

def knn(dadosTreino, dadosTeste, k):
    resposta = []
    for i in dadosTeste:
        distancias = []
        for j in dadosTreino:
            distancia = i.calculaDistancia(j)
            distancias.append((distancia, j))
            
        distancias.sort(key=lambda x: x[0])
        vizinhosProximos = [x[1] for x in distancias[:k]]
        resposta.append(desempateKnn(vizinhosProximos))
    return resposta
    
def calculaAcuracia(dadosTreino, dadosTeste, k):
    previsoes = knn(dadosTreino, dadosTeste, k)
    acertos = 0
    for i in range(len(dadosTeste)):
        if dadosTeste[i].species == previsoes[i]:
            acertos += 1
    return (acertos / len(dadosTeste)) * 100

def avaliaModelo(dadosTreino, dadosTeste, k):
    print(f"=== Avaliando modelo para K = {k} ===")

    previsoes = knn(dadosTreino, dadosTeste, k)
    reais = [ponto.species for ponto in dadosTeste]
    classes_iris = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    relatorio = classification_report(reais, previsoes, target_names=classes_iris)
    print("\nRelatório de Métricas:")
    print(relatorio)
    
    matriz = confusion_matrix(reais, previsoes, labels=classes_iris)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_iris, yticklabels=classes_iris)
    
    plt.title(f'Matriz de Confusão (KNN com K={k})')
    plt.xlabel('Classe Prevista pelo Modelo')
    plt.ylabel('Classe Real')
    
    plt.tight_layout()
    plt.show()

print("Carregando o dataset na memória...")
todos_os_pontos = carregaTodosDados(caminhoDados)

dadosTreino, dadosTeste = embaralhaEDivide(todos_os_pontos, proporcao_treino=0.7)

ks = [1,3,5,7]
for i in ks: #Ao fechar uma matriz de confusão a do proximo k sera aberta automaticamente
  avaliaModelo(dadosTreino, dadosTeste, i)