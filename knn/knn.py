import math
import pandas as pd

caminhoDados = "M:\Programacao\Inteligencia Artificial\datasets\Iris.csv"

class Ponto:
  def __init__(self,id,sepalLengthCm,sepalWidthCm,petalLengthCm,petalWidthCm,species):
    self.id = id
    self.sepalLengthCm = sepalLengthCm
    self.sepalWidthCm = sepalWidthCm
    self.petalLengthCm = petalLengthCm
    self.petalWidthCm = petalWidthCm
    self.species = species

  def calculaDistancia(self,ponto2):
    return math.sqrt((self.sepalLengthCm - ponto2.sepalLengthCm)**2 + (self.sepalWidthCm - ponto2.sepalWidthCm)**2 + (self.petalLengthCm - ponto2.petalLengthCm)**2 + (self.petalWidthCm - ponto2.petalWidthCm)**2)
  

def carregaDados(caminho):
  dadosTreino = []
  dadosTeste = []
  df = pd.read_csv(caminho)
  df = df.sample(frac=1).reset_index(drop=True)
  limite = int(len(df) * 0.7)

  for index,linha in df.iterrows():
    novoPonto = Ponto(linha["Id"],linha["SepalLengthCm"],linha["SepalWidthCm"],linha["PetalLengthCm"],linha["PetalWidthCm"],linha["Species"])
    if index < limite:
      dadosTreino.append(novoPonto)
    else:
      dadosTeste.append(novoPonto)
  return dadosTeste,dadosTreino

def desempateKnn(vizinhos):
  contador = {}
  for v in vizinhos:
    specie = v.species
    contador[specie] = contador.get(specie,0) + 1

  votos = sorted(contador.items(),key=lambda x: x[1],reverse=True)
  return votos[0][0]

def knn(dadosTreino, dadosTeste,k):
  resposta = []
  for i in dadosTeste:
    distancias = []
    for j in dadosTreino:
      distancia = i.calculaDistancia(j)
      distancias.append((distancia,j))
    distancias.sort(key=lambda x:x[0])
    vizinhosProximos = [x[1] for x in distancias[:k]]
    resposta.append(desempateKnn(vizinhosProximos))

  return resposta
    
dadosTreino, dadosTeste = carregaDados(caminhoDados)
previsoes = knn(dadosTreino, dadosTeste, k=11)
acertos = 0
for i in range(len(dadosTeste)):
    real = dadosTeste[i].species
    previsto = previsoes[i]
    
    if real == previsto:
        acertos += 1
    
    print(f"ID: {dadosTeste[i].id} | Real: {real} | Previsto: {previsto}")

print(f"\nAcurácia: {(acertos/len(dadosTeste)) * 100:.2f}%")

