import pandas as pd
import math
import random

pontos = []
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
    

def lerDados():
    df = pd.read_csv("./datasets/iris.csv")
    for index, data in df.iterrows():
        novoPonto = Ponto(data["Id"], data["SepalLengthCm"], data["SepalWidthCm"],data["PetalLengthCm"], data["PetalWidthCm"], data["Species"])
        pontos.append(novoPonto)
    

def k_means(k):
  centroides = []
  for i in range(k):
      centroidePonto = random.choice(pontos)
      print(centroidePonto)

lerDados()
k_means(3)