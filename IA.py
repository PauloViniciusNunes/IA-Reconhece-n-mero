from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from os import system as sys


sys('color a')


def sigmoid(x):
    x = np.clip(x, -709, 709)
    return 1 / (1 + np.exp(-x))


def selecionar_programa():
    root = tk.Tk()
    root.withdraw()  
    caminho_arquivo = filedialog.askopenfilename(title="Selecione uma imagem", filetypes=[("Imagem PNG", "*.png"), ("Todos os arquivos", "*.*")])
    return caminho_arquivo


weight1 = np.full(250000, 1.0, dtype=float)
weight2 = np.full(250000, 1.0, dtype=float)
weight3 = np.full(250000, 1.0, dtype=float)
weight4 = np.full(250000, 1.0, dtype=float)
weight5 = np.full(250000, 1.0, dtype=float)
weight6 = np.full(250000, 1.0, dtype=float)
weight7 = np.full(250000, 1.0, dtype=float)
weight8 = np.full(250000, 1.0, dtype=float)
weight9 = np.full(250000, 1.0, dtype=float)
weight10 = np.full(250000, 1.0, dtype=float)


bias = 1
bias_weight = (np.full(10, 0.5, dtype=float))


weights = [weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9, weight10]


taxa_de_aprendizado = 0.1


while True:
    caminho = selecionar_programa()
    imagem = Image.open(caminho).convert('L')
    pixels = np.array(imagem)


    def numero_arquivo(arquivo_caminho):
        nome_arquivo = arquivo_caminho.split("/")[-1]
        if(nome_arquivo == '10.png'):
            nome_arquivo = str(arquivo_caminho.split("/")[-1][0]) + str(arquivo_caminho.split("/")[-1][1])
        else:
            nome_arquivo = arquivo_caminho.split("/")[-1][0]
        return int(nome_arquivo)
       
    valores_normalizados = pixels / 255


    inp = []


    for i in range(valores_normalizados.shape[0]):
        for j in range(valores_normalizados.shape[1]):
            inp.append(valores_normalizados[i, j])




    esperado = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    esperado[numero_arquivo(caminho) - 1] = 1
    erro = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   
    geracao = 0
    while True:
        sys('cls')
        geracao += 1
        print(f"Geração: {geracao}° Geração")
        print(f"Out: {out}\n Esp: {esperado}\nErro: {erro}\n Tendencia: {out.index(max(out))}")
       
        for a in range(len(out)):
            for b in range(len(inp)):
                out[a] += (inp[b] * weights[a][b])  
            out[a] += (bias * bias_weight[a])
            out[a] = sigmoid(out[a])
           
        if out.index(max(out)) == esperado.index(max(esperado)):
            sys('cls')
            if numero_arquivo(caminho) != 10:
                print(f"A IA reconheceu a imagem como o número {out.index(max(out)) + 1}")
            else:
                print(f"A IA reconheceu a imagem como o número 10")
            input("")
            break
        else:
           
            for k in range(10):
                erro[k] = esperado[k] - out[k]
               
            for k in range(10):
                for t in range(250000):
                    weights[k][t] += (inp[t] * erro[k] * taxa_de_aprendizado)
                bias_weight[k] += (bias * erro[k] * taxa_de_aprendizado)



