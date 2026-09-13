# Mini Projeto Avaliativo - Módulo 2: Inspeção Visual e Classificação de Peças de Fundição

Este projeto foi desenvolvido como parte do Módulo 2 do curso **Machine Learning e Visão Computacional** do programa **SCTEC 2026**, com foco na aplicação integrada de Visão Computacional Clássica e Inteligência Artificial para controle de qualidade industrial.

O pipeline analisa imagens de peças de fundição metálica e classifica cada peça como **OK** ou **Defeituosa**. A solução combina técnicas de processamento de imagens com OpenCV, para compreender e destacar visualmente os defeitos, e uma Rede Neural Convolucional (CNN), para automatizar a classificação em escala.

## 📄 1. Contextualização

A Indústria 4.0 exige processos de inspeção mais rápidos, consistentes e escaláveis. A inspeção visual manual pode ser demorada e sofrer com variações de iluminação, fadiga e diferenças de interpretação entre operadores.

Neste projeto, a análise exploratória com OpenCV é utilizada antes do treinamento da rede neural. A etapa clássica reduz ruídos, evidencia bordas e ajuda a compreender as características visuais das anomalias. Em seguida, uma CNN aprende padrões espaciais das imagens para classificar automaticamente as peças.

O dataset utilizado é o **Casting Product Image Data for Quality Inspection**, composto por imagens de peças de fundição metálica com e sem defeitos estruturais.

## 📁 2. Estrutura do Projeto

- `metal_casting_classification_pipeline.ipynb`: notebook principal com todo o pipeline;
- `data/casting_512x512/`: diretório destinado às imagens do dataset;
  - `def_front/`: imagens de peças defeituosas;
  - `ok_front/`: imagens de peças aprovadas.

## 🎯 3. Objetivos do Projeto

O objetivo principal é desenvolver um pipeline completo de inspeção visual que una processamento clássico e aprendizado profundo.

Objetivos específicos:

1. Selecionar amostras do dataset para análise exploratória;
2. Aplicar Grayscale, Gaussian Blur, Thresholding, Canny e operações morfológicas;
3. Carregar o conjunto completo de imagens com `image_dataset_from_directory`;
4. Dividir os dados em treino e validação de forma reprodutível;
5. Aplicar Data Augmentation com rotação, zoom e variação de brilho;
6. Construir e treinar uma CNN com `Conv2D`, `MaxPooling2D`, `Flatten` e camadas densas;
7. Auditar o treinamento por meio das curvas de Loss e Acurácia;
8. Avaliar o modelo com matriz de confusão e relatório de classificação.

## 🔬 4. Pipeline de Processamento

### 4.1 Análise exploratória com OpenCV

Uma amostra das imagens é processada com as seguintes operações:

- **Grayscale:** conversão para escala de cinza, reduzindo a imagem a um canal;
- **Gaussian Blur:** suavização de ruídos e texturas de alta frequência;
- **Thresholding de Otsu:** separação automática de regiões com diferentes intensidades;
- **Canny:** detecção de bordas e descontinuidades estruturais;
- **Dilatação e erosão:** reforço de traços relevantes e redução de falhas ou ruídos na segmentação.

Essa etapa demonstra que os métodos clássicos conseguem destacar geometrias anômalas, mas também podem sofrer interferência de reflexos e variações de iluminação nas superfícies metálicas.

### 4.2 Ingestão e preparação dos dados

O carregamento é realizado com `tf.keras.utils.image_dataset_from_directory`, utilizando:

- imagens redimensionadas para `128 x 128` pixels;
- lote de tamanho `32`;
- divisão de `80%` para treino e `20%` para validação;
- semente fixa `42` para reprodutibilidade;
- `cache()` e `prefetch()` para otimização da entrada de dados.

### 4.3 Data Augmentation

Durante o treinamento, o pipeline aplica transformações para simular variações encontradas em uma linha industrial:

- rotação aleatória;
- zoom aleatório;
- alteração aleatória de brilho.

O objetivo é reduzir o sobreajuste e aumentar a capacidade de generalização do modelo.

### 4.4 Arquitetura da CNN

O modelo possui três blocos de extração de características, cada um formado por `Conv2D` e `MaxPooling2D`, seguidos por:

- `Flatten` para transformar os mapas de características em um vetor;
- camada densa com 128 unidades e ativação ReLU;
- `Dropout` para reduzir sobreajuste;
- camada final com uma unidade e ativação sigmoide para classificação binária.

O treinamento utiliza o otimizador **Adam**, a função de perda **binary crossentropy**, a métrica de **acurácia** e `EarlyStopping` com restauração dos melhores pesos.

## 📊 5. Resultados e Auditoria

O notebook gera curvas de treinamento e validação para:

- **Loss:** acompanhamento da função de perda ao longo das épocas;
- **Acurácia:** comparação do desempenho em treino e validação.

Também são gerados:

- matriz de confusão;
- relatório de classificação com precisão, recall e F1-Score.

Na execução documentada, o modelo alcançou aproximadamente **95% de acurácia global**, com perda de validação próxima de **0,1051**. A matriz de confusão registrou 138 peças defeituosas e 109 peças OK classificadas corretamente, além de 11 falsos negativos e 2 falsos positivos.

Os falsos negativos representam o principal risco industrial, pois correspondem a peças defeituosas classificadas como aprovadas. Por isso, o projeto sugere como próximos passos o ajuste do limiar de decisão, uma abordagem híbrida com características do OpenCV e o uso de Transfer Learning.

## 🛠️ 6. Tecnologias Utilizadas

- Python;
- OpenCV;
- TensorFlow/Keras;
- NumPy;
- Matplotlib;
- Seaborn;
- Scikit-learn;
- Jupyter Notebook.

## ▶️ 7. Como Executar

1. Instale o Python e as dependências do projeto:

   ```bash
   pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn jupyter
   ```

2. Baixe o dataset **Casting Product Image Data for Quality Inspection** e extraia-o em:

   `/data/casting_512x512/`

   O diretório deve conter as pastas `def_front` e `ok_front`.

3. Abra o notebook:

   `metal_casting_classification_pipeline.ipynb`

4. Confira a variável `DATASET_DIR` e ajuste o caminho caso o dataset esteja em outro local.

5. Execute as células em ordem, da preparação do ambiente até as conclusões finais.
