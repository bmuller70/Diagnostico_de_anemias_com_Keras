# Diagnostico de anemias com Keras

![](https://www.foa.unesp.br/Home/ensino/departamentos/cienciasbasicas/histologia/2g-sang-pp1.jpg)


# O projeto
Esse projeto foi desenvolvido em conjunto com a formação da [Alura Deep Learning com Tensorflow e Keras](https://cursos.alura.com.br/formacao-deep-learning-tensorflow-keras), essa experiementação é uma adaptação do projeto que a instrutora fez em aula utilizando a base Iris, como eu gosto de ir fazendo com um tema do interesse em paralelo, acabei esse experimento. A ideia desse texto é compartilhar um pouco do que aprendi e as possibilidades do Keres/Tensorflow e também fortalecer o meu aprendizado. Se você chegou aqui espero que o texto seja útil, e caso você tenha correções, sugestões ou comentarios só me manda por favor, vou ficar feliz demais! 

# Objetivo
O objetivo deste projeto é desenvolver um modelo preditivo capaz de identificar diferentes tipos de anemia a partir de exames de sangue utilizando técnicas de aprendizado de máquina, mais especificamente redes neurais implementadas com a biblioteca Keras. O sistema busca automatizar o processo de diagnóstico, detectando condições como anemia por deficiência de ferro, leucemia, anemias macrocíticas, entre outras, com base em resultados de hemogramas. Por ser parte da formação citada anteriormente não foram exploradas todas as possibilidades e otimizações disponíveis, trabalhando apenas com os aspectos apresentados no curso.

# Descrição
O projeto parte de um [conjunto de dados do Kaggle](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification/data) contendo 1281 amostras de exames de sangue, onde cada entrada representa os resultados de exames hematológicos conforme o exemplo abaixo:

```
    'WBC': 2.740,    # Contagem de glóbulos brancos
    'LYMp': 36.9,   # Porcentagem de linfócitos
    'NEUTp': 60.0,  # Porcentagem de neutrófilos
    'LYMn': 2.0,    # Contagem absoluta de linfócitos 
    'NEUTn': 3.5,   # Contagem absoluta de neutrófilos
    'RBC': 4.77,    # Contagem de glóbulos vermelhos 
    'HGB': 13.5,    # Hemoglobina 
    'HCT': 39.7,    # Hematócrito 
    'MCV': 83.2,    # Volume corpuscular médio 
    'MCH': 28.3,    # Hemoglobina corpuscular média
    'MCHC': 320.0,  # Concentração de hemoglobina corpuscular
    'PLT': 167.0,   # Contagem de plaquetas
    'PDW': 0.15,    # Largura de distribuição de plaquetas
    'PCT': 0.02     # Proporção de plaquetas
```

Cada entrada está associada a um diagnósticos binário relacionado a diferentes tipos de anemia distribuidos da seguinte forma:

![](graficos_anemias.png)


## Pré-Processamento
O dataset do Kaggle possuía uma coluna chamada “Diagnosis”, contendo os diagnósticos de forma descritiva. Para que o modelo pudesse lidar com essas informações, foi necessário transformar essa coluna em variáveis binárias (dummies).

```
diagnosis_dummies = pd.get_dummies(df['Diagnosis'])
df = pd.concat([df, diagnosis_dummies], axis=1)

```


Os dados numéricos dos exames de sangue foram normalizados utilizando a técnica de StandardScaler, que padroniza os valores com média 0 e desvio padrão 1. Essa normalização foi necessária devido à grande amplitude entre os parâmetros dos exames de sangue.

![](graficos_parametros.png)

## Modelo

A criação do modelo e suas definições iniciais foram feitas utilizando o método [`keras.Sequential`](https://keras.io/api/models/sequential/) do Keras, que permite a definição de camadas como  [`Dense`](https://keras.io/api/layers/core_layers/dense/) que são totalmente conectadas. Cada camada contém um número definido de neurônios; neste caso, foram utilizadas duas camadas com 64 e 32 neurônios, respectivamente.

```
model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(Y_train.shape[1], activation='sigmoid'))

```

Aquilo que chamamos de "neuronios" são denominados perceptrons, eles são a estrutura básica dos modelos de deep learning, considerado o "átomo" de uma rede neural. Ele é composto por uma entrada, um processamento e uma saída, mimetizando de forma simbólica as funções de um neurônio biológico (dendritos, axônio e terminais sinápticos). Esse processamento é expresso pela equação:


$$
y = f(\mathbf{w} \cdot \mathbf{x} + b)
$$


- y = saida
- f = funçaõ de ativação
- w = pesos
- x = entrada
- b = vies(bias)

Na prática, com o Keras, a definição da camada Dense corresponde basicamente à definição dos termos dessa equação, da seguinte forma:

**X (Entradas):** Refere-se à quantidade de "colunas" ou variáveis de entrada que irão alimentar o "neurônio", ou seja, os valores que serão usados para calcular e fazer a inferência, correspondendo às entradas do perceptron. Esse termo é definido na primeira linha do `keras.Sequential` com `Input(shape=X)`.

**W (Pesos):** Existe um peso para cada entrada que o modelo recebe. Cada peso é um valor (ou coeficiente) que multiplica a entrada correspondente, conforme descrito na equação. No Keras, os pesos são inicializados pelo parâmetro [`keras.initializers`](https://keras.io/api/layers/initializers/), onde existem várias opções de inicialização. Caso não seja especificado, eles serão iniciados com [`glorot_uniform`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/core.py).

**b (Viés):** O viés é semelhante aos pesos, mas ele é somado ao produto das entradas e dos pesos, permitindo mais flexibilidade ao modelo. Isso garante que, mesmo que as entradas sejam zero, a saída não seja necessariamente zero. No Keras, o viés é inicializado pelo parâmetro `bias_initializer`, que por padrão é inicializado com zeros.

**Y (Saída):** A última camada define a saída, que no nosso caso foi definida com 9 unidades, correspondendo ao número de saídas (diagnósticos) presentes no dataset.

Após a definição desses parâmetros, o próximo passo é a função soma: Ela agrega os valores das entradas multiplicadas pelos pesos, gerando o "valor de ativação", que será passado para a função de ativação. Esse processo não é um parâmetro da camada Dense, mas ocorre automaticamente dentro dela.

**Função de Ativação:** Ela recebe o valor de ativação e gera a saída. As camadas ocultas usam a função de ativação ReLU (Retificação Linear), enquanto a camada de saída utiliza a função sigmoid, que é adequada para problemas de classificação multirrótulo (multilabel).

- ReLu - A função relu de retificação linear executa uma ação bem simples, se o valor é negativo tem como saída 0 e se for positivo é devolvido o próprio valor. A ReLu é mais comumente usada por ser computacionalmente eficiente e também conseguir lidar bem com o [problema de gradiente desaparecido](https://www.deeplearningbook.com.br/o-problema-da-dissipacao-do-gradiente/) que de forma resumida é quando os valores os valores dos gradientes, que são necessários para atualizar os pesos durante o aprendizado, se tornam muito pequenos, tornando o processo de aprendizado lento ou ineficaz.

``` activation='relu' ```

- Sigmoid - A função Sigmoid transforma qualquer valor de entrada em um valor entre 0 e 1, o que a torna ideal para problemas de classificação binária. Valores grandes se aproximam de 1, e valores pequenos se aproximam de 0, permitindo que a saída seja interpretada como uma probabilidade.

``` activation='sigmoid' ```

- Softmax - A função Softmax é usada em redes neurais para problemas de classificação multiclasse. Ela transforma um conjunto de valores de entrada em probabilidades, distribuindo-as de forma que a soma seja igual a 1. Cada saída representa a probabilidade de pertencer a uma das classes, sendo a mais alta geralmente a previsão do modelo.

``` activation='sigmoid' ```

**O aprendizado aconteça nas funções de somas e ativação**

Após a construção e execução do código vamos ter criado a estrutura do modelo, o Keras oferece um método chamado summary() onde é possível visualizar as suas características:

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 64)                  │             960 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 9)                   │             297 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 3,337 (13.04 KB)
 Trainable params: 3,337 (13.04 KB)
 Non-trainable params: 0 (0.00 B)

```
Os valores são calculados da seguinte forma

Valor anterior * neurônios + neurônios

A primeira linha com 960 parâmetros se refere:

14(entrada) * 64(neurônios) + 64 (neurônios) = 960
64 * 32 + 32 = 2080
32 * 9 + 9 = 297

## Treinamento





