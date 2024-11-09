# Meu Projeto

!Print do Projeto 1
!Print do Projeto 2

### Classificação de Imagens: Homer vs Bart usando Redes Neurais Convolucionais (CNN)

Este projeto utiliza redes neurais convolucionais (CNN) para a classificação de imagens dos personagens Homer e Bart do desenho animado *Os Simpsons*. O objetivo é treinar um modelo capaz de identificar corretamente os personagens em imagens e marcar o local onde eles são encontrados com um retângulo ao redor.

### Destaques do Projeto

- **Modelo Treinado com CNN:** Utiliza redes neurais convolucionais (CNN) para uma classificação precisa das imagens dos personagens.
  
- **Previsão em Tempo Real:** O modelo permite prever a identidade dos personagens em imagens de forma rápida e eficiente.

- **Interface Intuitiva:** A aplicação oferece uma interface amigável, onde os usuários podem selecionar diretamente os personagens e visualizar os resultados das previsões.

- **Efeito Visual de Confete:** Um efeito visual de confete é acionado quando a previsão está correta, proporcionando uma experiência interativa e divertida para o usuário.

- **Painel de Contagem de Acertos:** A aplicação conta e exibe quantas vezes cada personagem foi corretamente identificado, permitindo aos usuários acompanhar seu desempenho.

- **Implementação Online:** O projeto foi desenvolvido para ser executado online, permitindo que os usuários acessem e utilizem a aplicação sem necessidade de instalação. Você pode testá-lo diretamente no seguinte link: ***Teste Online***.

### Pré-requisitos

Certifique-se de que as seguintes bibliotecas estão instaladas. Para facilitar o processo, foi gerado o arquivo requirements.txt, que contém todas as dependências necessárias para rodar o projeto.

### Instalação das Dependências

Clone o repositório do projeto para o seu ambiente local:
```bash
git clone https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS






















### Classificação de Imagens: Homer vs Bart usando Redes Neurais Convolucionais (CNN)

Este projeto utiliza redes neurais convolucionais (CNN) para a classificação de imagens dos personagens Homer e Bart do desenho animado *Os Simpsons*. O objetivo é treinar um modelo capaz de identificar corretamente os personagens em imagens e marcar o local onde eles são encontrados com um retângulo ao redor.


### Destaques do Projeto

- **Modelo Treinado com CNN:** Utiliza redes neurais convolucionais (CNN) para uma classificação precisa das imagens dos personagens.
  
- **Previsão em Tempo Real:** O modelo permite prever a identidade dos personagens em imagens de forma rápida e eficiente.

- **Interface Intuitiva:** A aplicação oferece uma interface amigável, onde os usuários podem selecionar diretamente os personagens e visualizar os resultados das previsões.

- **Efeito Visual de Confete:** Um efeito visual de confete é acionado quando a previsão está correta, proporcionando uma experiência interativa e divertida para o usuário.

- **Painel de Contagem de Acertos:** A aplicação conta e exibe quantas vezes cada personagem foi corretamente identificado, permitindo aos usuários acompanhar seu desempenho.

- **Implementação Online:** O projeto foi desenvolvido para ser executado online, permitindo que os usuários acessem e utilizem a aplicação sem necessidade de instalação. Você pode testá-lo diretamente no seguinte link: ***[Teste Online](https://cp5rnredecnn-simpsons-fc23pczypcqkf3tcmnmsdb.streamlit.app/)***.

### Pré-requisitos

Certifique-se de que as seguintes bibliotecas estão instaladas. Para facilitar o processo, foi gerado o arquivo requirements.txt, que contém todas as dependências necessárias para rodar o projeto.

### Instalação das Dependências

Clone o repositório do projeto para o seu ambiente local:
```bash
git clone https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS

Navegue até o diretório do projeto:
cd Homer_vs_Bart_CNN

Utilize o pip para instalar as dependências:
pip install -r requirements.txt

Estrutura do Projeto

	•	dataset/: Pasta contendo as imagens de treino e teste dos personagens Homer e Bart.
	•	Homer_vs_Bart_CNN.ipynb: Arquivo Jupyter Notebook com o código principal para o treinamento e avaliação do modelo.
	•	model/: Diretório onde o modelo treinado é salvo.
	•	requirements.txt: Arquivo que lista todas as bibliotecas e dependências necessárias para rodar o projeto.

Funcionamento do Código

O modelo é baseado em uma rede neural convolucional (CNN) construída com o Keras e TensorFlow. O código segue o seguinte fluxo:
	1.	Pré-processamento dos Dados: As imagens são carregadas e redimensionadas para o formato esperado pela CNN.
	2.	Divisão do Dataset: Utiliza-se a função train_test_split do scikit-learn para dividir os dados em conjuntos de treino e teste.
	3.	Criação da CNN: O modelo CNN é construído com camadas convolucionais e pooling, seguido de camadas densas para a classificação.
	4.	Treinamento: O modelo é treinado com os dados de treino, utilizando aumentação de imagens para melhorar a generalização.
	5.	Avaliação: O modelo é avaliado no conjunto de teste para medir sua precisão.
	6.	Predição e Visualização: Após o treinamento, o modelo faz previsões sobre novas imagens e exibe a imagem do personagem identificado, destacando-o com um retângulo ao redor.

### Código do Projeto

```python
# Instalação das bibliotecas necessárias
# Aqui, estou instalando as bibliotecas que preciso para o projeto
!pip install streamlit opencv-python tensorflow numpy

# Importação das bibliotecas
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregando o modelo treinado
# Estou carregando o modelo que treinei anteriormente
model = load_model('/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/Treino_modelo/modelo_personagens.h5')  # Ajuste o caminho conforme necessário

# Função para prever o personagem
def predict_character(image):
    # Redimensiona a imagem para o tamanho esperado pela CNN
    img = cv2.resize(image, (64, 64))
    # Normaliza e adiciona uma nova dimensão à imagem
    img = np.expand_dims(img, axis=0) / 255.0
    # Faz a previsão usando o modelo carregado
    prediction = model.predict(img)
    return np.argmax(prediction)

# Interface Streamlit
# Configurando o título da aplicação
st.title("Classificador de Personagens")

# Seletor de personagens
characters = ['Homer', 'Bart', 'Lisa', 'Maggie']
selected_character = st.selectbox("Selecione um personagem", characters)

# Uploader de imagens
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ler a imagem
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Exibir a imagem
    st.image(image, channels="BGR", caption='Imagem carregada', use_column_width=True)

    # Prever o personagem
    prediction = predict_character(image)
    character_names = ['Homer', 'Bart', 'Lisa', 'Maggie']

    # Exibir o resultado
    st.write(f"O modelo previu: {character_names[prediction]}")

    # Comparo a previsão do modelo com a seleção do usuário
    if character_names[prediction] == selected_character:
        st.success("Você acertou!")
    else:
        st.error("Tente novamente!") ```

### Análise do Projeto

Este projeto demonstra a aplicação prática de redes neurais convolucionais para a classificação de imagens, um campo de grande relevância em inteligência artificial. A implementação fornece uma interface interativa que não só permite a identificação de personagens, mas também envolve os usuários de forma lúdica, como demonstrado pelo efeito de confete e pelo painel de contagem de acertos.

### Potenciais Usos do Código

	1.	Educação: O modelo pode ser utilizado em ambientes educacionais para ensinar conceitos de machine learning e inteligência artificial, facilitando a compreensão de como as CNNs funcionam.
	2.	Jogos: A lógica de classificação pode ser adaptada para jogos que requerem identificação de personagens ou objetos em imagens, aumentando a interatividade e o engajamento.
	3.	Aplicações de Reconhecimento de Imagens: O código pode ser modificado para reconhecer diferentes classes de imagens, permitindo seu uso em segurança, monitoramento e análise de dados visuais.
	4.	Desenvolvimento de APIs: O modelo pode ser exposto como uma API que aceita imagens e retorna classificações, permitindo sua integração em aplicativos móveis ou web.
	5.	Análise de Dados Visuais: Empresas que lidam com grandes volumes de dados visuais podem usar o modelo para classificar e organizar essas informações de maneira eficiente.






