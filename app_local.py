import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Carregar o modelo treinado
model_path = '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/modelo_personagens.h5'  # Ajuste para o caminho correto
model = load_model(model_path)

# Função para prever o personagem
def predict_character(image):
    img = cv2.resize(image, (64, 64))  # Redimensiona para o tamanho esperado
    img = np.expand_dims(img, axis=0) / 255.0  # Normaliza e adiciona dimensão
    start_time = time.time()  # Marca o tempo de início da previsão
    prediction = model.predict(img)
    processing_time = time.time() - start_time  # Calcula o tempo de processamento
    return np.argmax(prediction), processing_time

# Mapeando os personagens
characters = {
    'Homer': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/personagens/Homer.png',
    'Marge': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/personagens/Marge.png',
    'Bart': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/personagens/Bart.png',
    'Lisa': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/personagens/Lisa.png',
    'Maggie': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/personagens/Maggie.png'
}

# Inicializando o contador de acertos e tentativas na sessão
if 'correct_counts' not in st.session_state:
    st.session_state.correct_counts = {name: 0 for name in characters.keys()}
if 'total_attempts' not in st.session_state:
    st.session_state.total_attempts = 0

# Configuração da interface
st.markdown("<h1 style='text-align: center; color: #FFCC00; font-family: Comic Sans MS;'>CLASSIFICADOR DE PERSONAGENS</h1>", unsafe_allow_html=True)

# Exibir a caixa de seleção para escolher qual personagem analisar
selected_character = st.selectbox("Selecione um personagem para análise:", list(characters.keys()))

# Exibir imagens dos personagens em uma grade
cols = st.columns(3)  # Três colunas para exibir as imagens

for i, (name, image_file) in enumerate(characters.items()):
    img = Image.open(image_file)  # Abre a imagem com Pillow

    with cols[i % 3]:  # Distribuir as imagens nas colunas
        st.image(img, caption=name, use_column_width=True)  # Exibe a imagem

        # Cria um botão que ativa o efeito de clique ao clicar na imagem
        if st.button(f'Selecionar {name}', key=name):  # Botão para selecionar o personagem
            # Prever o personagem
            selected_image = cv2.imread(image_file)  # Lê a imagem correspondente
            predicted_class, processing_time = predict_character(selected_image)

            # Incrementar tentativas totais
            st.session_state.total_attempts += 1

            if selected_character.lower() == name.lower():  # Compara a seleção do usuário com o nome
                st.success("Você acertou!", icon="✅")
                st.session_state.correct_counts[name] += 1  # Incrementa a contagem de acertos
                # Mensagem de sucesso
                st.write("A rede neural conseguiu identificar corretamente o personagem!")
                # Chamar função de confete
                st.markdown('<script>fireConfetti();</script>', unsafe_allow_html=True)
            else:
                st.error("Tente novamente!", icon="❌")
                # Mensagem de erro
                st.write("A rede neural identificou que a imagem não é o personagem escolhido!")

# Exibir contadores de acertos na lateral direita
st.sidebar.header("Contagem de Acertos")
for name, count in st.session_state.correct_counts.items():
    st.sidebar.write(f"{name}: {count} vez(es)")

# Calcular percentual de acerto
if st.session_state.total_attempts > 0:
    total_correct = sum(st.session_state.correct_counts.values())
    accuracy_percentage = (total_correct / st.session_state.total_attempts) * 100
    # Destacar a mensagem de percentual de acertos
    st.sidebar.markdown(f"**Percentual de Acertos:** {accuracy_percentage:.2f}%")
else:
    st.sidebar.write("Ainda não houve tentativas.")

# Mostrar o tempo de processamento
st.sidebar.markdown(f"**Tempo de Processamento:** {processing_time:.4f} segundos")