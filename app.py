import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Carregar o modelo treinado
model_path = 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/modelo_personagens.h5'  # Ajuste para o caminho correto
model = load_model(model_path)

# Função para prever o personagem usando Pillow
def predict_character(image_path):
    img = Image.open(image_path)  # Abre a imagem com Pillow
    img = img.resize((64, 64))  # Redimensiona para o tamanho esperado
    img_array = np.array(img) / 255.0  # Normaliza
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão
    prediction = model.predict(img_array)
    return np.argmax(prediction)

# Mapeando os personagens
characters = {
    'Homer': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/Homer.jpg',
    'Marge': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/Margie.jpg',
    'Bart': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/Bart.jpg',
    'Lisa': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/Lisa.jpg',
    'Maggie': '/Users/alansms/PycharmProjects/Fiap/MACHINE LEARNING/5ºCheckPoint_Redes_Neurais/Maggie.jpg'
}

# Configuração da interface
st.title("Jogo da Memória - Classificador de Personagens")
st.markdown("<style>h1 {font-family: 'Comic Sans MS'; color: #FFCC00;}</style>", unsafe_allow_html=True)

# Adicionar script do confete
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.4.0/dist/confetti.browser.min.js"></script>
<script>
function fireConfetti() {
    var count = 200; // Quantidade de confetes
    var defaults = {
        origin: { y: 0.7 } // Origem do confete
    };

    function fire(particleRatio, opts) {
        confetti(Object.assign({}, defaults, opts, { particleCount: Math.floor(count * particleRatio) }));
    }

    fire(0.25, { spread: 26, startVelocity: 55 });
    fire(0.2, { spread: 60 });
    fire(0.35, { spread: 100, decay: 0.91, scalar: 1 });
    fire(0.1, { spread: 120, decay: 0.92, scalar: 1.2 });
}
</script>
""", unsafe_allow_html=True)

# Caixa de seleção para escolher qual personagem analisar
selected_character = st.selectbox("Selecione um personagem para análise:", list(characters.keys()))

# Exibir imagens dos personagens em uma grade
cols = st.columns(3)  # Três colunas para exibir as imagens

for i, (name, image_file) in enumerate(characters.items()):
    img = Image.open(image_file)  # Abre a imagem com Pillow

    with cols[i % 3]:  # Distribuir as imagens nas colunas
        st.image(img, caption=name, use_column_width=True)
        if st.button(f'Selecionar {name}'):
            # Prever o personagem
            predicted_class = predict_character(image_file)  # Chama a função com o caminho da imagem

            if selected_character.lower() == name.lower():
                st.success("Você acertou!", icon="✅")
                # Chamar função de confete usando o método adequado
                st.markdown('<script>fireConfetti();</script>', unsafe_allow_html=True)
            else:
                st.error("Tente novamente!", icon="❌")
