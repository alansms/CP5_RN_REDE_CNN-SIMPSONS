import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import requests

# ID do arquivo do modelo no Google Drive
file_id = '1jJ77L4X6YGlLfFYgvaOWxSBHeWqrcOJ7'
model_path = 'modelo_personagens.h5'  # Nome do arquivo que será salvo

# Função para baixar o modelo
def download_model(file_id, model_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
    else:
        st.error("Erro ao baixar o modelo.")

# Baixar o modelo do Google Drive
download_model(file_id, model_path)

# Carregar o modelo treinado
model = load_model(model_path)

# Função para prever o personagem
def predict_character(image):
    img = image.resize((64, 64))  # Redimensiona para o tamanho esperado
    img_array = np.array(img) / 255.0  # Normaliza
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão
    prediction = model.predict(img_array)
    return np.argmax(prediction)

# Mapeando os personagens com links brutos
characters = {
    'Homer': 'https://raw.githubusercontent.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/main/Homer.png',
    'Marge': 'https://raw.githubusercontent.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/main/Marge.png',
    'Bart': 'https://raw.githubusercontent.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/main/Bart.png',
    'Lisa': 'https://raw.githubusercontent.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/main/Lisa.png',
    'Maggie': 'https://raw.githubusercontent.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/main/Maggie.png'
}

# Inicializando o contador de acertos na sessão
if 'correct_counts' not in st.session_state:
    st.session_state.correct_counts = {name: 0 for name in characters.keys()}

# Configuração da interface
st.markdown("<h1 style='text-align: center; color: #FFCC00; font-family: Comic Sans MS;'>CLASSIFICADOR DE PERSONAGENS</h1>", unsafe_allow_html=True)

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

# Exibir imagens dos personagens em uma grade
cols = st.columns(3)  # Três colunas para exibir as imagens

for name, image_file in characters.items():
    img = Image.open(requests.get(image_file, stream=True).raw)  # Abre a imagem com Pillow a partir da URL

    with cols:  # Distribuir as imagens nas colunas
        st.image(img, caption=name, use_column_width=True)  # Exibe a imagem

        if st.button(f'Selecionar {name}', key=name):  # Botão para selecionar o personagem
            # Prever o personagem
            predicted_class = predict_character(img)  # Chama a função com a imagem

            if predicted_class == list(characters.keys()).index(name):  # Verifica se a previsão está correta
                st.success("Você acertou!", icon="✅")
                st.session_state.correct_counts[name] += 1  # Incrementa a contagem de acertos
                st.markdown('<script>fireConfetti();</script>', unsafe_allow_html=True)
            else:
                st.error("Tente novamente!", icon="❌")

# Exibir contadores de acertos na lateral direita
st.sidebar.header("Contagem de Acertos")
for name, count in st.session_state.correct_counts.items():
    st.sidebar.write(f"{name}: {count} vez(es)")
