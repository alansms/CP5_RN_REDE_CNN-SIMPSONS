import streamlit as st
import numpy as np
import requests
from tensorflow.keras.models import load_model
from PIL import Image

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
    img = cv2.resize(image, (64, 64))  # Redimensiona para o tamanho esperado
    img = np.expand_dims(img, axis=0) / 255.0  # Normaliza e adiciona dimensão
    prediction = model.predict(img)
    return np.argmax(prediction)

# Mapeando os personagens
characters = {
    'Homer': 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/Homer.png',
    'Marge': 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/Marge.png',
    'Bart': 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/Bart.png',
    'Lisa': 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/Lisa.png',
    'Maggie': 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/Maggie.png'
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

# Exibir imagens dos personagens em uma grade
cols = st.columns(3)  # Três colunas para exibir as imagens

for name, image_file in characters.items():
    img = Image.open(image_file)  # Abre a imagem com Pillow

    with cols:  # Distribuir as imagens nas colunas
        st.image(img, caption=name, use_column_width=True)  # Exibe a imagem
        if st.button(f'Selecionar {name}'):  # Botão para selecionar o personagem
            # Prever o personagem
            selected_image = cv2.imread(image_file)  # Lê a imagem correspondente
            predicted_class = predict_character(selected_image)

            if name.lower() == list(characters.keys())[predicted_class].lower():  # Compara a seleção do usuário com o nome
                st.success("Você acertou!", icon="✅")
                # Chamar função de confete usando o método adequado
                st.markdown('<script>fireConfetti();</script>', unsafe_allow_html=True)
            else:
                st.error("Tente novamente!", icon="❌")
