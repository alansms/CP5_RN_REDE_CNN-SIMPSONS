import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

# ID do arquivo do modelo no Google Drive
file_id = '1jJ77L4X6YGlLfFYgvaOWxSBHeWqrcOJ7'  # Substitua pelo seu FILE_ID
model_path = 'modelo_personagens.h5'  # Nome do arquivo que será salvo

# Baixar o modelo do Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)

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
    'Marge': 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/MArge.png',
    'Bart': 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/Homer.png',
    'Lisa': 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/Lisa.png',
    'Maggie': 'https://github.com/alansms/CP5_RN_REDE_CNN-SIMPSONS/blob/main/Maggie.png'
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

function addClickEffect(elementId) {
    var element = document.getElementById(elementId);
    element.classList.add("clicked");
    setTimeout(() => {
        element.classList.remove("clicked");
    }, 300);
}
</script>
<style>
.clicked {
    transform: scale(0.95);
    transition: transform 0.1s ease;
}
</style>
""", unsafe_allow_html=True)

# Exibir imagens dos personagens em uma grade
cols = st.columns(3)  # Três colunas para exibir as imagens

for i, (name, image_file) in enumerate(characters.items()):
    img = Image.open(image_file)  # Abre a imagem com Pillow

    with cols[i % 3]:  # Distribuir as imagens nas colunas
        st.image(img, caption=name, use_column_width=True)  # Exibe a imagem

        # Cria um botão que ativa o efeito de clique ao clicar na imagem
        if st.button(f'Selecionar {name}'):  # Botão para selecionar o personagem
            # Chamar função de efeito de clique
            st.markdown(f'<script>addClickEffect("{name}");</script>', unsafe_allow_html=True)

            # Prever o personagem
            selected_image = cv2.imread(image_file)  # Lê a imagem correspondente
            predicted_class = predict_character(selected_image)

            if predicted_class == i:  # Compara a classe prevista com o índice do personagem
                st.success("Você acertou!", icon="✅")
                st.session_state.correct_counts[name] += 1  # Incrementa a contagem de acertos
                # Chamar função de confete usando o método adequado
                st.markdown('<script>fireConfetti();</script>', unsafe_allow_html=True)
            else:
                st.error("Tente novamente!", icon="❌")

# Exibir contadores de acertos na lateral direita
st.sidebar.header("Contagem de Acertos")
for name, count in st.session_state.correct_counts.items():
    st.sidebar.write(f"{name}: {count} vez(es)")
