from PIL import Image, ImageDraw
from IPython.display import display
import face_recognition

# Caminho das imagens armazenado em uma variavel do tipo dicionario
images = {
    'billgates': './images/solo/billgates.jpeg',
    'mulher-negra': './images/solo/Dia-da-mulher-negra.jpg',
    'mulher-branca': './images/solo/model-girl-with.jpg',
    'time': './images/group/time_corinthians.jpg',
    'anime': './images/cartoon/frieren-group.jpg',
    'cartoon': './images/cartoon/hilda.jpg',
    'arte-retrato': './images/art/vermeer.JPG',
    'arte-pessoas': './images/art/observation_by_valerie_lin.jpg',
    'arte-artista': './images/art/tranquility_by_valerie_lin.jpg',
    'arte-ilusao': './images/art/art_by_octavio-ocampo.jpg',
    'desenho': './images/art/mother-by-albrecht-durer.jpg',
    'cachorro': './images/animals/cachorro-com-cara-de-humano.jpeg',
    'gato': './images/animals/cat.jpg',
    'chimpanze': './images/animals/chimpanze.jpg',
}

def draw_on_picture(image, positions):
    # Converte a imagem para a PIL-format, possibilitando que consigamos trabalhar com a imagem.
    # Acesse a documentação de PIL/Pillow para mais imformações: http://pillow.readthedocs.io/
    pil_image = Image.fromarray(image)
    # Crie uma instância do ImageDraw para que consigamos desenhar sobre a imagem.
    draw = ImageDraw.Draw(pil_image)

    # dentro da imagem, para cada posição que recebermos
    for position in positions:
        # Organiza cada valor da posição para ficar mais intuitivo de ler.
        top = position[0]
        right = position[1]
        bottom = position[2]
        left = position[3]

        # Desenha um retângulo em volta do rosto usando o Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)
    
    # Remove a biblioteca de desenhos da memória conforme a documentação do Pillow
    del draw

    # Retorna a imagem desenhada
    return pil_image

# *** Analisa todas as imagens ***
# For para fazer a analise de imagem por imagem
for key in images.keys():
    # Pega a URL da imagem
    image_url = images[key]
    # Com a URL da imagem, carrego a imagem no face_recognition
    image = face_recognition.load_image_file(image_url)

    # Executo o face_recognition para encontrar rostos na foto e armazenar as posições destes rostos
    # face_locations = [(top, right, bottom, left), ...]
    face_locations = face_recognition.face_locations(image)

    # Mostra qual a key da imagem
    print("*** "+ key + " ***")

    # Mostra quantos rostos foram identificados
    print("\nForam identificadas " + str(len(face_locations)) + " pessoas nesta foto.")

    # Executo a função que criei para desenhar na imagem
    image_drawn = draw_on_picture(image, face_locations)

    # Mostra a imagem com os rostos destacados (no Deepnote)
    display(image_drawn)

    # Abre a Imagem (Abre todas as imagens, então preferi deixar comentado.)
    # image_drawn.show()
    
