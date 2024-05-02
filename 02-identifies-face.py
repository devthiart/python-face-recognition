from PIL import Image, ImageDraw
# from IPython.display import display
import face_recognition

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

# ********** Exemplo **********
# Armazena o endereço da imagem
image_url = "./images/solo/billgates.jpeg"
# Com o endereço da imagem, carrego a imagem no face_recognition
image = face_recognition.load_image_file(image_url)

# Executo o face_recognition para encontrar rostos na foto e armazenar as posições destes rostos
# [top, right, bottom, left]
face_locations = face_recognition.face_locations(image)

# Executo a função que criei para desenhar na imagem
image_drawn = draw_on_picture(image, face_locations)
# Mostra a imagem (no Deepnote)
# display(image_drawn)

# Abre a imagem
image_drawn.show()
