import face_recognition
import numpy as np
from PIL import Image, ImageDraw
# from IPython.display import display

# ***** Aprendendo com uma foto *****

# Carrega uma imagem de amostra e aprende como reconhece-la
obama_image = face_recognition.load_image_file("./images/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Carrega uma segunda imagem de amostra e aprende como reconhece-la
biden_image = face_recognition.load_image_file("./images/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Carrega uma terceira imagem de amostra e aprende como reconhece-la
rock_image = face_recognition.load_image_file("./images/compare-face/the-rock.jpg")
rock_face_encoding = face_recognition.face_encodings(rock_image)[0]

# Carrega uma quarta imagem de amostra e aprende como reconhece-la
hanks_image = face_recognition.load_image_file("./images/compare-face/tom-hanks.webp")
hanks_face_encoding = face_recognition.face_encodings(hanks_image)[0]

# Cria matrizes de codificações faciais conhecidas com seus nomes
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    rock_face_encoding,
    hanks_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "The Rock",
    "Tom Hanks"
]

# Imprime quantos rostos foram aprendidos
print('Rostos aprendidos e codificados de ', len(known_face_encodings), 'imagens.')

# ***** Analisando foto com rostos conhecidos *****

# Carrega uma imagem com um rosto desconhecido
unknown_image = face_recognition.load_image_file("./images/compare-face/the-rock-tom-hanks.webp")

# Encontre todos os rostos e codificações de rostos na imagem desconhecida
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Converte a imagem para a PIL-format, possibilitando que consigamos trabalhar com a imagem.
# Acesse a documentação de PIL/Pillow para mais imformações: http://pillow.readthedocs.io/
pil_image = Image.fromarray(unknown_image)
# Crie uma instância do ImageDraw para que consigamos desenhar sobre a imagem.
draw = ImageDraw.Draw(pil_image)

# Loop para cada rosto encontrado na imagem
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Faz a comparação dos rostos encontrados com os rostos conhecidos.
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    # Verifica se algum rosto se parece com os conhecidos
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Desenhe uma caixa ao redor do rosto usando o módulo Pillow
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Desenhe uma etiqueta com o nome abaixo da face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove a biblioteca de desenhos da memória conforme a documentação do Pillow
del draw

# Mostra o resultado
# display(pil_image)
pil_image.show()
