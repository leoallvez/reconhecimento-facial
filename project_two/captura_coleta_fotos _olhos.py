import cv2
import numpy as np


classificadorRosto = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
classificadorOlhos = cv2.CascadeClassifier("haarcascade-eye.xml")
#Definindo a camera
camera = cv2.VideoCapture(0)

amostra = 1
numeroAmostras = 25;
id = input('Digite o seu identificador: ')
# tamanho da imagens que serão coletadas
larguraImagemCapturada, alturaImagemCapturada = 220, 220
print('capturando faces....')

# loop de camptura
while (True) :
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorRosto.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150, 150))

    for(x, y, largura, altura) in facesDetectadas:
        #relangulo da face.
        cv2.rectangle(imagem,(x, y), (x + largura, y + altura), (0, 0, 255), 2)
        # retagulos dos olhos
        # pegando a face que foi detectada.
        regiao = imagem[y:y + altura, x:x + largura]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlhos.detectMultiScale(regiao)
        for (ox, oy, olargura, oaltura) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + olargura, oy + oaltura), (0, 0, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #teste de luminisidade
                print('luminosidade: ' + str(np.average(imagemCinza)))
                if np.average(imagemCinza) > 100:
                    imagemFace = cv2.resize(imagemCinza[y:y + altura, x:x + largura], (larguraImagemCapturada, alturaImagemCapturada))

                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                    print('[foto: '+ str(amostra) + " capturada com sucesso]")
                    amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break

print('faces capturadas com sucesso')
#liberando memoria
camera.release()
#fechando todas janelas
cv2.destroyAllWindows()
