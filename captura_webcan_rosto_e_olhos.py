import cv2

classificadorRosto = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
classificadorOlhos = cv2.CascadeClassifier("haarcascade-eye.xml")
#Definindo a camera
camera = cv2.VideoCapture(0)

# loop de camptura
while (True) :
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorRosto.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100, 100))

    for(x, y, largura, altura) in facesDetectadas:
        #relangulo da face.
        cv2.rectangle(imagem,(x, y), (x + largura, y + altura), (0, 0, 255), 2)
        #retagulos dos olhos
        # pegando a face que foi detectada.
        regiao = imagem[y:y + altura, x:x + largura ]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlhos.detectMultiScale(regiao)
        for(ox, oy, olargura, oaltura) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + olargura, oy + oaltura), (0, 0, 255), 2)


    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
#liberando memoria
camera.release()
#fechando todas janelas
cv2.detroyAllWindows()
