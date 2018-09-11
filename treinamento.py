import os
import cv2
import numpy as np

lbph = cv2.face.LBPHFaceRecognizer_create()
eigenFace = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(10)
        ids.append(id)
        faces.append(imagemFace)
    return np.array(ids), faces


ids, faces = getImagemComId()

print(faces)