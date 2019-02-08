import cv2
import numpy as np 

classificador = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
classificadorOlho = cv2.CascadeClassifier('haarcascade-eye.xml')
camera = cv2.VideoCapture(0) # 0 pq usar apenas 1 camera

amostra = 1
numeroAmostra = 50
id = input('Digite seu identificador: ')

largura, altura = 220, 220

print("Capturando as faces...")

while True:
	conectado, imagem = camera.read()
	imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	facesDetectadas = classificador.detectMultiScale(imagemCinza,
													scaleFactor = 1.5,
													minSize = (100, 100))

	for (x, y, l, a) in facesDetectadas:
		cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0, 255), 2)
		regiao = imagem[y:y + a, x:x + l]
		regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
		olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

		for (ox, oy, ol, oa) in olhosDetectados:
			cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)


			if cv2.waitKey(1) & 0xFF == ord('q'):
				if np.average(imagemCinza) > 110:
					imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
					cv2.imwrite('fotos/pessoa.{}.{}.jpg'.format(id, amostra), imagemFace)
					print("Foto %s capturada com sucesso!"%(amostra))
					amostra += 1	

	cv2.imshow('Face', imagem)
	#cv2.waitKey(1)

	if amostra >= numeroAmostra + 1:
		break

camera.release()
cv2.destroyAllWindows()