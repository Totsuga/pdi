import numpy as np
import matplotlib.pyplot as pyp

def imread(local):
	if(local[-3:] == 'png'):
		return np.uint8(np.round(pyp.imread(local)*255))
	else:
		return np.uint8(np.round(pyp.imread(local)))

def nchannels(imagem):
    return np.size(imagem[0][0])

def size(imagem):
    return [imagem.shape[0],imagem.shape[1]]

def rgbToGray(imagem):
	return np.dot(imagem, [0.299, 0.587, 0.114]).astype(int)

def imreadgray(local):
	imagem = imread(local)
	if nchannels(imagem) == 1:
		return imagem
	else:
		if nchannels(imagem) == 3:
			return rgbToGray(imagem)

def imshow(imagem):
	if nchannels(imagem) == 1:
		pyp.imshow(imagem, cmap='gray')
		pyp.show()
	else:
		if nchannels(imagem) == 3:
			pyp.imshow(imagem)
			pyp.show()

def contrast(imagem, r, m):
    if nchannels(imagem) == 1:
        im = np.zeros((imagem.shape[0],imagem.shape[1]), dtype=np.uint8)
        for i in range(imagem.shape[0]):
            for j in range(imagem.shape[1]):
                im[i][j] = r*(imagem[i][j]-m)+m
        return im
    else:
        im = np.zeros((imagem.shape[0],imagem.shape[1],imagem.shape[2]), dtype=np.uint8)
        for i in range(imagem.shape[0]):
            for j in range(imagem.shape[1]):
                for k in range(imagem.shape[2]):
                    im[i][j][k] = r*(imagem[i][j][k]-m)+m
        return im

def thresh(imagem, valor):
	m = 255
	im = np.zeros((imagem.shape[0],imagem.shape[1]), dtype=np.uint8)
	for i in range(imagem.shape[0]):
		for j in range(imagem.shape[1]):
			if imagem[i][j] >= valor:
				im[i][j] = m
	return im

def negative(imagem):
	if nchannels(imagem) == 1:
		im = np.zeros((imagem.shape[0],imagem.shape[1]), dtype=np.uint8)
		for i in range(imagem.shape[0]):
			for j in range(imagem.shape[1]):
				im[i][j] = 255 - imagem[i][j]
		return im
	else:
		im = np.zeros((imagem.shape[0],imagem.shape[1], imagem.shape[2]), dtype=np.uint8)
		for i in range(imagem.shape[0]):
			for j in range(imagem.shape[1]):
				for k in range(imagem.shape[2]):
					im[i][j][k] = 255 - imagem[i][j][k]
		return im

def hist(imagem):
	if nchannels(imagem) == 1:
		hist = np.zeros((256,1), dtype=np.uint)
		for i in range(imagem.shape[0]):
			for j in range(imagem.shape[1]):
				hist[int(imagem[i][j])]+=1
		return hist
	else:
		if nchannels(imagem) == 3:
			hist = np.zeros((256,3), dtype=np.uint)
			for i in range(imagem.shape[0]):
				for j in range(imagem.shape[1]):
					hist[int(imagem[i][j][0]),0]+=1
					hist[int(imagem[i][j][1]),1]+=1
					hist[int(imagem[i][j][2]),2]+=1
		return hist

## Resolver a parte de bin != 1
def showhist(hist, bin=1):
	x_axis = []
	for i in range(256):
		x_axis.append(i)
	if hist.shape[1] == 1:
		y_axis = []
		if(bin != 1):
			s = 0
			for i in range(len(hist.transpose()[0])):
				s += hist.transpose()[0][i]
				if (i+1)%bin == 0:
					y_axis.append(s)
					s = 0
				else: 
					if i == len(hist.transpose()[0])-1:
						y_axis.append(s)
		else:
			y_axis = hist.transpose()[0]
		pyp.bar(x_axis, y_axis, color='black', align='center')
	else:
		if hist.shape[1] == 3:
			red_bar = pyp.bar(x_axis, hist.transpose()[0], color='red', align='center')
			green_bar = pyp.bar(x_axis, hist.transpose()[1], color='green', align='center')
			blue_bar = pyp.bar(x_axis, hist.transpose()[2], color='blue', align='center')
			pyp.show()

def histeq(imagem):
	histogram = hist(imagem)
	corr = len(imagem)*len(imagem[0])
	for i in range(len(histogram)):
		if i != 0:
			histogram[i] = histogram[i]+histogram[i-1]
	histogram = histogram*(1/corr)*255
	imagem_out = np.zeros((len(imagem),len(imagem[0])))
	for i in range(len(imagem)):
		for j in range(np.size(imagem[0])):
			imagem_out[i][j] = histogram[imagem[i][j]]
	return imagem_out


def extrapola(matriz, linhas, colunas):
	corr_linha = int((linhas-1)/2)
	corr_coluna = int((colunas-1)/2)
	m = np.zeros((len(matriz)+(linhas-1),len(matriz[0])+(colunas-1)))
	for i in range(len(m)):
		for j in range(len(m[0])):
			pos_x = i - corr_linha
			pos_y = j - corr_coluna
			if pos_x < 0:
				pos_x = 0
			if pos_y < 0:
				pos_y = 0
			if pos_x > len(matriz)-1:
				pos_x = len(matriz)-1
			if pos_y > len(matriz[0])-1:
				pos_y = len(matriz[0])-1
			m[i][j] = matriz[pos_x][pos_y]
	return m

def convolve(imagem, mascara):
	corr_linha = int((len(mascara)-1)/2)
	corr_coluna = int((np.size(mascara[0])-1)/2)
	imagem_extrapolada = extrapola(imagem, len(mascara), np.size(mascara[0]))
	imagem_saida = np.zeros((len(imagem), np.size(imagem[0])))
	for i in range(corr_linha, len(imagem_extrapolada)-corr_linha):
		for j in range(corr_coluna, np.size(imagem_extrapolada[0])-corr_coluna):
			pos_x = i - corr_linha
			pos_y = j - corr_coluna

			valor = 0.0
			for k in range(len(mascara)):
				for l in range(np.size(mascara[0])):
					if np.size(mascara[0]) == 1:
						valor += imagem_extrapolada[k+pos_x][l+pos_y]*mascara[k]
					else:
						valor += imagem_extrapolada[k+pos_x][l+pos_y]*mascara[k][l]
			imagem_saida[pos_x][pos_y] = int(valor/(len(mascara)*np.size(mascara[0])))
	return imagem_saida

def maskBlur():
	return np.array(((1,2,1),(2,4,2),(1,2,1)))*(1/16)

def blur(imagem):
	return convolve(imagem, maskBlur())

def seSquare3():
	return np.array(((1,1,1),(1,1,1),(1,1,1)))

def seCross3():
	return np.array(((0,1,0),(1,1,1),(0,1,0)))

def fft_print(local):
    img = imreadgray(local)
    f = np.fft.fft2(img, axes=(0,1))
    f = f/(img.shape[0]*img.shape[1])
    fshift = np.fft.fftshift(f)
    #magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.abs(f)
    pyp.subplot(121),pyp.imshow(img, cmap = 'gray')
    pyp.title('Input Image'), pyp.xticks([]), pyp.yticks([])
    pyp.subplot(122),pyp.imshow(magnitude_spectrum, cmap = 'gray')
    pyp.title('Magnitude Spectrum'), pyp.xticks([]), pyp.yticks([])
    pyp.show()

def fft_print_shift(local):
    img = imreadgray(local)
    f = np.fft.fft2(img, axes=(0,1))
    f = f/(img.shape[0]*img.shape[1])
    fshift = np.fft.fftshift(f)
    #magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.abs(fshift)
    pyp.subplot(121),pyp.imshow(img, cmap = 'gray')
    pyp.title('Input Image'), pyp.xticks([]), pyp.yticks([])
    pyp.subplot(122),pyp.imshow(magnitude_spectrum, cmap = 'gray')
    pyp.title('Magnitude Spectrum'), pyp.xticks([]), pyp.yticks([])
    pyp.show()

def fft(img):
	return np.fft.fft2(img)#*(1/(img.shape[0]*img.shape[1]))
#	return np.fft.fft2(img)/(img.shape[0]*img.shape[1])

def ifft(img):
	return np.fft.ifft2(img)

def ft_ift(local):
	img = imreadgray(local)
	ft = fft(img)
	ift = ifft(ft)
	print_img_ft_ift(img, ft, ift)

def print_img_ft_ift(img, ft, ift):
	magnitude_spectrum = np.abs(ft)
	pyp.subplot(131),pyp.imshow(img, cmap = 'gray')
	pyp.title('Input Image'), pyp.xticks([]), pyp.yticks([])
	pyp.subplot(132),pyp.imshow(magnitude_spectrum, cmap = 'gray')
	pyp.title('Magnitude Spectrum'), pyp.xticks([]), pyp.yticks([])
	pyp.subplot(133),pyp.imshow(np.abs(ift), cmap = 'gray')
	pyp.title('Inverse Fourrier'), pyp.xticks([]), pyp.yticks([])
	pyp.show()

def ruido_lena(img):
	ft = fft(img)
	pos = [16, 32, 48, 64, 80, 86, 112, 128, 144, 160, 176, 192, 208, 224, 240, 253, 254, 255]
	for i in pos:
		ft[i][i] = 0
#		if(m < np.abs(ft[i][i])):
#			pos = i
#	if (pos != 0):
#		ft[pos][pos] = 0
#	print(pos)
	ift = ifft(ft)
	pyp.subplot(121),pyp.imshow(img, cmap = 'gray')
	pyp.title('Input'), pyp.xticks([]), pyp.yticks([])
	pyp.subplot(122),pyp.imshow(np.abs(ift), cmap = 'gray')
	pyp.title('Output'), pyp.xticks([]), pyp.yticks([])
	pyp.show()

def my_ft(img):
	m = img.shape[0]
	n = img.shape[1]
	g = np.zeros((m,n))*1j
	for u in range(m):
		for v in range(n):
			for x in range(m):
				for y in range(n):
					#a = (np.cos(2*np.pi*((u*x)/m+(v*y)/n)) - 1j*np.sin(2*np.pi*((u*x)/m+(v*y)/n)))

					a = np.exp(-1j*2*np.pi*((u*x/m)+(v*y/n)))
					g[u][v] = g[u][v]+a*img[x][y]
	return g*(1/(m*n))

def my_ift(img):
	m = img.shape[0]
	n = img.shape[1]
	g = np.zeros((m,n))*1j
	for u in range(m):
		for v in range(n):
			for x in range(m):
				for y in range(n):
					a = (np.cos(2*np.pi*((u*x)/m+(v*y)/n)) + 1j*np.sin(2*np.pi*((u*x)/m+(v*y)/n)))
					g[u][v] = g[u][v]+a*img[x][y]
	return g

def erode(imagem, elemento):
	corr_linha = int((len(elemento)-1)/2)
	corr_coluna = int((np.size(elemento[0])-1)/2)
	imagem_extrapolada = extrapola(imagem, len(elemento), np.size(elemento[0]))
	imagem_saida = np.zeros((len(imagem), np.size(imagem[0])))
	for i in range(corr_linha, len(imagem_extrapolada)-corr_linha):
		for j in range(corr_coluna, np.size(imagem_extrapolada[0])-corr_coluna):
			pos_x = i - corr_linha
			pos_y = j - corr_coluna

			valor = []
			for k in range(len(elemento)):
				for l in range(np.size(elemento[0])):
					if(elemento[k][l] != 0):
						valor.append(imagem_extrapolada[k+pos_x][l+pos_y])
			if valor:
				imagem_saida[pos_x][pos_y] = min(valor)
	return imagem_saida

def rotacionaElem(elemento):
	newElement = np.zeros((elemento.shape[0], elemento.shape[1]))
	for i in range(len(elemento)):
		newElement[i] = elemento[i][::-1]
	return newElement[::-1]

def dilate(imagem, elemento):
	elemento = rotacionaElem(elemento)
	corr_linha = int((len(elemento)-1)/2)
	corr_coluna = int((np.size(elemento[0])-1)/2)
	imagem_extrapolada = extrapola(imagem, len(elemento), np.size(elemento[0]))
	imagem_saida = np.zeros((len(imagem), np.size(imagem[0])))
	for i in range(corr_linha, len(imagem_extrapolada)-corr_linha):
		for j in range(corr_coluna, np.size(imagem_extrapolada[0])-corr_coluna):
			pos_x = i - corr_linha
			pos_y = j - corr_coluna

			valor = []
			for k in range(len(elemento)):
				for l in range(np.size(elemento[0])):
					if(elemento[k][l] != 0):
						valor.append(imagem_extrapolada[k+pos_x][l+pos_y])
			if valor:
				imagem_saida[pos_x][pos_y] = max(valor)
	return imagem_saida