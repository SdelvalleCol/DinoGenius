import numpy as np
np.random.seed(5)

from keras.layers import Input, Dense, SimpleRNN
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as K
 
# 1. Lectura a la base de datos (txt) --> Nombres Dinosaurios
nombre = open('names.txt','r').read()
nombre = nombre.lower()
alfabeto = sorted(list(set(nombre)))
diccionario_principal ={}
diccionario_secundario = {}
tamano = len(alfabeto)
for i in range(0,tamano):
    diccionario_secundario.update({i:alfabeto[i]})
    diccionario_principal.update({alfabeto[i]:i})
print(diccionario_principal)
tamano = len(diccionario_principal)

# 2. Implementación en Keras

neuronas = 25
entrada = Input(shape=(None,tamano))
a0 = Input(shape=(neuronas,))
celda = SimpleRNN(neuronas,activation='tanh',return_state=True)
capa_salida = Dense(len(alfabeto),activation='softmax')
hs,_ = celda(entrada,initial_state=a0)
salida = []
salida.append(capa_salida(hs))
modelo = Model([entrada,a0],salida)
opt = SGD(lr=0.0005)
modelo.compile(optimizer=opt,loss='categorical_crossentropy')

# 3. ejemplos de entrenamiento

with open("names.txt") as f:
    ejemplos = f.readlines()
ejemplos = [x.lower().strip() for x in ejemplos]
np.random.shuffle(ejemplos)

def train_generator():
    while True:
        ejemplo = ejemplos[np.random.randint(0,len(ejemplos))]
        X = [None] + [diccionario_principal[c] for c in ejemplo]
        Y = X[1:] + [diccionario_principal['\n']]
        x = np.zeros((len(X),1,tamano))
        onehot = to_categorical(X[1:],tamano).reshape(len(X)-1,1,tamano)
        x[1:,:,:] = onehot
        y = to_categorical(Y,tamano).reshape(len(X),tamano)
        a = np.zeros((len(X), neuronas))

        yield [x, a], y

# 4. ENTRENAMIENTO

BATCH_SIZE = 500			# Número de ejemplos de entrenamiento a usar en cada iteración
NITS = 5000			# Número de iteraciones

for j in range(NITS):
    historia = modelo.fit_generator(train_generator(), steps_per_epoch=BATCH_SIZE, epochs=1, verbose=0)
    if j%100 == 0:
        print('\nIteración: %d, Error: %f' % (j, historia.history['loss'][0]) + '\n')


# 5. GENERACIÓN DE NOMBRES USANDO EL MODELO ENTRENADO

def generar_nombre(modelo,car_a_num,tamano,n_a):
    x = np.zeros((1,1,tamano,))
    a = np.zeros((1, n_a))
    nombre_generado = ''
    fin_linea = '\n'
    car = -1
    contador = 0
    while (car != fin_linea and contador != 50):
          a, _ = celda(K.constant(x), initial_state=K.constant(a))
          y = capa_salida(a)
          prediccion = K.eval(y)
          ix = np.random.choice(list(range(tamano)),p=prediccion.ravel())
          car = diccionario_secundario[ix]
          nombre_generado += car
          x = to_categorical(ix,tamano).reshape(1,1,tamano)
          a = K.eval(a)
          contador += 1
          if (contador == 50):
            nombre_generado += '\n'
    print(nombre_generado)

# Generar 100 ejemplos de nombres generados por el modelo ya entrenado
for i in range(100):
    generar_nombre(modelo,diccionario_principal,tamano,neuronas)