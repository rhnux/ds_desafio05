import os
from flask import Flask, request, render_template
from flask import send_from_directory
import pickle
import numpy as np
import re
from sklearn.externals import joblib
from lime.lime_text import LimeTextExplainer
static_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/getcat',methods=['POST','GET'])
def get_delay():
    if request.method=='GET':
        result=request.form

        cats = ('AUTOAYUDA','BIOGRAFIAS','CUENTOS','ENSAYOS','FICCION',
        'INFANTILES','NARRATIVA','NOVELA','OBRAS DIVERSAS')

        explainer = LimeTextExplainer(class_names=cats)

        #Prepare the feature vector for prediction
        new_vector = 'estoy cansado de mi personajes crea sebastian wainraich generan risa poridentificacion sabe observar nadie detalles cotidianosque pasan inadvertidos deberian ser extraordinarios despues todode trata humor hombre cuesta disfrutar sesometen situaciones denigrantes tal ganarse chica estees justo punto recorre cansado mi hombre piensandemasiado neuroticos amigueros antiheroes futboleros inseguros yellas mujeres cuentos dialogosinternos angustia muerte hijos analista suenoshabia sonado contrataban jugar carrera embolsadoscontra hitler si ganaba debia dejarme bigotes elresto vida si ganaba yo el propios medios quehacerse circuncision cuenta protagonista cuento datitulo volumenesta reedicion primer libro sebastian wainraich dondeencontramos ser feliz da verguenza mejor relatosmordaces inteligentes autenticos'

        new_vector_02 = 'tu rostro en el fuego ernesto salvadores arzuaga ferviente opositor gobierno juan manuel rosa unido tropas justo jose urquiza avanza hacia buenos aire procura nuevo ordenamiento territorio embargo destino juega mala pasada y arrojarse caballo enemigo queda atrapado estribo debe ser conducido hacia campamento pierna hecha jirones recuperarse visita tierras pergamino vive antiguo companero estudios honorio iriarte perteneciente familia estancieros netamente rosista alli ernesto conoce piedad hermana honorio duena belleza soberbia convencional medio guerras civiles desangran pais pasion secreta asfixiante une jovenes desafiados huir solo artero fanatismo honorio sino crueldad dona augusta madre joven espiritu oscuro propuesto destruirlos rostro fuego autora vuelve emocionar trama amorosa gran intensidad marco periodos cruciales tragicos historia argentinaen medio luchas unitarios federales pasion igualmente tormentosa une piedad ernesto pertenecientes familias enfrentadas pareja vera desafiada huir solo persecuciones desangran pais sino crueldad madre piedad espiritu oscuro desvela verlos padecer'

        new_vector_03 = 'EX MACHINA teóricamente sólo va a realizarse un test de Turing en Ava para ver si podría pasar por una persona, si su inteligencia artificial es realmente inteligente y es capaz de aprender, de reaccionar ante lo que se le presenta y de utilizar todas sus capacidades para conseguir lo que quiere. Es una película bastante intelectual hasta en la evolución de Ava, y nunca terminamos de saber hasta qué punto ha habido algún engaño en la historia, o si simplemente la historia no podía seguir otro camino que no fuera ése. El diseño de la robot es impresionante, y aunque surgen las cuestiones éticas sobre el trato que recibe, todo queda más en el plano cerebral, teórico, como si dijéramos.'

        new_vector_04 = 'neuromante Los hackers, en vez de ser tipos duros al asalto de fortalezas informáticas, se calan el gorro de Willy y se aposentan en los consejos de administración de las corporaciones, y si no es eso, descubren que ponerse un uniforme, o desgranar nimiedades ante una cámara y colgarlo en la red da bastante más dinero que enredar con unos sistemas que cada vez son más difíciles de atacar. Porque una de las cuestiones que obviaron los Gibson y compañía fue que no necesariamente los chicos más listos debían ser sus antihéroes. En las carreras armamentísticas siempre hay genios por ambos bandos. Esa idea de que el masterblaster del universo siempre está en el lado “oscuro” no se corresponde exactamente con la realidad. Hay muy poco de romántico en el trabajo del hacker, su principal actividad es darse de cabezazos contra el sistema atacado hasta que consigue colarse por un agujero sin tapar. Algo para lo que hace falta constancia, pero no una especial genialidad, que sin embargo si es necesaria para saber que hacer una vez abierta la puerta.'

        new_vector_05 = 'La primera es la autoconciencia. Hace referencia a nuestra capacidad para entender lo que sentimos y de estar siempre conectados a nuestros valores, a nuestra esencia. El segundo aspecto es la automotivación y nuestra habilidad por orientarnos hacia nuestras metas, de recuperarse de los contratiempos, de gestionar el estrés. La tercera dimensión tiene que ver con la conciencia social y con nuestra empatía. El cuarto eslabón es sin duda la piedra filosofal de la Inteligencia Emocional: nuestra habilidad para relacionarnos, para comunicar, para llegar acuerdos, para conectar positiva y respetuosamente con los demás.'

        new_vector_06 = 'El Abismo Censura. Tarifazo. Desempleo. Endeudamiento. Represión. Corrupción. Persecución judicial, linchamiento mediático... Desde que llegó al poder, el gobierno conservador de Mauricio Macri aumentó salvajemente la desigualdad entre ricos y pobres. Roberto Navarro escribe para quebrar el cerco periodístico que busca acallar las críticas y para denunciar las funestas consecuencias sociales, económicas y políticas de una gestión que favorece solo a una elite sin reglas que se maneja a base de amistades y pactos. Con valentía y claridad, el autor revela de modo descarnado y como nadie hasta ahora, lo que realmente piensan los empresarios, jueces y funcionarios que deciden el destino de todo un país. El Abismo es un documento fundamental para evaluar con serenidad la Argentina de los últimos años, entender la dimensión del cambio actual y evitar que el destino sea otra vez la dramática condena a la exclusión de millones de argentinos.'

        vector = re.sub(r'([^\s\w]|_)+', '', new_vector.lower())

        pkl_file = open('model_03.pkl', 'rb')
        loaded_model = joblib.load(pkl_file)
        prediction = loaded_model.predict([vector])

        exp = explainer.explain_instance(vector, loaded_model.predict_proba, num_features=16, top_labels=2)
        repo = exp.as_html()
        report = open('static/report.html','w')
        report.write(repo)
        report.close()

        def f(x):
            return {0:'AUTOAYUDA',1:'BIOGRAFIAS',2:'CUENTOS',3:'ENSAYOS',
            4:'FICCION',5:'INFANTILES',6:'NARRATIVA',7:'NOVELA',8:'OBRAS DIVERSAS'}[x]

        pred = f(prediction.item())

        return render_template('result.html',text=vector,prediction=pred)

@app.route('/exp', methods=['GET'])
def serve_dir_directory_index():
    return send_from_directory(static_file_dir, 'report.html')

if __name__ == '__main__':
    app.run()
