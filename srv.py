import numpy as np
import os
from flask import Flask, request, render_template, make_response
import joblib

app = Flask(__name__, static_url_path='/static')
model = joblib.load('model/model.pkl')


@app.route('/')
def display_gui():
    return render_template('template.html')

@app.route('/verificar', methods=['POST'])
def verificar():
	CriterioConfirmacao = request.form['gridRadiosCriterioConfirmacao']
	FaixaEtaria = request.form['gridRadiosFaixaEtaria']
	Sexo = request.form['gridRadiosSexo']
	RacaCor = request.form['gridRadiosRacaCor']
	Febre = request.form['gridRadiosFebre']
	DificuldadeRespiratoria = request.form['gridRadiosDificuldadeRespiratoria']
	Tosse = request.form['gridRadiosTosse']
	Coriza = request.form['gridRadiosCoriza']
	DorGarganta = request.form['gridRadiosDorGarganta']
	Diarreia = request.form['gridRadiosDiarreia']
	Cefaleia = request.form['gridRadiosCefaleia']
	ComorbidadePulmao = request.form['gridRadiosComorbidadePulmao']
	ComorbidadeCardio = request.form['gridRadiosComorbidadeCardio']
	ComorbidadeRenal = request.form['gridRadiosComorbidadeRenal']
	ComorbidadeDiabetes = request.form['gridRadiosComorbidadeDiabetes']
	ComorbidadeTabagismo = request.form['gridRadiosComorbidadeTabagismo']
	ComorbidadeObesidade = request.form['gridRadiosComorbidadeObesidade']
	FicouInternado = request.form['gridRadiosFicouInternado']
	
	dados = np.array([[CriterioConfirmacao,FaixaEtaria,Sexo,RacaCor,Febre,
										DificuldadeRespiratoria,Tosse,Coriza,DorGarganta,Diarreia,
										Cefaleia,ComorbidadePulmao,ComorbidadeCardio,ComorbidadeRenal,
										ComorbidadeDiabetes,ComorbidadeTabagismo,ComorbidadeObesidade,
										FicouInternado]])

	print(":::::: Dados de Teste ::::::")
	print("CriterioConfirmacao: {}".format(CriterioConfirmacao))
	print("FaixaEtaria: {}".format(FaixaEtaria))
	print("Sexo: {}".format(Sexo))
	print("RacaCor: {}".format(RacaCor))
	print("Febre: {}".format(Febre))
	print("DificuldadeRespiratoria: {}".format(DificuldadeRespiratoria))
	print("Tosse: {}".format(Tosse))
	print("Coriza: {}".format(Coriza))
	print("DorGarganta: {}".format(DorGarganta))
	print("Diarreia: {}".format(Diarreia))
	print("Cefaleia: {}".format(Cefaleia))
	print("ComorbidadePulmao: {}".format(ComorbidadePulmao))
	print("ComorbidadeCardio: {}".format(ComorbidadeCardio))
	print("ComorbidadeRenal: {}".format(ComorbidadeRenal))
	print("ComorbidadeDiabetes: {}".format(ComorbidadeDiabetes))
	print("ComorbidadeTabagismo: {}".format(ComorbidadeTabagismo))
	print("ComorbidadeObesidade: {}".format(ComorbidadeObesidade))
	print("FicouInternado: {}".format(FicouInternado))
	print("\n")

	classe = model.predict(dados)[0]
	print("Classe Predita: {}".format(str(classe)))

	classe0 = model.predict_proba(dados)

	print("Probabilidade: {}".format(str(classe0)))

	if classe == 0:
		classe1 = classe0[0,0]
	else:
		classe1 = classe0[0,1]

	classe1 = classe1*100
	classe1 = round(classe1, 4)

	return render_template('template.html',classe=str(classe),classe1=str(classe1))

if __name__ == "__main__":
        port = int(os.environ.get('PORT', 5500))
        app.run(host='0.0.0.0', port=port)


