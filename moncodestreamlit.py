import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import shap
import lightgbm as lgb
shap.initjs()
from plotly import tools
import plotly.offline as py
import os,joblib,warnings
warnings.filterwarnings('ignore')
import seaborn as sns
color=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
###########################################
st.set_page_config(layout="wide")
st.title("Tableaux de bord pour prédire un défaut de remboursement de crédit")
st.subheader("Ce tableau de bord permet de prédire si un client est capable ou non capable de rembourser un crédit")
#########################################
X_test_final=pd.read_csv("X_test_final.csv")
X_test_final.set_index("SK_ID_CURR", inplace = True)
y_test = pd.read_pickle("y_test")
#####################################
clf1=joblib.load('my_model.joblib')
############################################
selected_features=joblib.load('my_feature.joblib')
##################################################
predict_test=clf1.predict(X_test_final[selected_features])
predict_prob_test=clf1.predict_proba(X_test_final[selected_features])
df=X_test_final[selected_features].copy()
df['predict']=predict_test
df['proba']=predict_prob_test[:,1]
#################
if st.checkbox('Montrez la table'):
	st.subheader('Voici les données')
	st.dataframe(df)

def st_shap(plot, height=200,width=870):
	shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
	components.html(shap_html, height=height,width=width)


#@st.cache()
df1 = X_test_final.copy()
#df1.set_index('SK_ID_CURR',inplace=True)
df1['AGE']=round(np.abs(df1['DAYS_BIRTH']/365)).astype(int)

st.sidebar.title('PROFIL CLIENT')
id_client=st.sidebar.selectbox("Selectionnez l'identifiant du client", options=(df.index))
if id_client in list(df.index):
	a=(int(id_client))

	#st.sidebar.subheader("Profil CLIENT")
	st.sidebar.markdown(f'**Sexe:**<div style="color: green; font-size: medium">{df1["CODE_GENDER"].loc[a]}</div>',unsafe_allow_html=True)
	st.sidebar.markdown(f'**Profession:**<div style="color: green; font-size: medium">{df1["OCCUPATION_TYPE"].loc[a]}</div>',unsafe_allow_html=True)
	st.sidebar.markdown(f'**Source du Revenu:**<div style="color: green; font-size: medium">{df1["NAME_INCOME_TYPE"].loc[a]}</div>',unsafe_allow_html=True)
	st.sidebar.markdown(f'**Situation Familiale:**<div style="color: green; font-size: medium">{df1["NAME_FAMILY_STATUS"].loc[a]}</div>',unsafe_allow_html=True)
	st.sidebar.markdown(f'**Niveau d\'Etude:**<div style="color: green; font-size: medium">{df1["NAME_EDUCATION_TYPE"].loc[a]}</div>',unsafe_allow_html=True)
	st.sidebar.markdown(f'**Age:**<div style="color: green; font-size: medium">{df1["AGE"].loc[a]}</div>',unsafe_allow_html=True)


	b=np.round(df['proba'].loc[a],3)

	col1, col2= st.beta_columns(2)

	with col1:
		st.markdown("**IDENTIFIANT**")
		st.write("L'identifiant du client est:",a, use_column_width=True)
	with col2:
		st.markdown("**PROBABILITE**")
		st.write("Probabilité de ne pas rembourser est:",b,use_column_width=True)


	links3="https://github.com/SidiML/maria/blob/master/my_shap_model.joblib?raw=true"
	mfile3 = BytesIO(requests.get(links3).content)
	@st.cache()
	def get_data():

		return joblib.load(mfile3)
	lgbm_explainer=get_data()
	shap_values =lgbm_explainer.shap_values(df.drop(['predict','proba'],1).loc[[a]])

	vals= np.abs(shap_values).mean(0)
	feature_importance = pd.DataFrame(list(zip(df.drop(['predict','proba'],1).loc[[a]], vals)), columns=['col_name','feature_importance'])
	feature_importance.sort_values(by=['feature_importance'], ascending=False,inplace=True)


	st.sidebar.title("Features Importantes")
	if st.sidebar.checkbox("Voir les features"):
		st.sidebar.subheader('Les variables importantes')
		st.sidebar.dataframe(feature_importance)
	st.sidebar.subheader('Graphe de Décision')
	fig2, ax = plt.subplots(nrows=1, ncols=1)
	shap.summary_plot(shap_values, df.drop(['predict','proba'],1).loc[[a]], plot_type="bar")
	st.sidebar.pyplot(fig2,use_column_width=True)
	# plot the SHAP values for the 10th observation
	st_shap(shap.force_plot(lgbm_explainer.expected_value, shap_values, df.drop(['predict','proba'],1).loc[[a]]))
	col1, col2= st.beta_columns(2)
	with col1:
		st.subheader("Comparaison du Client avec la moyenne des Clients")
		cols=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','AGE','CREDIT_TERM','AMT_ANNUITY','DAYS_BIRTH','DAYS_EMPLOYED']
		colonnes=st.selectbox("Selectionnez une Variable", options=(cols))
		average_hold=df1.groupby(['TARGET']).mean()
		b1=round(df1.loc[a][colonnes],2)
		b2=round(average_hold[colonnes].values[0],2)
		b3=round(average_hold[colonnes].values[1],2)
		# intialise data of lists.
		x1 = ['CLIENT', 'AVERAGE_CLIENT_REPAID', 'AVERAGE_CLIENT_NO_REPAID']
		y1 = [b1,b2 ,b3]
		# Use textposition='auto' for direct text
		colors = ['#17becf',] * 3
		colors[0] = 'crimson'
		#fig = go.Figure(data=[go.Bar(x=x1, y=y1,text=y1,textposition='auto',marker_color=colors)])
		data = [go.Bar(x=x1, y=y1,text=y1,textposition='auto',marker_color=colors)]

		layout = go.Layout(
			title = "Comparaison de la variable {}".format(colonnes),
			xaxis=dict(
				title='{} du client contre la moyenne'.format(colonnes),
				),
			yaxis=dict(
				title=colonnes,
				)
		)
		fig1 = go.Figure(data = data, layout=layout)
		fig1.layout.template = "simple_white"
		#py.iplot(fig1)
		st.plotly_chart(fig1,use_container_width=True)

	with col2:
		st.subheader("Comparaison du client avec les clients ayant les mêmes profils.")
		cols1=["CODE_GENDER","OCCUPATION_TYPE","NAME_INCOME_TYPE","NAME_FAMILY_STATUS","NAME_EDUCATION_TYPE","NAME_TYPE_SUITE","NAME_HOUSING_TYPE"]
		colonnes1=st.selectbox("Selectionnez une Information", options=(cols1))
		age=round(df1.loc[a]['AGE'],2)
		df3=df1[df1['AGE']==age]
		#st.write(df3.head())

		temp = df3[colonnes1].value_counts()

		temp_val_y0 = []
		temp_val_y1 = []
		for val in temp.index:
			temp_val_y1.append(np.sum(df3['TARGET'][df3[colonnes1]==val] == 1))
			temp_val_y0.append(np.sum(df3['TARGET'][df3[colonnes1]==val] == 0))

		x2= temp.index
		y2=((temp_val_y1 / temp.sum()) * 100)
		y3=((temp_val_y0 / temp.sum()) * 100)
		data1 = [go.Bar(x = x2, y =y2 , name='Yes'),
				go.Bar(x = x2, y =y3 , name='No')]
		layout = go.Layout(
			title = "Comparaison du caractéristique {} en terme de remboursement de credit ou non en %".format(colonnes1),
			xaxis=dict(
				title='{} des demandeurs'.format(colonnes1),
				),
			yaxis=dict(
				title=colonnes1,
				)
		)
		fig2 = go.Figure(data = data1, layout=layout)
		fig2.layout.template = "plotly"
		#py.iplot(fig1)
		st.plotly_chart(fig2,use_container_width=True)

else:
	st.write("Cet identifiant n'est pas correcte")

#############################################
#age moyen des bons et mauvais emprunteurs
