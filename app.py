
# Core Pkgs

import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("SR Ticket Classification Web App")
st.subheader("Creator : Debmalya Ray")
st.sidebar.title("Categorical Classification")


def main():

	""" Semi Automated ML App with Streamlit """

	activities = ["EDA","Plots","Machine_Learning"]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Show Shape"):
				st.write(df.shape)

			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Summary"):
				st.write(df.describe())

			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())

			if st.checkbox("Correlation Plot(Matplotlib)"):
				plt.matshow(df.corr())
				st.pyplot()

			if st.checkbox("Correlation Plot(Seaborn)"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()





	elif choice == 'Plots':
		st.subheader("Data Visualization")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())


			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
				st.pyplot()
		
			# Customizable Plot

			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","kde"])
			selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			if st.button("Generate Plot"):
				st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
				if type_of_plot == 'area':
					cust_data = df[selected_columns_names]
					st.area_chart(cust_data)

				elif type_of_plot == 'bar':
					cust_data = df[selected_columns_names]
					st.bar_chart(cust_data)

				elif type_of_plot == 'line':
					cust_data = df[selected_columns_names]
					st.line_chart(cust_data)

				# Custom Plot 
				elif type_of_plot:
					cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					st.write(cust_plot)
					st.pyplot()



	elif choice == 'Machine_Learning':

		st.subheader("Machine Learning (ML)")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		if data is not None:
			df = pd.read_csv(data)
			label_encoder = preprocessing.LabelEncoder()
			df['RequestorSeniority'] = label_encoder.fit_transform(df['RequestorSeniority'])
			df['FiledAgainst'] = label_encoder.fit_transform(df['FiledAgainst'])
			df['TicketType'] = label_encoder.fit_transform(df['TicketType'])
			df['Severity'] = label_encoder.fit_transform(df['Severity'])
			df['Priority'] = label_encoder.fit_transform(df['Priority'])
			df['Satisfaction'] = label_encoder.fit_transform(df['Satisfaction'])

			X = (df.drop(columns=df[['Priority']],axis=0)).values
			Y = (df.iloc[:,-1:]).values
		# =======================================================================
		# SCALING :
		# ========================================================================
			scaler = StandardScaler()
			X=scaler.fit_transform(X)
			X  = pd.DataFrame(X)
			X.rename(columns={0:"ticket", 1:"requestor", 2:"RequestorSeniority", 3:"ITOwner", 4:"FiledAgainst", 5:"TicketType", 6:"Severity", 7:"daysOpen", 8:"Satisfaction"}, inplace=True)
		#	Y = scaler.fit_transform(Y)
		#	Y = pd.DataFrame(Y)
		#	Y.rename(columns={0:"Priority"}, inplace=True)
		# ==============================================================================
		#  FEATURE   SELECTION  :
		# ==============================================================================

			from sklearn.ensemble import ExtraTreesClassifier
			Ext = ExtraTreesClassifier()
			Ext.fit(X,Y)
			print(Ext.feature_importances_)


			X.drop(columns="TicketType", inplace=True)
			X.drop(columns="FiledAgainst", inplace=True)

			X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
			#st.sidebar.subheader("Model Hyper-parameters")
			#n_estimators = st.sidebar.number_input("n_estimators", 100, 200, step=10, key='n_estimators')
			#max_depth = st.sidebar.number_input("The maximum depth of the tree", 5, 20, step=1, key='max_depth')
			#bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
			#min_samples_leaf  = st.sidebar.number_input("The minimum sample leaf", 50, 95, step=5, key='min_samples_leaf')
			#min_samples_split  = st.sidebar.number_input("The minimum sample split", 500, 900, step=10, key='min_samples_split')

			st.sidebar.subheader("Choose Classifier")
			classifier = st.sidebar.selectbox("Classifier", ("Random Forest", "Gradient Boosting", "Ada Boosting"))
		#	if st.sidebar.button("Classify", key='classify'):

			if classifier == 'Random Forest':
				st.sidebar.subheader("Model Hyper-parameters")
				max_depth = st.sidebar.number_input("The maximum depth of the tree", 5, 20, step=1, key='max_depth')
				bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

				st.sidebar.button("Classify", key='classify')
				st.subheader("Random Forest Results")
				model = RandomForestClassifier(max_depth=max_depth, bootstrap=bootstrap)
				model.fit(X_train,Y_train)
				accuracy = model.score(X_test, Y_test)
				y_pred = model.predict(X_test)
				y_result = pd.DataFrame(y_pred)
				y_result.rename(columns={0:"Predict"}, inplace=True)
				d =  {0: '0 - Unassigned', 1: '1 - Low', 2: '2 - Medium', 3: '3 - High'}
				y_result = y_result["Predict"].map(d)
				accuracy = model.score(X_test, Y_test)
				plt.figure(figsize=(10,6), dpi=80, facecolor='blue')
				bw = y_result.value_counts()

				explode=(0.1, 0.1, 0.1, 0.3)
				st.write("Test Accuracy : ", accuracy.round(2))
				precision = round(precision_score(y_pred, Y_test, average='micro'),2)
				st.write("Precision : ", precision)
				recall = round(recall_score(y_pred, Y_test, average='micro'),2)
				st.write("Recall : ", recall)
				macro_averaged_f1 = f1_score(Y_test, y_pred, average = 'micro')
				st.write("F1 Score:", macro_averaged_f1)
				st.write("Confusion Matrix : ", confusion_matrix(Y_test, y_pred))
				st.write("Test Results Shape : ", X_test.shape)
				st.write("Prediction Results : ", y_result.value_counts())
				st.write("Bar Chart Representation :")
				st.bar_chart(bw)

			if classifier == 'Gradient Boosting':

				st.sidebar.subheader("Model Hyper-parameters")
				max_depth = st.sidebar.number_input("The maximum depth of the tree", 5, 20, step=1, key='max_depth')
				min_samples_leaf  = st.sidebar.number_input("The minimum sample leaf", 50, 95, step=5, key='min_samples_leaf')
				min_samples_split  = st.sidebar.number_input("The minimum sample split", 500, 900, step=10, key='min_samples_split')

				st.sidebar.button("Classify", key='classify')
				st.subheader("Gradient Boosting Results")
				model = GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
				model.fit(X_train,Y_train)
				accuracy = model.score(X_test, Y_test)
				y_pred = model.predict(X_test)
				y_result = pd.DataFrame(y_pred)
				y_result.rename(columns={0:"Predict"}, inplace=True)
				d =  {0: '0 - Unassigned', 1: '1 - Low', 2: '2 - Medium', 3: '3 - High'}
				y_result = y_result["Predict"].map(d)
				accuracy = model.score(X_test, Y_test)

				plt.figure(figsize=(10,6), dpi=80, facecolor='blue')
				bw = y_result.value_counts()
				explode=(0.1, 0.1, 0.1, 0.3)
				st.write("Test Accuracy : ", accuracy.round(2))
				precision = round(precision_score(y_pred, Y_test, average='micro'),2)
				st.write("Precision : ", precision)
				recall = round(recall_score(y_pred, Y_test, average='micro'),2)
				st.write("Recall : ", recall)
				macro_averaged_f1 = f1_score(Y_test, y_pred, average = 'micro')
				st.write("F1 Score:", macro_averaged_f1)
				st.write("Confusion Matrix : ", confusion_matrix(Y_test, y_pred))
				st.write("Test Results Shape : ", X_test.shape)
				st.write("Prediction Results : ", y_result.value_counts())
				st.write("Bar Chart Representation :")
				st.bar_chart(bw)

			if classifier == 'Ada Boosting':

				st.sidebar.subheader("Model Hyper-parameters")
				n_estimators = st.sidebar.number_input("n_estimators", 100, 200, step=10, key='n_estimators')
				st.subheader("Ada Boosting Results")
				model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=1, random_state=0)
				model.fit(X_train,Y_train)
				accuracy = model.score(X_test, Y_test)
				y_pred = model.predict(X_test)
				y_result = pd.DataFrame(y_pred)
				y_result.rename(columns={0:"Predict"}, inplace=True)
				d =  {0: '0 - Unassigned', 1: '1 - Low', 2: '2 - Medium', 3: '3 - High'}
				y_result = y_result["Predict"].map(d)
				accuracy = model.score(X_test, Y_test)

				plt.figure(figsize=(10,6), dpi=80, facecolor='blue')

				st.sidebar.button("Classify", key='classify')
				bw = y_result.value_counts()

				st.write("Test Accuracy : ", accuracy.round(2))
				precision = round(precision_score(y_pred, Y_test, average='micro'),2)
				st.write("Precision : ", precision)
				recall = round(recall_score(y_pred, Y_test, average='micro'),2)
				st.write("Recall : ", recall)
				macro_averaged_f1 = f1_score(Y_test, y_pred, average = 'micro')
				st.write("F1 Score:", macro_averaged_f1)
				st.write("Confusion Matrix : ", confusion_matrix(Y_test, y_pred))
				st.write("Test Results Shape : ", X_test.shape)
				st.write("Prediction Results : ", y_result.value_counts())
				st.write("Bar Chart Representation :")
				st.bar_chart(bw)


if __name__ == '__main__':
	main()
