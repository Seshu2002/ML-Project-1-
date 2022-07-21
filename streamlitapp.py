import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import keras
import pickle

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)


    st.title("Classification of Extinguishable Status of Flame")
    inputs_t="""<h3 style="font-family: 'Fascinate', cursive;font-size:20px">Inputs</h3>"""
    st.sidebar.markdown(inputs_t,unsafe_allow_html=True)
    data = [["Artificial Neural Network", 96.03, 94.44], ["k-Nearest Neighbor", 92.62, 96.07], ["Random Forest", 96.58, 96.61],["Deep Neural Network",94.88,96.34],["Stacking Model",97.06,97.39]]
    Accdata=pd.DataFrame(data, columns=["Algorithm", "Base Paper Accuracy", "Implementation Accuracy"])
    st.table(Accdata)

    def NoDataError():
        original_title = '<p style="font-family:Roboto Mono; color:Red; font-size: 20px;">Please upload a dataset with valid contents</p>'
        st.markdown(original_title, unsafe_allow_html=True)

    def load_data():
        # pd.read_csv('./mushrooms.csv')
        data = st.file_uploader("Upload Dataset", type=["csv"])

        if data is not None:
            df = pd.read_csv(data)
            label = LabelEncoder()
            for col in df.columns:
                df[col] = label.fit_transform(df[col].astype(str))
        else:
            NoDataError()
            return None
        return df

    @st.cache(persist=True)
    def processed_data(type,dist,db,airflow,freq,size):
        size=(size-1)/(7-1)
        dist=(dist-10)/(190-10)
        db=(db-72)/(113-72)
        airflow=(airflow-0)/(17-0)
        freq=(freq-1)/(75-1)
        input_l=[]
        input_l.append(size)
        input_l.append(dist)
        input_l.append(db)
        input_l.append(airflow)
        input_l.append(freq)
        for i in range(4):
            input_l.append(0)
        if(type=="Gasoline"):
            input_l[-4]=1
        elif(type=="Thinner"):
            input_l[-1]=1
        elif(type=="LPG"):
            input_l[-2]=1
        else:
            input_l[-3]=1
        return input_l


    @st.cache(persist=True)
    def split(df, label_column):
        y = df[label_column]
        x = df.drop(columns=[label_column])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list,x_test,y_test):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Cuve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()


    df = load_data()
    if df is not None:
        pickle_in = open('data.pickle', 'rb')
        temp = pickle.load(pickle_in)  # temp data to store th input array
        x_train, x_test, y_train, y_test = split(df, "STATUS")
        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox(
            "Classifier", ("K-Nearest Neighbours","Random Forest","Artificial Neural Network","Deep Neural Network","Stacking Model"))
        st.sidebar.subheader("INPUTS")
        size= st.sidebar.slider(
            "Size", 1, 7, key='size')
        fuel=st.sidebar.selectbox(
            "Fuel Type", ("Gasoline","LPG","Thinner","Kerosine"))
        dist = st.sidebar.slider(
            "Distance", 10, 190, key='dist')
        db = st.sidebar.slider(
            "Desibel", 72, 113, key='db')
        airflow= st.sidebar.slider(
            "Airflow", 0, 17, key='airflow')
        freq=st.sidebar.slider(
            "Frequency",1,75,key='freq')
        metrics_list = st.sidebar.multiselect("select graphs",
                                        ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Predict", key='classifier'):
            if (classifier == "K-Nearest Neighbours"):
                with open('Knn.pickle', 'rb') as file:
                    model = pickle.load(file)
                st.write("Accuracy : {}".format(model.score(temp[0],temp[1])))
                ans=model.predict([processed_data(type,dist,db,airflow,freq,size)]).ravel()
                if ans[0]:
                    anim = '<p style="font-family:; color:Green; font-size: 15px;">[ 🔥 Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                else:
                    anim = '<p style="font-family:Courier; color:Red; font-size: 15px;">[ 🔥 Not Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                plot_metrics(metrics_list,temp[0],temp[1])

            if (classifier == "Stacking Model"):
                with open('stk_model.pickle', 'rb') as file:
                    model = pickle.load(file)
                st.write("ACCURACY : {}".format(model.score(temp[0], temp[1])))
                ans = model.predict([processed_data(type, dist, db, airflow, freq, size)]).ravel()
                if ans[0]:
                    anim = '<p style="font-family:; color:Green; font-size: 15px;">[ 🔥 Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                else:
                    anim = '<p style="font-family:Courier; color:Red; font-size: 15px;">[ 🔥 Not Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                plot_metrics(metrics_list, temp[0], temp[1])
            if (classifier == "Artificial Neural Network"):
                model=keras.models.load_model("Ann_model")
                ans = model.predict([processed_data(type, dist, db, airflow, freq, size)]).ravel()
                st.write("Accuracy : {}".format(model.evaluate(temp[0], temp[1])[1]))
                if ans[0]:
                    anim = '<p style="font-family:; color:Green; font-size: 15px;">[ 🔥 Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                else:
                    anim = '<p style="font-family:Courier; color:Red; font-size: 15px;">[ 🔥 Not Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                y_pred=(model.predict(temp[0])>0.5).astype(int).ravel()
                temp2=np.array(temp[1])
                tp=0
                tn=0
                fn=0
                fp=0
                for i in range(len(y_pred)):
                    if(y_pred[i]==1):
                        if temp2[i]==1:
                            tp=tp+1
                        else:
                            fp=fp+1
                    else:
                        if temp2[i]==0:
                            tn=tn+1
                        else:
                            fn=fn+1

                st.write("True positive :{}".format(tp))
                st.write("False positive :{}".format(fp))
                st.write("True Negative :{}".format(tn))
                st.write("False Negative :{}".format(fn))
            if (classifier == "Random Forest"):
                with open('Rf.pickle', 'rb') as file:
                    model = pickle.load(file)
                st.write("Accuracy : {}".format(model.score(temp[0],temp[1])))
                ans=model.predict([processed_data(type,dist,db,airflow,freq,size)]).ravel()
                if ans[0]:
                    anim = '<p style="font-family:; color:Green; font-size: 15px;">[ 🔥 Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                else:
                    anim = '<p style="font-family:Courier; color:Red; font-size: 15px;">[ 🔥 Not Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                plot_metrics(metrics_list,temp[0],temp[1])
            if (classifier == "Deep Neural Network"):
                model = keras.models.load_model("Dnn_model")
                ans = (model.predict([processed_data(type, dist, db, airflow, freq, size)])>0.5).astype(int).ravel()
                st.write("Accuracy : {}".format(model.evaluate(temp[0], temp[1])[1]))
                if ans[0]:
                    anim = '<p style="font-family:; color:Green; font-size: 15px;">[ 🔥 Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                else:
                    anim = '<p style="font-family:Courier; color:Red; font-size: 15px;">[ 🔥 Not Extinguishable ]</p>'
                    st.markdown(anim, unsafe_allow_html=True)
                y_pred = (model.predict(temp[0]) > 0.5).astype(int).ravel()
                temp2 = np.array(temp[1])
                tp = 0
                tn = 0
                fn = 0
                fp = 0
                for i in range(len(y_pred)):
                    if (y_pred[i] == 1):
                        if temp2[i] == 1:
                            tp = tp + 1
                        else:
                            fp = fp + 1
                    else:
                        if temp2[i] == 0:
                            tn = tn + 1
                        else:
                            fn = fn + 1

                st.write("True positive :{}".format(tp))
                st.write("False positive :{}".format(fp))
                st.write("True Negative :{}".format(tn))
                st.write("False Negative :{}".format(fn))




    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Dataset")
        st.write(df)

if __name__ == '__main__':
    main()