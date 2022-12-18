import streamlit as st
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report, recall_score,confusion_matrix, roc_auc_score, f1_score, auc, plot_confusion_matrix,plot_roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import shap
import streamlit.components.v1 as components

selected_features = ['customerID','SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
selected_features_nocust = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
# dfe
features_needed = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache #(persist=True)
def load_data():
    data = pd.read_csv(".\processed.csv")
    return data

@st.cache (allow_output_mutation=True)#(persist=True)
def load_encoder():
    encoder = joblib.load('encode.joblib')
    return encoder

@st.cache (allow_output_mutation=True)
def load_model():
    model = joblib.load('model.joblib')
    return model

df = load_data()
encoder = load_encoder()
model = load_model()
df_back = df.copy()
def no_selected_data():
    st.error('Please select data first!', icon="🚨")

def whole_processor(data: pd.DataFrame):
    encoded_data = encoder.transform(data[selected_features_nocust])
    target = encoded_data['Churn']
    features = encoded_data.drop('Churn', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=0.25, random_state=463, stratify=target)
    return [x_test, y_test]

def input_update(input_candidate):
    st.session_state.input_data = input_candidate #.reset_index(drop=True, inplace=True)

def get_input() -> pd.DataFrame:
    dataframe = st.session_state.input_data
    return dataframe

# customer ID & filter select
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    modify = st.checkbox("Add filters")

    if not modify:
        st.dataframe(df)

    else:
        df = df.copy()
        table, modification_container = st.tabs(["Table", "Filter"])
        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df[features_needed].columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_object_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                    
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]

                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]
        with table:
            st.dataframe(df)
# input page
def page_input_select():
    st.title("Select Data via Customer ID & Manual Entry (What if Analysis)")
    st.markdown("**Selected Customer Info**")
    input_data = get_input()
    st.write(input_data)
    

    input_select = st.radio(
        "How would you select Input Data",
        ('By Customer ID', 'Manual Fill - What If Analysis?'))

    if input_select == 'By Customer ID':
            sel_custID = st.text_input('Enter Customer ID', '0000-XXXXX')
            submit_id = st.button('Submit')
            
            filter_dataframe(df)
            if submit_id:
                input_data = df[selected_features].loc[df['customerID'] == sel_custID]
                input_update(input_data)
                submit_id = False

    if input_select == 'Manual Fill - What If Analysis?':
        #input_data = get_input()
        #st.write(selected_features)
        for feat in selected_features:
            if feat == 'customerID':
                input_data[feat] = 'Manual'
                input_data[feat] = input_data[feat].astype(object)
            if feat == 'Churn':
                input_data[feat] = 'No'
                input_data[feat]= input_data[feat].astype(object)
            elif (df[feat].dtype == 'O') or (len(df[feat].unique()) < 10):
                input_data[feat] = st.selectbox(label=feat, options=df[feat].unique())
            else:
                input_data[feat] = st.number_input(label=feat, min_value = 0) #, value=input_data[feat])
        submit_man = st.button(label="Submit")
        if submit_man:
            input_update(input_data)
            submit_man = False

def result_func():
    input_data = get_input().copy()
    cust_id = input_data['customerID']
    input_data = input_data.drop('customerID', axis=1)
  
    if cust_id.item() == 'Initial':
        no_selected_data()
    elif cust_id.item() == 'Manual':
        st.info('Data Selected By Hand', icon="ℹ️")
    else:
        st.info(f"Selected Customer's ID:{cust_id.item()}", icon="ℹ️")
    with st.expander("Selected Data"):
        st.markdown("**Selected Data**")
        st.write(input_data.drop('Churn', axis=1))
    
    # encoding the labels
    encoded_data = encoder.transform(input_data)
    encoded_data = encoded_data.drop('Churn', axis=1)
    with st.expander("Encoded Data"):
        st.markdown("**Encoded Data**")
        st.write(encoded_data)
    # threshold
    p_threshold = 0.5
    #c_threshold = st.checkbox("Select Custom Threshold(Makes no difference to SHAP)")
    #if c_threshold :
        #p_threshold = st.slider("Select Threshold", min_value=0.0, max_value=1.0,
                              #value=0.5,step=0.05)
    # predictions
    model.set_probability_threshold(p_threshold)
    pred = model.predict(encoded_data)
    pred_proba = model.predict_proba(encoded_data)


    st.write(f"Predicted Probability of Churn: {round(pred_proba[0][1], 3)}")
    st.write(f"Prediction of Churn: {pred[0]}")
    
    tab1, tab2 = st.tabs(["Decision plot", "Waterfall chart"])
    explainer = shap.TreeExplainer(model = model)
    input_data_shap_values = explainer(encoded_data)
        #decision plot
    with tab1:
        st.markdown("**Explaning the Prediction with decision plot**")
        shap.decision_plot(explainer.expected_value, input_data_shap_values.values[0], encoded_data, link= "logit")
        st.pyplot()
    # waterfall chart
    with tab2:
        st.markdown("**Explaning the Prediction with waterfall chart**")
        shap.plots.waterfall(input_data_shap_values[0], max_display=15)
        st.pyplot()    

def metric_analysis():
    # select the evaluation metrics
    metric = st.selectbox("Select the Metric", 
                          options=["Confusion Matrix","ROC-AUC", "Sensitivity Analysis"])
    [features, target] = whole_processor(df_back)
    explainer = shap.TreeExplainer(model = model)
    features_shap_values = explainer(features)
    if metric == "Confusion Matrix":
        p_threshold = st.slider("Select Threshold", min_value=0.0, max_value=1.0,
                              value=0.5,step=0.05)
        
        model.set_probability_threshold(p_threshold)
        # make predictions for churn
        pred = model.predict(features)
        
        col1, col2 = st.columns((2,1))
        
        # confusion matrix
        with col1:
            fig, ax = plt.subplots(figsize = (3,3))
            conf_mat = confusion_matrix(target, pred)
            sns.heatmap(conf_mat, cmap='Blues', cbar=False, annot=True, fmt='.4g',
                        xticklabels=['No', 'Yes'],yticklabels=['No', 'Yes'], ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("True Labels")
            
            st.pyplot(fig)
        
        with col2:
            
            st.markdown(f'F1 Score: {round(f1_score(target, pred),3)}')
            st.markdown(f'Recall: {round(recall_score(target, pred),3)}')
        
    if metric == "ROC-AUC":
        fig, ax = plt.subplots(figsize = (2,2))
        plot_roc_curve(estimator=model, X=features, y=target)
        plt.plot([0,1], [0,1], linestyle='--', color='orange')
        st.pyplot()

    if metric == "Sensitivity Analysis":
        st.markdown("**Global Variable Importance (SHAP Summary Plot)**")
        shap.summary_plot(features_shap_values, plot_type="bar",plot_size=(12,12))
        st.pyplot()
#preprocess_raw(df)

def embed_iframe():
    st.components.v1.iframe("https://public.tableau.com/app/profile/kaan.nl./viz/MIS463-Group7-TelcoChurnAnalysis/ChurnTenure?:showVizHome=no&:embed=true"
    , width=1800, height=1250, scrolling=True)
    

if "input_data" not in st.session_state:
    init_df = pd.DataFrame(index = [0], columns=selected_features)  #pd.DataFrame(index = [0], columns=selected_features)
    init_df["customerID"][0] = 'Initial'
    st.session_state.input_data = init_df


select_display = st.sidebar.radio("Select Module", options=["Data Select","Prediction Results & Plots","General Metrics & Sensitivity Analysis", "Dashboard - iframe"])
if select_display == "Data Select":
    page_input_select()
if select_display == "Prediction Results & Plots":
    result_func()
if select_display == "General Metrics & Sensitivity Analysis":
    metric_analysis()
if select_display == "Dashboard - iframe":
    embed_iframe()


