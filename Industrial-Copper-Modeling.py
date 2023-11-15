# Imports
import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
from sklearn import metrics
# from sklearn.metrics import accuracy_score,confusion_matrix,recall_score
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.svm import SVR, SVC
import numpy as np
import random

# List of algorithm for Regression
reg_algo = {'DecisionTreeRegressor': DecisionTreeRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'AdaBoostRegressor': AdaBoostRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
            'SVR': SVR}
# List of algorithm for Classification
clas_algo = {'DecisionTreeClassifier': DecisionTreeClassifier,
             'RandomForestClassifier': RandomForestClassifier,
             'AdaBoostClassifier': AdaBoostClassifier,
             'GradientBoostingClassifier': GradientBoostingClassifier,
             'ExtraTreesClassifier': ExtraTreesClassifier,
             'SVC': SVC}
# Page configuration
st.set_page_config(layout="wide", page_icon=Image.open(r"D:\copper proj\copper img.webp"),
                   page_title="Industrial Copper Modeling")
selected = option_menu(None, ["Home", "To_find_selling_price", "To_find_to_status"], icons=["house", "pencil-square ",
                                                                                            "trash"],default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"}})

# Selecting Home Menu
if selected == 'Home':
    # Open the image
    image = Image.open(r"D:\copper proj\copper.jpg")

    # Resize the image to the desired dimensions
    new_width = 300  # Set your desired width
    new_height = 200  # Set your desired height
    resized_image = image.resize((new_width, new_height))
    # Display the resized image
    st.image(resized_image, caption="Industrial Copper")
    st.title('Industrial Copper Modeling')
    st.write(
        "Welcome to the Industrial Copper Modeling project. This project focuses on applying machine "
        "learning techniques to address challenges in the copper industry related to sales and pricing.")
    st.write("Here are the key objectives of this project:")
    st.write("1. Data Preprocessing: Handle missing values, outliers, and skewness.")
    st.write("2. ML Modeling: Build regression and classification models to predict selling prices and lead status.")
    st.write("3. Streamlit GUI: Create an interactive page to make predictions on new data.")
    st.write(
        "4. Learning Outcomes: Develop proficiency in Python programming, "
        "data analysis, and machine learning techniques.")
# Selecting Regression dependent variable Selling price
if selected == 'To_find_selling_price':
    # Selecting Test percentage i.e. 10% to 40%
    st.subheader('Selecting Test percentage 10% to 40%')
    test_size = st.slider('Test percentage', 0.10, 0.40, 0.15)
    st.write('Selected Test Size : ',test_size * 100, '%')
    # Regression cleaned files
    df = pd.read_csv(r"D:\copper proj\regressor.csv")
    y = df['selling_price']
    x = df.drop(['selling_price'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    st.subheader('Select a Regression Algorithm')
    algo = st.selectbox('Select after selecting Algorithm Predict_Algorithm button for test ', list(reg_algo.keys()))
    if algo:
        st.write('Selected Algorithm : ',algo)
    model_class = reg_algo.get(algo)
    if model_class:
        model = model_class()
        result = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    else:
        st.write(f"Model '{algo}' is not supported.")

    if st.button('Predict_Algorithm'):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        st.write('Predicted Algorithm Test')
        st.write('Mean Absolute Error (MSE):', mae)
        st.write('Mean Squared Error (MSE):', mse)
        st.write('Root Mean Squared Error (RMSE):', rmse)
        st.write('R-squared (R2):', r2)
    st.write('*****************')
    # Split the Predict column
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.subheader('Product Reference')
        st.write('Select the product Reference Button to change new reference')
        product_data = x.product_ref.unique()
        product_refer = random.choice(product_data)
        if st.button('Product_ref'):
            product_data = x.product_ref.unique()
            product_refer = random.choice(product_data)
        st.write('Random_Product_ref', product_refer)
        st.subheader('Item Date')
        st.write('Select the date for Item Date')
        date = st.slider(
            "select the date",
            min_value=datetime(2020, 7, 2), max_value=datetime(2021, 4, 1),
            format="DD/MM/YY")
        st.write("Selected Item_date", date)
        item_date = float(date.strftime("%Y%m%d.0"))

        st.subheader('Quantity in Ton')
        st.write('Select any numerical value in Decimal')
        quantity = st.number_input('Quantity tons', 768.0)
        st.write('Quantity tons', quantity)

        st.subheader('Customer in count')
        st.write('Select any numerical value ')
        customer = st.number_input('Customer', 12458.0)
        st.write('Customer', customer)

        st.subheader('Item Type')
        st.write('Select Any type')
        item_type0 = st.selectbox('Item_type', ('IPL', 'others', 'PL', 'S', 'SLAWR', 'W', 'WI'))
        st.write(f'Selected Item Type : :green[{item_type0}')
        if item_type0 == 'IPL':
            item_type = 0
        elif item_type0 == 'others':
            item_type = 1
        elif item_type0 == 'PL':
            item_type = 2
        elif item_type0 == 'S':
            item_type = 3
        elif item_type0 == 'SLAWR':
            item_type = 4
        elif item_type0 == 'W':
            item_type = 5
        else:
            item_type = 6


        st.subheader('Material Reference')
        st.write('Select any Material reference')
        material_ref0 = st.selectbox('material_ref', ('DEQ1 S460MC', '0',
                                         'S0380700', 'other', 'MAS65550', '4.11043_1060X5_BRE',
                                         '202006170005.IO.1.1'))
        st.write(f'Selected Material Reference : :green[{material_ref0}]')
        if material_ref0 == 'DEQ1 S460MC':
            material_ref = 5
            st.write(5)
        elif material_ref0 == '0':
            material_ref = 6
            st.write(6)
        elif material_ref0 == 'S0380700':
            material_ref = 3
            st.write(3)
        elif material_ref0 == 'other':
            material_ref = 1
            st.write(1)
        elif material_ref0 == 'MAS65550':
            material_ref = 2
            st.write(2)
        elif material_ref0 == '4.11043_1060X5_BRE':
            material_ref = 0
            st.write(0)
        else:
            material_ref = 4
            st.write(4)

        st.subheader('Country')
        st.write('Select the Country code for reference 0 for unknown customer')
        country = st.slider('Country', 25, 113, 26)
        st.write(country)

    with col2:
        st.subheader('Application')
        st.write('Select the any number')
        application = st.slider('Selected Application : ', 2, 100, 41)
        st.write(application)

        st.subheader('Thickness')
        st.write('Select the Thickness in number')
        thickness = st.number_input('Thickness', 2)
        st.write('Selected Thickness', thickness)

        st.subheader('Width')
        st.write('Select the Width in number')
        width = st.number_input('Width', 70)
        st.write('Selected Width', width)

        st.subheader('Item_Status')
        st.write('Selecte the Status of data')
        item_status = st.selectbox('Status', ('Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
                                   'Wonderful', 'Revised', 'Offered', 'Offerable'))
        st.write(f'Selected Stetus : :green[{item_status}')
        if item_status == 'Won':
            status = 7
        elif item_status == 'Draft':
            status = 0
        elif item_status == 'To be approved':
            status = 6
        elif item_status == 'Lost':
            status = 1
        elif item_status == 'Not lost for AM':
            status = 2
        elif item_status == 'Wonderful':
            status = 8
        elif item_status == 'Revised':
            status = 5
        elif item_status == 'Offered':
            status = 4
        else:
            status = 3

        st.subheader('Delivery Date')
        st.write('Select the Delivery Date')
        date1 = st.slider("select the date", min_value=datetime(2020, 7, 2), max_value=datetime(2021, 4, 1),
                          key="date_slider_1", format="DD/MM/YY")
        st.write("Selected Date :", date1)
        delivery_date = float(date.strftime("%Y%m%d.0"))

    with col3:
        st.subheader('Prediction Unknown Data')
        st.write("""Selecting Prediction Button to Predicting the unknown data from the previous selection. 
        """)
        if st.button('Predict'):
            st.write('The predicted Selling Price : ')
            prid_data = np.array(
                [[item_date, quantity, customer, country, status, item_type, application, thickness, width,
                  material_ref, product_refer, delivery_date]])
            given_predict = result.predict(prid_data)
            st.write(given_predict)

if selected == 'To_find_to_status':
    # Selecting Test percentage i.e. 10% to 40%
    st.subheader('Selecting Test percentage 10% to 40%')
    test_size = st.slider('Test percentage', 0.10, 0.40, 0.15)
    st.write('Selected Test Size : ', test_size * 100, '%')
    # Classification cleaned files
    df1 = pd.read_csv(r"D:\copper proj\classifier.csv")
    y1 = df1['status']
    x1 = df1.drop(['status'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=test_size)
    st.subheader('Select a Classification algorithm ')
    algo1 = st.selectbox('Select after selecting Algorithm Predict_Algorithm button for test', list(clas_algo.keys()))
    if algo1:
        st.write('Selected Algorithm :',algo1)
    model_class = clas_algo.get(algo1)
    if model_class:
        model = model_class()
        result = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    else:
        st.write(f"Model '{algo1}' is not supported.")

    if st.button('Predict_Algo'):
        result = model.fit(x_train, y_train)
        y_pred = result.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write('Accuracy : ', accuracy)
        st.write('Confusion_matrix :', metrics.confusion_matrix(y_test, y_pred))
        st.write('Recall_score :', metrics.recall_score(y_test, y_pred, average= 'macro'))
    st.write('************')


    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.subheader('Product Reference')
        st.write('Select the product Reference Button to change new reference')
        product_data = x1.product_ref.unique()
        product_refer = random.choice(product_data)
        if st.button('Product_Reference'):
            product_data = x1.product_ref.unique()
            product_refer = random.choice(product_data)
        st.write('Random_Producted_ref', product_refer)

        st.subheader('Item Date')
        st.write('Select the date for Item Date')
        date = st.slider(
            "Select the date",
            min_value=datetime(2020, 7, 2), max_value=datetime(2021, 4, 1),
            format="DD/MM/YY")
        st.write("Selected Item_date", date)
        item_date = float(date.strftime("%Y%m%d.0"))

        st.subheader('Quantity in Tons')
        st.write('Select any Numerical value in Decimal')
        quantity = st.number_input('Quantity tons', 0.02)
        st.write('Quantity tons', quantity)

        st.subheader('Customer')
        st.write('Select any numerical value')
        customer = st.number_input('Customer', 12458.0)
        st.write('Customer', customer)

        st.subheader('Item Type')
        st.write('Select Any type')
        item_type0 = st.selectbox('Item_type', ('IPL', 'others', 'PL', 'S', 'SLAWR', 'W', 'WI'))
        st.write(f'Item type : :green[{item_type0}]')
        if item_type0 == 'IPL':
            item_type = 0
        elif item_type0 == 'others':
            item_type = 1
        elif item_type0 == 'PL':
            item_type = 2
        elif item_type0 == 'S':
            item_type = 3
        elif item_type0 == 'SLAWR':
            item_type = 4
        elif item_type0 == 'W':
            item_type = 5
        else:
            item_type = 6

        st.subheader('Material Reference')
        st.write('Select any Material Reference')
        material_ref0 = st.selectbox('Selected Material Reference', ('DEQ1 S460MC', '0',
                                          'S0380700', 'other', 'MAS65550', '4.11043_1060X5_BRE',
                                          '202006170005.IO.1.1'))
        st.write(f'Selected Material Reference : :green[{material_ref0}]')
        if material_ref0 == 'DEQ1 S460MC':
            material_ref = 5
        elif material_ref0 == '0':
            material_ref = 6
        elif material_ref0 == 'S0380700':
            material_ref = 3
        elif material_ref0 == 'other':
            material_ref = 1
        elif material_ref0 == 'MAS65550':
            material_ref = 2
        elif material_ref0 == '4.11043_1060X5_BRE':
            material_ref = 0
        else:
            material_ref = 4

    with col2:
        st.subheader('Country')
        st.write('Select any country code')
        country = st.slider('Country', 25, 113, 26)
        st.write(country)

        st.subheader('Application')
        st.write('Select any Application')
        application = st.slider('Application', 2, 100, 26)
        st.write('Selected Application :', application)

        st.subheader('Thickness')
        st.write('Select any Thickness')
        thickness = st.number_input('Thickness', 2)
        st.write('Selected Thickness :', thickness)

        st.subheader('Width')
        st.write('Select any Width')
        width = st.number_input('Width', 70)
        st.write('Selected Width :', width)

        st.subheader('Delivery Date')
        date1 = st.slider("Select the date", min_value=datetime(2020, 7, 2), max_value=datetime(2021, 4, 1),
                          key="date_slider_1", format="DD/MM/YY")
        st.write("Selected Delivery Date :", date1)
        delivery_date = float(date.strftime("%Y%m%d.0"))

        st.subheader('Selling Price')
        st.write('Select Any Selling Price ')
        selling_price_data = x1.selling_price.unique()
        selling_price = random.choice(selling_price_data)
        if st.button('Selling Price'):
            selling_price_data = x1.selling_price.unique()
            selling_price = random.choice(selling_price_data)
        st.write('Random_selling_price', selling_price)
    with col3:
        st.subheader('Predict Unknown Data')
        st.write('Selecting Prediction Button to Predicting the unknown data from the previous selection.')
        if st.button('Predict Unknown Data'):

            prid_data = np.array(
                [[item_date, quantity, customer, country, item_type, application, thickness, width,
                  material_ref, product_refer, delivery_date,selling_price]])
            given_predict = result.predict(prid_data)

            if given_predict == 7:
                st.write('Won')
            elif given_predict == 1:
                st.write('Lost')
