
# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# Set up the app title and image
st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', use_column_width = True, 
         caption = "Utilize our advanced machine learning application to predict fetal health classifications")

default_df = pd.read_csv('fetal_health.csv')
sample_df = default_df.drop(columns=['fetal_health'])
with st.sidebar:
    st.header('Upload your data')
    userdiamond = st.sidebar.file_uploader('Choose a CSV File')
    st.header('Sample Data Format for Upload')
    st.write(sample_df.head())
    model = st.radio(
    "Choose method for prediction",
    ["Random Forest","Decision Tree","AdaBoost","Soft Voting"])
    st.write("You selected:", model)
if userdiamond:
    if model == "Decision Tree":
            dt_pickle = open('decision_tree_health.pickle', 'rb') 
            clf = pickle.load(dt_pickle) 
            dt_pickle.close()


            user_df = pd.read_csv(userdiamond) # User provided data
            original_df = pd.read_csv('fetal_health.csv') # Original data to create ML model
            
            # Dropping null values
            user_df = user_df.dropna() 
            original_df = original_df.dropna() 
            
            
            original_df = original_df.drop(columns = ['fetal_health'])
            
            
            # Ensure the order of columns in user data is in the same order as that of original data
            user_df = user_df[original_df.columns]

            # Concatenate two dataframes together along rows (axis = 0)
            combined_df = pd.concat([original_df, user_df], axis = 0)

            # Number of rows in original dataframe
            original_rows = original_df.shape[0]

            # Create dummies for the combined dataframe
            combined_df_encoded = pd.get_dummies(combined_df)

            # Split data into original and user dataframes using row index
            original_df_encoded = combined_df_encoded[:original_rows]
            user_df_encoded = combined_df_encoded[original_rows:]

            # Predictions for user data
            user_pred = clf.predict(user_df_encoded)

            # Predicted species
            user_pred_species = user_pred

            # Adding predicted species to user dataframe
            user_df['Predicted Price'] = user_pred_species

            predicitons = clf.predict_proba(user_df_encoded)
        
            
            predicted_prob_percent=predicitons.max(axis=1)*100
            user_df['Probability']=predicted_prob_percent
            predictions = clf.predict_proba(user_df_encoded)
    
    if model == "Random Forest":
            rf_pickle = open('randomforest_health.pickle', 'rb') 
            clf_rf = pickle.load(rf_pickle) 
            rf_pickle.close()


            user_df = pd.read_csv(userdiamond) # User provided data
            original_df = pd.read_csv('fetal_health.csv') # Original data to create ML model
            
            # Dropping null values
            user_df = user_df.dropna() 
            original_df = original_df.dropna() 
            
            
            original_df = original_df.drop(columns = ['fetal_health'])
            
            
            # Ensure the order of columns in user data is in the same order as that of original data
            user_df = user_df[original_df.columns]

            # Concatenate two dataframes together along rows (axis = 0)
            combined_df = pd.concat([original_df, user_df], axis = 0)

            # Number of rows in original dataframe
            original_rows = original_df.shape[0]

            # Create dummies for the combined dataframe
            combined_df_encoded = pd.get_dummies(combined_df)

            # Split data into original and user dataframes using row index
            original_df_encoded = combined_df_encoded[:original_rows]
            user_df_encoded = combined_df_encoded[original_rows:]

            # Predictions for user data
            user_pred = clf_rf.predict(user_df_encoded)

            # Predicted species
            user_pred_species = user_pred

            # Adding predicted species to user dataframe
            user_df['Predicted Price'] = user_pred_species

            predicitons = clf_rf.predict_proba(user_df_encoded)
        
            
            predicted_prob_percent=predicitons.max(axis=1)*100
            user_df['Probability']=predicted_prob_percent
            predictions = clf_rf.predict_proba(user_df_encoded)
    if model == "Soft Voting":
            sf_pickle = open('softvoting_health.pickle', 'rb') 
            clf_sf = pickle.load(sf_pickle) 
            sf_pickle.close()


            user_df = pd.read_csv(userdiamond) # User provided data
            original_df = pd.read_csv('fetal_health.csv') # Original data to create ML model
            
            # Dropping null values
            user_df = user_df.dropna() 
            original_df = original_df.dropna() 
            
            
            original_df = original_df.drop(columns = ['fetal_health'])
            
            
            # Ensure the order of columns in user data is in the same order as that of original data
            user_df = user_df[original_df.columns]

            # Concatenate two dataframes together along rows (axis = 0)
            combined_df = pd.concat([original_df, user_df], axis = 0)

            # Number of rows in original dataframe
            original_rows = original_df.shape[0]

            # Create dummies for the combined dataframe
            combined_df_encoded = pd.get_dummies(combined_df)

            # Split data into original and user dataframes using row index
            original_df_encoded = combined_df_encoded[:original_rows]
            user_df_encoded = combined_df_encoded[original_rows:]

            # Predictions for user data
            user_pred = clf_sf.predict(user_df_encoded)

            # Predicted species
            user_pred_species = user_pred

            # Adding predicted species to user dataframe
            user_df['Predicted Price'] = user_pred_species

            predicitons = clf_sf.predict_proba(user_df_encoded)
        
            
            predicted_prob_percent=predicitons.max(axis=1)*100
            user_df['Probability']=predicted_prob_percent
            predictions = clf_sf.predict_proba(user_df_encoded)
    if model == "AdaBoost":
             ada_pickle = open('ada_boost.pickle', 'rb') 
             clf_ada = pickle.load(ada_pickle) 
             ada_pickle.close()


             user_df = pd.read_csv(userdiamond) # User provided data
             original_df = pd.read_csv('fetal_health.csv') # Original data to create ML model
            
             
             user_df = user_df.dropna() 
             original_df = original_df.dropna() 
            
            
             original_df = original_df.drop(columns = ['fetal_health'])
            
            
             # Ensure the order of columns in user data is in the same order as that of original data
             user_df = user_df[original_df.columns]

             # Concatenate two dataframes together along rows (axis = 0)
             combined_df = pd.concat([original_df, user_df], axis = 0)

             # Number of rows in original dataframe
             original_rows = original_df.shape[0]

             # Create dummies for the combined dataframe
             combined_df_encoded = pd.get_dummies(combined_df)

    #         # Split data into original and user dataframes using row index
             original_df_encoded = combined_df_encoded[:original_rows]
             user_df_encoded = combined_df_encoded[original_rows:]

    #         # Predictions for user data
             user_pred = clf_ada.predict(user_df_encoded)

    #         # Predicted species
             user_pred_species = user_pred

    #         # Adding predicted species to user dataframe
             user_df['Predicted Price'] = user_pred_species

             predicitons = clf_ada.predict_proba(user_df_encoded)
        
            
             predicted_prob_percent=predicitons.max(axis=1)*100
             user_df['Probability']=predicted_prob_percent
             predictions = clf_ada.predict_proba(user_df_encoded)

    ##############################################################################################################################     
 ### Used ChatGPT see appendix for details
    def highlight_conditions(val):
        if val == "Normal":
            color = "background-color: lime" 
        elif val == "Suspect":
            color = "background-color: yellow" 
        elif val == "Pathological":
            color = "background-color: orange" 
        else:
            color = ""  # No color if not matched
        return color

    # Apply the styling to the 'Condition' column
    styled_df = user_df.style.applymap(highlight_conditions, subset=['Predicted Price'])
    ##############################################################################################################################  
    # Display the styled DataFrame in Streamlit
    st.dataframe(styled_df)
    # # Tab 1: Visualizing Decision Tree
    st.subheader("Prediction Performance")
    if model == "Decision Tree":
        tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])
    if model == "Random Forest":
        tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
    if model == "AdaBoost":
        tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
    if model == "Soft Voting":
        tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
    if model == "Decision Tree":
        with tab1:
            st.write("### Decision Tree Visualization")
            st.image('dt_visual.svg')
            st.caption("Visualization of the Decision Tree used in prediction.")

    #Tab 2: Feature Importance Visualization

    with tab2:
        st.write("### Feature Importance")
        if model == "Decision Tree":
         st.image('dt_feature_imp.svg')
        if model == "Random Forest":
            st.image('rf_feature_imp.svg')
        if model == "AdaBoost":
            st.image('ada_feature_imp.svg')
        if model == "Soft Voting":
            st.image('sf_feature_imp.svg')
        st.caption("Features used in this prediction are ranked by relative importance.")
    

    # Tab 3: Confusion Matrix
    with tab3:
        st.write("### Confusion Matrix")
        if model == "Decision Tree":
            st.image('dtconfusion_mat.svg')
        if model == "Random Forest":
            st.image('rfconfusion_mat.svg')
        if model == "AdaBoost":
            st.image('adaconfustion_mat.svg')
        if model == "Soft Voting":
            st.image('sfconfusion_mat.svg')
        st.caption("Confusion Matrix of model predictions.")


    # Tab 4: Classification Report
    with tab4:
        st.write("### Classification Report")
        if model == "Decision Tree":
            report_df = pd.read_csv('dt_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each device.")
        if model == "Random Forest":
            report_df = pd.read_csv('rf_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each device.")
        if model == "AdaBoost":
            report_df = pd.read_csv('ada_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each device.")
        if model == "Soft Voting":
            report_df = pd.read_csv('sf_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each device.")
        