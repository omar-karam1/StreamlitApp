import streamlit as st
import pandas as pd
import io
from langchain_ollama import OllamaLLM
import numpy as np
from scipy.stats import zscore
import plotly.express as px

def main():
    st.title("Data Quality Task App")
   
    
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])
 

    if uploaded_file:
       
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        
        if 'df' not in st.session_state:
            st.session_state.df = df
        if 'temp_df' not in st.session_state:
            st.session_state.temp_df = st.session_state.df.copy()

        df = st.session_state.temp_df

       
        missing_columns = df.columns[df.isnull().any()].tolist()
        duplicate_rows_count = df.duplicated().sum()
        outlier_columns = []

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            z_scores = np.abs(zscore(df[column].dropna())) 
            if (z_scores > 3).any():  
                outlier_columns.append(column)

       
        st.sidebar.header("Options")
        task = st.sidebar.radio("Select a Task", [
            "Show data",
            "Info",
            "Describe",
            "Change Data Type",
            "Rename Column",
            "Issues",
            "Handle Missing Values",
            "Handle Duplicates",
            "Handle Outliers",
            "Visualization",
            "Chat using RAG"
        ])

       
        if task=="Show data":
            st.write(df.head())

        elif task == "Info":
            st.write("### Info")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

       
        elif task == "Describe":
            st.subheader("Dataset Description")
            st.write(df.describe())
        
        elif task == "Change Data Type":
            st.subheader("Change Data Type of a Column")
            column = st.selectbox("Select Column to Change Type", df.columns)
            new_type = st.selectbox("Select New Data Type", ["int", "float", "string", "datetime"])

            if st.button("Change Data Type"):
                try:
                    if new_type == "int":
                        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
                    elif new_type == "float":
                        df[column] = pd.to_numeric(df[column], errors="coerce")
                    elif new_type == "string":
                        df[column] = df[column].astype(str)
                    elif new_type == "datetime":
                        df[column] = pd.to_datetime(df[column], errors="coerce")

                    st.session_state.temp_df = df.copy()
                    st.success(f"Column '{column}' type changed to {new_type}.")
                except Exception as e:
                    st.error(f"Error changing column type: {e}")

       
        elif task == "Rename Column":
            st.subheader("Rename a Column")
            column = st.selectbox("Select Column to Rename", df.columns)
            new_name = st.text_input("Enter New Column Name")

            if st.button("Rename Column"):
                if new_name:
                    df.rename(columns={column: new_name}, inplace=True)
                    st.session_state.temp_df = df.copy()
                    st.success(f"Column '{column}' renamed to '{new_name}'.")
                else:
                    st.warning("Please enter a valid name.")

       
        elif task == "Issues":
            st.subheader("Data Quality Issues")

            st.write("### Missing Values:")
            if missing_columns:
                st.write(f"Columns with missing values: {missing_columns}")
            else:
                st.write("No missing values detected.")

            st.write("### Duplicate Rows:")
            if duplicate_rows_count > 0:
                st.write(f"Number of duplicate rows: {duplicate_rows_count}")
            else:
                st.write("No duplicate rows detected.")

            st.write("### Outliers:")
            if outlier_columns:
                st.write(f"Columns with outliers (Z-Score > 3): {outlier_columns}")
            else:
                st.write("No outliers detected.")

        
        elif task == "Handle Missing Values":
            if missing_columns:
                st.subheader("Handle Missing Values")
                selected_column = st.selectbox("Select Column with Missing Values", missing_columns)
                temp_df_copy = st.session_state.temp_df.copy()
                col1, col2 = st.columns(2)
               
            
                 

                with col1:
                    st.write("### Before")
                    st.write(temp_df_copy[selected_column].isnull().sum())

                action = st.selectbox("Select Action", ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Custom Value"])

                if action == "Drop Rows":
                    temp_df_copy = temp_df_copy.dropna(subset=[selected_column])
                elif action == "Fill with Mean":
                    if pd.api.types.is_numeric_dtype(temp_df_copy[selected_column]):
                        temp_df_copy[selected_column] = temp_df_copy[selected_column].fillna(temp_df_copy[selected_column].mean())
                    else:
                        st.warning(f"Cannot fill missing values with mean in column '{selected_column}' because it is not numeric.")
                elif action == "Fill with Median":
                    if pd.api.types.is_numeric_dtype(temp_df_copy[selected_column]):
                        temp_df_copy[selected_column] = temp_df_copy[selected_column].fillna(temp_df_copy[selected_column].median())
                    else:
                        st.warning(f"Cannot fill missing values with median in column '{selected_column}' because it is not numeric.")
                elif action == "Fill with Mode":
                    if pd.api.types.is_object_dtype(temp_df_copy[selected_column]):
                        mode_value = temp_df_copy[selected_column].mode().iloc[0] if not temp_df_copy[selected_column].mode().empty else None
                        temp_df_copy[selected_column] = temp_df_copy[selected_column].fillna(mode_value)
                    else:
                        st.warning(f"Cannot fill missing values with mode in column '{selected_column}' because it is not a string column.")
                elif action == "Fill with Custom Value":
                    custom_value = st.text_input("Enter Custom Value")
                    temp_df_copy[selected_column] = temp_df_copy[selected_column].fillna(custom_value)

                with col2:
                    st.write("### After")
                    st.write(temp_df_copy[selected_column].isnull().sum())

                if st.button("Save Missing Value Changes"):
                    st.session_state.temp_df = temp_df_copy.copy()  
                    st.success("Changes saved successfully!")
                    st.rerun()
                    
                  
            else:
                st.success("No missing values detected")
                
        
        elif task == "Handle Duplicates":
            if duplicate_rows_count > 0:
                st.subheader("Handle Duplicates")
                col1, col2 = st.columns(2)

                temp_df_copy = st.session_state.temp_df.copy()  

                with col1:
                    st.write("### Before")
                    st.write(temp_df_copy.duplicated().sum())

                if st.button("Remove Duplicates"):
                    temp_df_copy = temp_df_copy.drop_duplicates()
                    st.rerun()

                with col2:
                    st.write("### After")
                    st.write(temp_df_copy.duplicated().sum())

                if st.button("Save Duplicate Changes"):
                    st.session_state.temp_df = temp_df_copy.copy()  
                    st.success("Changes saved successfully!")
                    st.rerun()
            else:
              st.success("No duplicate rows detected.")
       
        elif task == "Handle Outliers":
            if outlier_columns:
                st.subheader("Handle Outliers")
                selected_column = st.selectbox("Select Column with Outliers", outlier_columns)
                method = st.selectbox("Select Method", ["Z-Score", "IQR"])

                col1, col2 = st.columns(2)

                temp_df_copy = st.session_state.temp_df.copy() 
                with col1:
                    st.write("### Before")
                    st.write(temp_df_copy[selected_column])

                if method == "Z-Score":
                    temp_df_copy = temp_df_copy[np.abs(zscore(temp_df_copy[selected_column].dropna())) < 3]
                elif method == "IQR":
                    Q1 = temp_df_copy[selected_column].quantile(0.25)
                    Q3 = temp_df_copy[selected_column].quantile(0.75)
                    IQR = Q3 - Q1
                    temp_df_copy = temp_df_copy[(temp_df_copy[selected_column] >= (Q1 - 1.5 * IQR)) & (temp_df_copy[selected_column] <= (Q3 + 1.5 * IQR))]

                with col2:
                    st.write("### After")
                    st.write(temp_df_copy[selected_column])

                if st.button("Save Changes"):
                   st.session_state.temp_df = temp_df_copy.copy()  
                   st.success("Changes saved successfully!")
                   st.rerun()  
            else:
              st.success("No outliers detected..")
        

   


        elif task == "Visualization":
            st.subheader("Data Visualization")

            
            columns = df.columns.tolist()
            selected_column = st.selectbox("Select a Column for Visualization", columns)

              
            plot_type = st.selectbox("Select Plot Type", ["Histogram", "Boxplot", "Scatterplot", "Correlation Heatmap"])

            if plot_type == "Histogram":
                bins = st.slider("Number of Bins", min_value=5, max_value=50, value=10)
                fig = px.histogram(df, x=selected_column, nbins=bins, title=f"Histogram of {selected_column}")
                st.plotly_chart(fig)

            elif plot_type == "Boxplot":
                 fig = px.box(df, y=selected_column, title=f"Boxplot of {selected_column}")
                 st.plotly_chart(fig)

            elif plot_type == "Scatterplot":
                 numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                 second_column = st.selectbox("Select a Second Column for Scatterplot", numeric_columns)
                 fig = px.scatter(df, x=selected_column, y=second_column, title=f"Scatterplot of {selected_column} vs {second_column}")
                 st.plotly_chart(fig)

            elif plot_type == "Correlation Heatmap":
                 numeric_columns = df.select_dtypes(include=[np.number])
                 corr_matrix = numeric_columns.corr()
                 fig = px.imshow(
                          corr_matrix, 
                          text_auto=True, 
                          color_continuous_scale="viridis",  
                          title="Correlation Heatmap"
                        )

                 st.plotly_chart(fig)

        
       
        elif task == "Chat using RAG":
            st.subheader("Ask Questions About Your Dataset")
            llm = OllamaLLM(model="llama3.2")

            user_question = st.text_input("Type your question here:")

            if st.button("Submit Question"):
                if user_question:
                    data_as_text = df.to_string(index=False)

                    prompt = f"""
                    You are an AI assistant that answers questions based on the provided data.
                    Data: {data_as_text}
                    Question: {user_question}
                    Answer:"""

                    response = llm.invoke(input=prompt, stop=["<|eot.id|>"])

                    st.write("### Response:")
                    st.write(response)

       
        st.sidebar.header("Download Dataset")
        st.sidebar.download_button(
            label="Download cleaned dataset",
            data=st.session_state.temp_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
  
if __name__ == "__main__":
    main()
