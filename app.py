import streamlit as st
import pandas as pd
import io
import time
from langchain_ollama import OllamaLLM
import numpy as np
import plotly.express as px
import functions

def main():
    st.set_page_config(
        page_title="Data Refiner",
        page_icon="📊"
    )

    st.title("Data Refiner App")

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
        outlier_columns = functions.get_outlier_columns(df)

        st.sidebar.header("Options")
        task = st.sidebar.radio("Select a Task", [
            "Show data",
            "Info",
            "Describe",
            "Change Data Type",
            "Rename Column",
            "Range Check",
            "Delete Column",
            "Issues",
            "Handle Missing Values",
            "Handle Duplicates",
            "Handle Outliers",
            "Visualization",
            "Feature Scaling",
            "Encoding",
            "Split Data",
            "Model",
            "Use Model",
            "Chat using RAG"
        ])

        if task == "Show data":
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
            functions.Change_Data_Type(column, new_type, df)

        elif task == "Rename Column":
            st.subheader("Rename a Column")
            functions.Rename_Column(df)

        elif task == "Range Check":
            functions.range_check(df)

        elif task == "Issues":
            st.subheader("Data Quality Issues")
            functions.Issues(missing_columns, duplicate_rows_count, outlier_columns)

        elif task == "Handle Missing Values":
            functions.handle_missing_values(missing_columns)

        elif task == "Handle Duplicates":
            functions.Handle_Duplicates(duplicate_rows_count)

        elif task == "Delete Column":
            if not len(df.columns) == 0:
                column = st.selectbox("Select Column", df.columns)
                if st.button("Delete Column"):
                    df.drop(column, axis=1, inplace=True)
                    st.success(f"Column '{column}' deleted successfully.")
                    time.sleep(1)
                    st.rerun()
            else:
                st.warning("No columns available.")

        elif task == "Handle Outliers":
            functions.handle_outliers(outlier_columns)

        elif task == "Feature Scaling":
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            functions.handle_feature_scaling(df, numeric_columns)

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

        elif task == "Split Data":
            st.subheader("Split Data into Train and Test Sets")
            target_column = st.selectbox("Select the Target Variable", options=df.columns, index=len(df.columns) - 1)
            test_size = st.slider("Select Test Size (as a percentage)", min_value=10, max_value=50, value=20, step=5) / 100
            if st.button("Split Data"):
                functions.split_data(df, target_column, test_size)

        elif task == "Encoding":
            columns = df.select_dtypes(include=['object']).columns
            if len(columns) == 0:
                st.warning("No text columns found in the dataset.")
            else:
                st.subheader("Label or One-Hot Encoding")
                encoding_option = st.selectbox("Select Encoding Type", ["Label Encoding", "One-Hot Encoding"])
                column = st.selectbox("Select Column", columns)
                functions.encoding(df, encoding_option, column)

        elif task == "Model":
            st.subheader("Model Evaluation")
            model_option = st.selectbox("Select a Model", ['XGBoost Classifier', 'Support Vector Machine', 'Random Forest Classifier'])
            if 'X_train' in st.session_state and 'y_train' in st.session_state:
                if st.button("Evaluate Model"):
                    X_train = st.session_state['X_train']
                    X_test = st.session_state['X_test']
                    y_train = st.session_state['y_train']
                    y_test = st.session_state['y_test']
                    functions.evaluate_model(model_option, X_train, X_test, y_train, y_test)
            else:
                st.error("Please split the data first before evaluating a model.")

        elif task == "Use Model":
            st.subheader("Use Trained Model for Prediction")
            functions.predict_with_model()

        elif task == "Chat using RAG":
            st.subheader("Ask Questions About Your Dataset")
            llm = OllamaLLM(model="llama3.2")
            messages = st.session_state.get("chat_messages", [])
            user_question = st.chat_input("Type your question here:")

            if user_question:
                messages.append({"role": "user", "content": user_question})
                data_as_text = df.to_string(index=False)
                prompt = f"""
                    You are an AI assistant that answers questions based on the provided data.
                    Data: {data_as_text}
                    Question: {user_question}
                    Answer:"""

                response = llm.invoke(input=prompt, stop=["<|eot.id|>"])
                messages.append({"role": "assistant", "content": response})
                st.session_state["chat_messages"] = messages

            for message in messages:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                elif message["role"] == "assistant":
                    st.chat_message("assistant").write(message["content"])

        st.sidebar.header("Download Dataset")
        st.sidebar.download_button(
            label="Download cleaned dataset",
            data=st.session_state.temp_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
