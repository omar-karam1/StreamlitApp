# DataRefiner App

**DataRefiner App** is an interactive web application built using [Streamlit](https://streamlit.io/) to analyze, clean, and enhance the quality of datasets. The app offers various tools to handle common data issues, such as missing values, duplicates, and outliers, along with data exploration, visualization, machine learning model building, and an AI-driven question-answering feature using a Large Language Model (LLM).

## **Key Features**
### **1. Upload and Preview Datasets**
- Supports CSV and Excel file formats.
- View and explore the dataset for initial insights.

### **2. Data Quality Analysis**
- Identify data issues like:
  - Missing values.
  - Duplicate rows.
  - Outliers using:
    - Z-Score analysis.
    - Interquartile Range (IQR).

### **3. Data Cleaning and Processing**
- **Handle Missing Values**:
  - Drop rows with missing values.
  - Fill missing values with:
    - Mean, Median, Mode.
    - Custom value.
- **Handle Duplicates**:
  - Remove duplicate rows.
- **Handle Outliers**:
  - Remove outliers using Z-Score or IQR.

### **4. Data Modification**
- Change the data type of any column.
- Rename columns easily.

### **5. Statistical Summary**
- View dataset statistics using the `describe()` function.

### **6. Data Visualization**
- Create interactive visualizations with [Plotly](https://plotly.com/):
  - Histograms.
  - Boxplots.
  - Scatterplots.
  - Correlation Heatmaps.

### **7. Machine Learning Model Building and Evaluation**
- **Split Data**: Split data into training and test sets.
- **Train Models**: Train models like **XGBoost**, **SVM**, and **Random Forest** for classification tasks.
- **Model Evaluation**: Evaluate the models' performance with metrics such as accuracy and classification report.
- **Make Predictions**: Use trained models to make predictions on new data.

### **8. Interactive Data Chat (AI-Powered)**
- Ask questions about your dataset with the "Chat using RAG" feature, powered by a Large Language Model like Ollama. This feature allows users to ask natural language questions about their data and receive AI-generated answers.

### **9. Feature Scaling and Encoding**
- Apply feature scaling techniques such as **MinMax**, **Standard**, and **Robust scaling** to numeric columns.
- Perform **Label Encoding** or **One-Hot Encoding** on categorical variables.

### **10. Download Cleaned Data**
- Download the cleaned dataset after making adjustments.

## **How to Use**
1. Run the app with:
   ```bash
   streamlit run app.py
