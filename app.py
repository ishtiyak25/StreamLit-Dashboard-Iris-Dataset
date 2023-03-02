import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Define a dictionary to map dataset names to actual datasets
DATASETS = {
    'iris': datasets.load_iris(),
    'wine': datasets.load_wine(),
    'breast_cancer': datasets.load_breast_cancer()
}

# Define a function to display a heatmap
def show_heatmap(data):
    st.write("## Heatmap")
    sns.heatmap(data.corr(), annot=True, cmap="YlGnBu")
    st.pyplot()

# Define a function to display a histogram
def show_histogram(data, column):
    st.write("## Histogram")
    sns.histplot(data[column])
    st.pyplot()

# Define a function to display a bar plot
def show_bar_plot(data, x_column, y_column):
    st.write("## Bar Plot")
    sns.barplot(x=x_column, y=y_column, data=data)
    st.pyplot()

# Define a function to display a scatter plot
def show_scatter_plot(data, x_column, y_column):
    st.write("## Scatter Plot")
    sns.scatterplot(x=x_column, y=y_column, data=data)
    st.pyplot()

# Define the main function to run the Streamlit app
def main():
    st.sidebar.title("Select Options")
    plot_type = st.sidebar.selectbox("Select a plot type", ["Heatmap", "Histogram", "Bar Plot", "Scatter Plot"])

    # Allow the user to select a dataset
    dataset_name = st.sidebar.selectbox("Select a dataset", list(DATASETS.keys()))

    # Load the selected dataset
    data = pd.DataFrame(DATASETS[dataset_name].data, columns=DATASETS[dataset_name].feature_names)

    # Show some basic information about the dataset
    st.write("### Dataset Information")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")
    st.write("### Sample Data")
    st.write(data.head())

    # Allow the user to select a column for the histogram and scatter plot
    column_options = list(data.columns)
    x_column = st.sidebar.selectbox("Select a column for the X axis", column_options)
    y_column = st.sidebar.selectbox("Select a column for the Y axis", column_options)

    # Allow the user to filter the data
    filter_column = st.sidebar.selectbox("Select a column to filter by", column_options)
    filter_value = st.sidebar.text_input("Enter a value to filter by")
    if filter_value:
        data = data[data[filter_column] == float(filter_value)]

    # Display the selected plot type
    if plot_type == "Heatmap":
        show_heatmap(data)
    elif plot_type == "Histogram":
        show_histogram(data, x_column)
    elif plot_type == "Bar Plot":
        show_bar_plot(data, x_column, y_column)
    elif plot_type == "Scatter Plot":
        show_scatter_plot(data, x_column, y_column)

if __name__ == '__main__':
    main()
