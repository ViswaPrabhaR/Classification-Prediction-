import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics.pairwise import cosine_similarity

def app():
    st.title('Product Recommendation System')
    # reading csv file
    data = pd.read_csv('data/Product_Recommend/product.csv')
    mylist = data['ItemDescription'].tolist()
    make_choice = st.selectbox('Please Choose Here ', mylist,
   placeholder="Select Your Product...",)
    st.write(f'Product Recommend for :red[***{make_choice}***]')
    # reading xlsx file
    df = pd.read_excel("data/Product_Recommend/grocery_dataset.xlsx")
    customer_item_matrix = df.pivot_table(index='CustomerID', columns='ItemDescription', values='Quantity',
                                          aggfunc='sum')
    customer_item_matrix =customer_item_matrix.applymap(lambda x: 1 if x>0 else 0)
    item_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))

    item_matrix.columns = customer_item_matrix.T.index
    item_matrix['ItemDescription'] = customer_item_matrix.T.index
    item_matrix = item_matrix.set_index('ItemDescription')
    top_5 = list(item_matrix.loc[make_choice].sort_values(ascending=False).iloc[:6].index)
    defr = pd.DataFrame(top_5, columns =['Product'])
    new = defr[1:]
    st.write(new)