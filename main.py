import streamlit as st
import langchain_helper
st.title("Restaurant Name Generator")

cuisine= st.sidebar.selectbox("Pick a Cuisine", ("German","Italian","Mexican","Pakistani","Indian"))

if cuisine:
    response= langchain_helper.generate_restaurant_name_and_items(cuisine)

    st.header(response['restaurant_name'].strip())
    menu_items = response['food_items'].strip().split(",")

    for item in menu_items:
        st.write("-",item)

