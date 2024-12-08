import streamlit as st

from app import load_labels

# let's print the values of this json (key then value)
# Load model and labels
with st.spinner("Loading labels..."):
    labels = load_labels()


# sidebar option checkbox to show the whole list
st.sidebar.header("Options")
show_all = st.sidebar.checkbox("Show all labels", value=True)
font_size = st.sidebar.slider("Font Size", 1.0, 2.0, 1.5, step=0.1)


# Display the labels
st.header("ImageNet Labels")
st.write("The ImageNet dataset consists of 1000 classes.")
st.write("This page allows you to search for a label or view all labels.")

# let's allow a search for the labels
search = st.text_input("Search for a label")
if search:
    # remove trailing and leading white space
    search = search.strip()

    search_results = {
        key: value for key, value in labels.items() if search.lower() in value.lower()
    }
    if search_results:
        st.write("Search Results:")
        for key, value in search_results.items():

            st.markdown(
                f"<span style='color:green;font-weight:bold;font-size:{font_size}em'>{key}:</span>\t <span style='color:red;font-size:{font_size}em'>{value}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.write("No results found.")


if show_all:
    st.header("All Labels")
    for key, value in labels.items():
        st.markdown(
            f"<span style='color:green;font-weight:bold;font-size:{font_size}em'>{key}:</span>\t <span style='color:red;font-size:{font_size}em'>{value}</span>",
            unsafe_allow_html=True,
        )
