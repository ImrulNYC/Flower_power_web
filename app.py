import os
import pandas as pd
import requests
import streamlit as st
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Function to load the dataset 
@st.cache_data
def load_dataset_from_local():
    dataset_path = os.path.join(os.path.dirname(__file__), "data/language-of-flowers.csv")
    try:
        data = pd.read_csv(dataset_path, quotechar='"', encoding='utf-8-sig', on_bad_lines='skip')
        data.columns = data.columns.str.strip()  
        return data
    except FileNotFoundError:
        st.error("Dataset file not found.")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error during dataset parsing: {e}")
        return None

#generating flower information
def generate_flower_info(flower_name, flower_info_dict, gpt2_pipeline):
    flower_description = flower_info_dict.get(flower_name, "No description available.")
    query = f"Why is the flower {flower_name} associated with the meaning '{flower_description}'? Explain the cultural or historical significance behind the {flower_name}."
    generated_info = gpt2_pipeline(query, max_length=200, truncation=True)[0]["generated_text"]
    sentences = re.split(r'(?<=\w[.!?])\s+', generated_info.strip())
    limited_output = "".join(sentences[:5])
    return flower_name, flower_description, limited_output

# Function to load flower image
def load_flower_image(flower_name):
    base_url = "https://raw.githubusercontent.com/ImrulNYC/flower-power/main/data/Flower_images/"
    words = flower_name.split()
    if len(words) == 1:
        
        formatted_name = words[0].capitalize()
    else:
        
        formatted_name = f"{words[1].capitalize()}_{words[0].lower()}"
    image_url = f"{base_url}{formatted_name}.jpg"
    
    try:
        response = requests.head(image_url)
        if response.status_code == 200:
            return image_url
    except requests.RequestException:
        return None

    return None

# Main app code
def developer_info():
    st.markdown(
        """
        <style>
            .dev-title {
                font-size: 2.5em;
                color: #4CAF50;
                text-align: center;
                font-weight: bold;
                margin-bottom: 0.5em;
            }
            .dev-content {
                font-size: 1.2em;
                line-height: 1.6em;
                color: #00695c;
                text-align: center;
                margin-bottom: 2em;
            }
        </style>
        <div class='dev-title'>Developer Information</div>
        <div class='dev-content'>
            <ul>
                <li><strong>Jessica </strong>: <a href='https://www.linkedin.com/in/jessica-lau-/' target='_blank'>LinkedIn Profile</a></li>
                <li><strong>Mansur </strong>: <a href='https://www.linkedin.com/in/mansur-mahdee-880204231' target='_blank'>LinkedIn Profile</a></li>
                <li><strong>Imrul </strong>: <a href='https://www.linkedin.com/in/imrul-nyc/' target='_blank'>LinkedIn Profile</a></li>
                <li><strong>Zahava </strong>: <a href='https://www.linkedin.com/in/zahava-lowy-b294b023a/' target='_blank'>LinkedIn Profile</a></li>
                 <li><strong>Github Link  </strong>: <a href='https://github.com/ImrulNYC/4900_Final' target='_blank'> Github</a></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
#main app 
def streamlit_app():
    
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 3em;
                color: #4CAF50;
                text-align: center;
                font-weight: bold;
                margin-bottom: 0.5em;
            }
            .sub-title {
                font-size: 1.5em;
                color: #666;
                text-align: center;
                margin-bottom: 2em;
            }
            .info-box {
                background-color: #e0f7fa;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #00acc1;
                box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);
                margin-bottom: 1.5em;
            }
            .info-title {
                font-size: 1.7em;
                color: #004d40;
                margin-bottom: 0.5em;
            }
            .info-content {
                font-size: 1.2em;
                line-height: 1.6em;
                color: #00695c;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='main-title'>Flower Power </div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Welcome to the Language of Flowers, Flower Recognition App.</div>", unsafe_allow_html=True)

    # Two model options for flower recognition
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <a href="https://flower-rec1-gupv5c67pmw57q4nlqea4o.streamlit.app" target="_blank">
                <button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 5px; cursor: pointer; text-decoration: underline;">
                    Pre-trained Flower Recognition
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <a href="https://flower-rec2-km4jkbdmfrnn3zztrvyzuw.streamlit.app" target="_blank">
                <button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 5px; cursor: pointer; text-decoration: underline;">
                    Flower Recognition from Scratch
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )

    # Load dataset of flower language
    data = load_dataset_from_local()
    if data is not None:
        # Creating a combined key with color and flower name
        data['Flower'] = data['Color'].fillna('') + ' ' + data['Flower']
        flower_info_dict = dict(zip(data['Flower'].str.strip().str.lower(), data['Meaning']))
        meaning_info_dict = dict(zip(data['Meaning'].str.strip().str.lower(), data['Flower']))

        # Initialize GPT-2 for text generation
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_pipeline = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer)

        # User input to get flower power
        flower_names = list(flower_info_dict.keys())
        flower_name = st.selectbox("Enter a flower name (e.g., 'Red Rose'):", options=["None"] + sorted(flower_names), index=0, key='flower').strip().lower()

        # Display information
        if flower_name != "none":
            if flower_name in flower_info_dict:
                flower_name, flower_description, generated_info = generate_flower_info(flower_name, flower_info_dict, gpt2_pipeline)
                image_path = load_flower_image(flower_name)
                if image_path:
                    st.image(image_path, caption=flower_name.title(), use_container_width=True)
                st.markdown(
                    f"<div class='info-box' style='background-color: #fce4ec; border-color: #f06292;'>"
                    f"<div class='info-title'>Information for {flower_name.title()}:</div>"
                    f"<div class='info-content'><strong>Meaning</strong>: {flower_description}</div>"
                    f"<div class='info-content'><strong>Cultural or Historical Significance</strong>: {generated_info}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"<div class='info-box' style='background-color: #ffccbc; border-color: #ff7043;'>Sorry, we don't have information on the flower: {flower_name.title()}</div>", unsafe_allow_html=True)

        # User input to get meaning
        meanings = list(meaning_info_dict.keys())
        meaning = st.selectbox("Enter a meaning to find the flower:", options=["None"] + sorted(meanings), index=0, key='meaning').strip().lower()

        # Display information 
        if meaning != "none":
            if meaning in meaning_info_dict:
                matching_flower = meaning_info_dict[meaning]
                image_path = load_flower_image(matching_flower)
                if image_path:
                    st.image(image_path, caption=matching_flower.title(), use_container_width=True)
                st.markdown(
                    f"<div class='info-box' style='background-color: #e8f5e9; border-color: #66bb6a;'>"
                    f"<div class='info-title'>Flower associated with '{meaning.title()}':</div>"
                    f"<div class='info-content'><strong>Flower</strong>: {matching_flower.title()}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"<div class='info-box' style='background-color: #ffccbc; border-color: #ff7043;'>Sorry, we don't have a flower associated with the meaning: {meaning.title()}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box' style='background-color: #ffccbc; border-color: #ff7043;'>There was an issue with loading the dataset.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the page:", ["Flower Information", "Developer Info"])
    if app_mode == "Flower Information":
        streamlit_app()
    elif app_mode == "Developer Info":
        developer_info()
