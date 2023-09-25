import streamlit as st # import the Streamlit library
# Set page config
st.set_page_config(
    page_title="ThoughtTracker",
    page_icon="uic.png",
)
from langchain.chains import LLMChain, SimpleSequentialChain # import LangChain libraries
from langchain.llms import OpenAI # import OpenAI model
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate # import PromptTemplate
from PIL import Image
import numpy as np
import pandas as pd
from time import sleep

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import re

import warnings
warnings.filterwarnings("ignore")





# Define CSS styles
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .highlight {
        color: #FF5E5E;
        font-weight: bold;
    }
    .header-image {
        width: 700px;
        max-width: 100%;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)




st.title("A smart AI tool to detect suicide ideation")
st.markdown("\n")
st.markdown("_The application may experience a slight delay during the initial start-up as it requires loading the models. Your patience is greatly appreciated_")
st.markdown("\n")

image_1 = Image.open('bg-mindwatch_1.png')
st.image(image_1, caption=None, width=500, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.markdown("\n")



# Footer message
st.markdown("---")
st.markdown("**_© University of Illinois at Chicago (UIC) @2023_**")
st.markdown("---")
st.markdown("\n")

# open-ai-key

st.markdown('**Enter the OpenAI-key to continue, refer: https://platform.openai.com/account/api-keys \
            to generate a key if you dont have one..!!**')
openai_key = st.text_input('OpenAI-key', type="password")
openai_key = str(openai_key)



if openai_key:

    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002', openai_api_key = openai_key)


    user_question = st.text_input(
    "Enter the patient/user text: "
    )
    st.markdown("\n")

      


    if st.button("Click here to check answer"):

            
        llm = OpenAI(model_name = 'gpt-3.5-turbo', temperature=0.0, openai_api_key=openai_key) # text-davinci-003
        # Generating the final answer to the user's question using all the chains
        if user_question == ' ':
            st.success('No Input text provided')
        else:
            # Chain 1: Generating a rephrased version of the user's question
            template = """Patient Note/Social-Media Post Analysis: Assessing Mental State and Providing Support only if post or text is suicidal or depressing else not required.
    
            Act as "Senior psychiatrist/mental health specialist" and analyze the following text and determine the mental state of the person. If the individual is showing signs of being "Depressed" or "Normal" or "happy" or "hopeful" or "not much context to state or conclude anything", suggest potential solutions or resources provide assistance. If necessary, provide helpline numbers or suggest prescriptions based on the symptoms analyzed from text for further support.
    
            Please be creative in drafting the solutions, keep them short (strict word limit of 200 words), and informative like prescribing solutions, etc. "Note: Do not suggest solutions if the post is normal/non-suicidal."
    
            {question}
    
            \n\n"""
            prompt_template = PromptTemplate(input_variables=["question"], template=template)
            question_chain = LLMChain(llm=llm, prompt=prompt_template)
    
            overall_chain = SimpleSequentialChain(
                chains=[question_chain]
            )
    
            # Running all the chains on the user's question and displaying the final answer
            st.success(overall_chain.run(user_question))

           