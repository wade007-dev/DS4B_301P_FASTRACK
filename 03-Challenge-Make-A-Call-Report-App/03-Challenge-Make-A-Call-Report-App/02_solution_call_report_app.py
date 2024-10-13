# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# CHALLENGE #1 - CREATE A CALL REPORT APP
# ***
# GOAL: Exposure to using LLM's, Document Loaders, and Prompts

# streamlit run 03-Challenge-Make-A-Call-Report/02_solution_call_report_app.py

import yaml

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import streamlit as st
import os
from tempfile import NamedTemporaryFile

# Load API Key
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']

# 1.0 LOAD AND SUMMARIZE FUNCTION
def load_and_summarize(file):
    
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        file_path = tmp.name
    
    try:
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        model = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        prompt_template = prompt_template = """
        Write a business report from the following earnings call transcript:
        {text}

        Use the following Markdown format:
        # Insert Descriptive Report Title

        ## Earnings Call Summary
        Use 3 to 7 numbered bullet points

        ## Important Financials:
        Describe the most important financials discussed during the call. Use 3 to 5 numbered bullet points.

        ## Key Business Risks
        Describe any key business risks discussed on the call. Use 3 to 5 numbered bullets.

        ## Conclusions
        Conclude with any overaching business actions that the company is pursuing that may have a positive or negative implications and what those implications are. 
        """
        
        prompt = PromptTemplate(input_variables = ["text"], template=prompt_template)


        llm_chain = LLMChain(llm=model, prompt=prompt)
        
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        response = stuff_chain.invoke(docs)
        
    finally:
        
        os.remove(file_path)

    return response['output_text']

# Streamlit Interface
st.title('PDF Earnings Call Summarizer')

st.subheader('Upload a PDF document:')
uploaded_file = st.file_uploader("Choose a file", type="pdf")

if uploaded_file is not None:
    
    if st.button('Summarize Document'):
        with st.spinner('Summarizing...'):
            
            summary = load_and_summarize(uploaded_file)
            
            st.subheader('Summarization Result:')
            st.markdown(summary)
            
else:
    st.write("No file uploaded. Please upload a PDF file to proceed.")
    
    