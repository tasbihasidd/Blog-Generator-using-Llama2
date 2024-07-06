import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function to get response from LLama2 model

def getLlamaResponse(input_text,no_words,blog_style):
    ## Calling llama2 model

    # The C Transformers library provides Python bindings for GGML models.
    
    llm = CTransformers(model='models\llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',config={'max_new_tokens':256,'temperature':0.01})
    
    ## Prompt Template
    template = """
        Write a blog for {blog_style} job profile for topic {input_text} within {no_words} words.
        """
    #creating prompt template
    prompt = PromptTemplate(input_variables=['blog_style','input_text','no_words'],
                            template=template)
    
    #generate the response for llama 2 model
    response = llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)

    return response






st.set_page_config(page_title='Generate Blog',page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')
st.header("Generate Blogs 🤖 ")
input_text = st.text_input("Enter the Blog Topic")

## creating two more columns for additional two fields 
col1,col2 = st.columns((5,5))
with col1:
    no_words = st.text_input("No.of words")
with col2:
    blog_style = st.selectbox("Writing the blog for",('Researchers','Data Scientists','Common People'),index=0) # for whom im writing this blog for

submit = st.button("Generate")

## Final Response
if submit:
    st.write(getLlamaResponse(input_text,no_words,blog_style))