import streamlit as st
import PyPDF2 as pdf
import os 
import glob
import re
import fitz
from transformers import pipeline
A="<<<<< Click Here first before summarize"
Download = st.button("DOWNLOAD EACH MODEL"),st.write(A)
Summarizeit = st.button("CLICK TO SUMMARIZE")
st.write("My pdf has 1 research per pdf || Better summarization and keyword graph)")
Summarizebunch = st.button("CLICK TO SUMMARIZE(multiple research per pdf)")
st.write("My pdf has multiple research per pdf")
uploaded_files = st.file_uploader("Choose a PDF file", type=["pdf"] ,accept_multiple_files=True)

#--- getting pre trained model


from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])


#-------------- 




if Summarizeit == True:
    i = 0  
    num_paper = 0
    alternate = 0
    fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base")
    model_name = "ml6team/keyphrase-extraction-kbir-inspec"
    extractor = KeyphraseExtractionPipeline(model=model_name)
    summarizer = pipeline('summarization')

    for paper in uploaded_files:
        file = paper
        pdf_reader = pdf.PdfFileReader(file) #file reader
        page_num = pdf_reader.getNumPages()
        for i in range(page_num):
            pages =pdf_reader.getPage(i)
            text1 = pages.extractText()
            if "TABLE OF CONTENTS" in text1 or "Table of contents" in text1 :
                st.write("Table of contents at page",i +1 )
                i = i+1
            elif "INTRODUCTION"  in text1:
                text1 = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text1)
                corrected = fix_spelling(text1,max_length=4000)
                correctedintro = corrected[0]['generated_text']
                st.write("Introduction at page",i +1)
                Intro = True
                i = i +1 
                num_paper = num_paper + 1
            elif "ABSTRACT" in text1 :
                text1 = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text1)
                corrected = fix_spelling(text1,max_length=4000)
                correctedabs = corrected[0]['generated_text'] 
                st.write("Abstract at page",i +1)
                i = i + 1
                Abs = True
                if Intro == True:
                    correctedintro = correctedabs + correctedintro
                else:
                    correctedintro = correctedabs 
            elif "CONCLUSION" in text1 :
                text1 = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text1)
                corrected = fix_spelling(text1,max_length=4000)
                correctecon = corrected[0]['generated_text']
                st.write('Conclusion at page',i +1)
                i = i + 1 
                if Abs == True or Intro == True:
                    correctedintro =  correctedintro + correctecon
                else:
                    correctedintro =  correctecon
            else:
                i = i +1 
    summaryfin = summarizer(correctedintro,max_length = 70 , min_length = 30 ,do_sample = False)
    summaryfin = summaryfin[0]['summary_text'] 
    st.write('Path to Pdf File',paper)
    st.write("summary of research:",summaryfin)
    with fitz.open(paper) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
            text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
            keyphrases = extractor(text)
            for word in (keyphrases):
                text.count(word)
                st.write(word,"Appeared",text.count(word),"times")

    st.write('-------------------------------------------------------------------------------------------------------------------------------------------')


