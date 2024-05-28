import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text

def read_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Web app
def main():
    st.set_page_config(page_title="Resume Screening App", layout="centered", initial_sidebar_state="expanded")
    
    st.markdown(
        """
        <style>
        header{display:none !important}
        body h1, h2 ,p{color:#fff !important}
        .main {
            height: 100vh;
            background-color: #4158D0;
            background-image: linear-gradient(43deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%);
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color:#fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.title("Resume Screening App")
    st.header("Upload Your Resume")
    uploaded_file = st.file_uploader('Choose a file', type=['txt', 'pdf'])
    
    if uploaded_file is not None:
        with st.spinner('Processing...'):
            try:
                if uploaded_file.type == "application/pdf":
                    resume_text = read_pdf(uploaded_file)
                else:
                    resume_bytes = uploaded_file.read()
                    resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 decoding fails, try decoding with 'latin-1'
                resume_text = resume_bytes.decode('latin-1')
        
        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.success(f"Predicted Category: {category_name}")

        st.balloons()

# Python main
if __name__ == "__main__":
    main()
