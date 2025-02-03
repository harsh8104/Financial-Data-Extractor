import streamlit as st
import getpass
import os
import pytesseract
import base64
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pypdfium2 as pdfium
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PIL import Image
from io import BytesIO
from langchain_groq import ChatGroq

# Configure Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

POPPLER_PATH = r'C:\Program Files\poppler-24.08.0\Library\bin'

load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Define emojis/icons for user and document assistant
user_icon = "👤"  # Emoji for user
doc_icon = "📄"   # Emoji for document assistant

def show_pdf(file_path):
    try:
        # Read file as bytes
        bytes_data = file_path.getvalue()
        
        # Encode to base64
        base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
        
        # Embed PDF viewer
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        
        # Display in Streamlit
        st.markdown(pdf_display, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def convert_pdf_to_images(file_path, scale=300/72):
    pdf_file = pdfium.PdfDocument(file_path)  
    page_indices = [i for i in range(len(pdf_file))]
    
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices, 
        scale=scale,
    )
    
    list_final_images = [] 
    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append(dict({i: image_byte_array}))
    
    return list_final_images

def convert_images_to_text(images):
    extracted_text = ""
    for img_dict in images:
        for _, img_bytes in img_dict.items():
            img = Image.open(BytesIO(img_bytes))
            # Detect orientation and rotate if necessary
            orientation = pytesseract.image_to_osd(img, output_type='dict')['orientation']  
            if orientation in [90, 180, 270]:
                img = img.rotate(int(orientation), expand=True)
            text = pytesseract.image_to_string(img)
            extracted_text += text + "\n\n"
    return extracted_text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=256,
        separator="\n",
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversion_chain(vectorstore):
    chat = ChatGroq(temperature=0.7, groq_api_key="your_api_key", model_name="llama-3.3-70b-versatile")

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False
    )
    
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"{user_icon} **User:** {message.content}")
        else:
            st.markdown(f"{doc_icon} **DocuChat:** {message.content}")

def display_pdf(pdf_docs):
    # Create tabs for each PDF
    if pdf_docs:
        pdf_tabs = st.tabs([f"PDF {i+1}: {pdf.name}" for i, pdf in enumerate(pdf_docs)])
        
        for i, tab in enumerate(pdf_tabs):
            with tab:
                pdf_display = pdf_docs[i]
                
                # Show PDF file details
                st.write(f"Filename: {pdf_display.name}")
                st.write(f"File size: {pdf_display.size/1024:.2f} KB")
                
                # Display the PDF using base64
                show_pdf(pdf_display)

def main():
    load_dotenv()
    st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    # Create two columns: one for chat and one for PDF preview
    chat_col, preview_col = st.columns([2, 1])
    
    with chat_col:
        st.header("DocuChat - Assistant")
        
        user_question = st.text_input("Ask questions about your documents:")
        submit_button = st.button("Submit")
        
        if submit_button and user_question:
            handle_user_input(user_question)
    
    with preview_col:
        st.subheader("Document Preview")
        pdf_docs = st.file_uploader(
            "Upload PDF/DOCX files and click Process",
            accept_multiple_files=True,
            type=['pdf']  # Added type restriction
        )
        
        # Display PDF preview
        if pdf_docs:
            display_pdf(pdf_docs)
    
        if st.button("Process"):
            with st.spinner("Processing documents..."):
                if pdf_docs:
                    # Check for 0MB files
                    invalid_files = [pdf.name for pdf in pdf_docs if pdf.size == 0]
                    if invalid_files:
                        st.error(f"The following files are empty (0MB) and cannot be processed: {', '.join(invalid_files)}")
                        return  # Stop processing if any file is 0MB

                    images = []
                    raw_text = ""

                    for pdf in pdf_docs:
                        pdf_bytes = pdf.read()  # Read the uploaded file as bytes
                        pdf_io = BytesIO(pdf_bytes)  # Convert bytes to file-like object
                        
                        # Convert PDF to images
                        images.extend(convert_pdf_to_images(pdf_io))  

                        # Reset file pointer for text extraction
                        pdf.seek(0)
                        # Extract text from PDF
                        raw_text += get_pdf_text([pdf])
                    
                    # Extract text from images
                    image_text = convert_images_to_text(images)
                    
                    # Combine extracted text
                    final_text = raw_text + "\n" + image_text

                    # Split text into chunks
                    text_chunks = get_chunks(final_text)

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Store the conversation chain in session state
                    st.session_state.conversation = get_conversion_chain(vectorstore)

if __name__ == "__main__":
    main()