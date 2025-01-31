import streamlit as st
import os
import pytesseract
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pypdfium2 as pdfium
from PIL import Image
from io import BytesIO
from langchain_groq import ChatGroq
import tempfile

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

# Function to convert PDF to images
def convert_pdf_to_images(file_path, scale=300/72):
    try:
        # Load PDF document
        pdf_file = pdfium.PdfDocument(file_path)
        page_indices = [i for i in range(len(pdf_file))]
        
        # Render PDF pages as images
        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices,
            scale=scale,
        )
        
        # Store images in a list
        list_final_images = []
        for i, image in zip(page_indices, renderer):
            image_byte_array = BytesIO()
            image.save(image_byte_array, format='jpeg', optimize=True)
            image_byte_array = image_byte_array.getvalue()
            list_final_images.append({i: image_byte_array})
        
        return list_final_images
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

# Function to extract text from images
def convert_images_to_text(images):
    extracted_text = ""
    for img_dict in images:
        for _, img_bytes in img_dict.items():
            img = Image.open(BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
            extracted_text += text + "\n\n"
    return extracted_text

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=256,
        separator="\n",
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vectorstore(text_chunks):
    if not text_chunks:
        st.warning("No text was extracted from the documents. Please upload valid files.")
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversion_chain(vectorstore):
    if vectorstore is None:
        return None
    chat = ChatGroq(
        temperature=0.7,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
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

# Function to handle user input
def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("No documents have been processed yet. Please upload and process documents first.")
        return
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"User: {message.content}")
        else:
            st.write(f"DocuChat: {message.content}")

# Main function
def main():
    load_dotenv()
    st.set_page_config(page_title="DocuChat", page_icon="📄")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("DocuChat - Open Source Document Assistant")
    
    # User input for questions
    user_question = st.text_input("Ask questions about your documents:")
    if user_question:
        handle_user_input(user_question)
        
    # Sidebar for document upload
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF/DOCX files and click Process",
            accept_multiple_files=True
        )
    
        if st.button("Process"):
            with st.spinner("Processing documents..."):
                if pdf_docs:
                    images = []
                    raw_text = ""

                    for pdf in pdf_docs:
                        # Check if the file size is 0 MB
                        if pdf.size == 0:
                            st.warning(f"Warning: The file '{pdf.name}' is 0 MB and will be skipped.")
                            continue

                        # Save the uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(pdf.read())
                            tmp_file_path = tmp_file.name

                        try:
                            # Convert PDF to images
                            images.extend(convert_pdf_to_images(tmp_file_path))

                            # Extract text from PDF
                            with open(tmp_file_path, "rb") as f:
                                raw_text += get_pdf_text([f])
                        except Exception as e:
                            st.error(f"Error processing file '{pdf.name}': {e}")
                        finally:
                            # Clean up the temporary file
                            os.remove(tmp_file_path)
                    
                    # Extract text from images
                    image_text = convert_images_to_text(images)
                    
                    # Combine extracted text
                    final_text = raw_text + "\n" + image_text

                    # Split text into chunks
                    text_chunks = get_chunks(final_text)

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Store the conversation chain in session state
                    if vectorstore is not None:
                        st.session_state.conversation = get_conversion_chain(vectorstore)
                    else:
                        st.session_state.conversation = None

# Run the app
if __name__ == "__main__":
    main()