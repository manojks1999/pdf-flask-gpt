from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from flask import Flask
from flask import request, session
from flask_session import Session
from flask_cors import CORS, cross_origin
from datetime import timedelta
import csv

app = Flask(__name__)
SESSION_TYPE='filesystem'
app.config.from_object(__name__)
Session(app)
CORS(app, supports_credentials = True)
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text() + '\n'
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def load_default_csvs(file_name):
    text = 'This is the content of uploaded file '
    file_path = './csv_files/' + file_name + '.csv'
    with open(file_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Read and print each row in the CSV file
        for row in csv_reader:
            text += row[0] + '\n'
    return text

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5})
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = session['conversation']({'question': user_question})
    session.chat_history = response['chat_history']

    total_chat_conversion = []
    for i, message in enumerate(session.chat_history):
        if i % 2 == 0:
            total_chat_conversion.append(message.content)
        else:
            total_chat_conversion.append(message.content)
    return total_chat_conversion


@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

@app.route('/file', methods = ['POST'])
def file_upload():  
    try:
        file = request.files['file'] 
        raw_text = get_pdf_text(file)
        # print("dsdsfsf", data)
        # get the text chunks
        text_chunks = get_text_chunks(raw_text)
        # create vector store
        vectorstore = get_vectorstore(text_chunks)

        # create conversation chain
        session['conversation'] = get_conversation_chain(
            vectorstore)
        return {'message': 'File uploaded successfully'}, 200
    except Exception as error:
        print("Error in file api", error)
        return {
            'message': 'Something went wrong'
        }, 500
    


@app.route('/default_csv', methods = ['GET'])
def csv_read_shell():  
    try:
        default_file = request.args.get('file_name')
        raw_text = load_default_csvs(default_file)
        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # create vector store
        vectorstore = get_vectorstore(text_chunks)

        # create conversation chain
        session['conversation'] = get_conversation_chain(
            vectorstore)
        return {
            'message': 'File uploaded successfully'
            }, 200
    except Exception as error:
        print("error in default csv api", error)
        return {
            'message': 'Something went wrong'
        }, 500
    

@app.route('/ask_question', methods = ['POST'])
def ask_questions():
    try:
        print("session_ids", session.sid)
        if 'conversation' not in session:
            return {
                'message': "please upload file first"
            }, 400
        data = request.get_json()
        
        output = handle_userinput(data.get('question'))
        
        return {
            'answer': output[len(output) - 1]
        }, 200
    except Exception as error:
        print("errrrror in qna api", error)
        return {
            'message': 'Something went wrong'
        }, 500
    

if __name__ == '__main__':
    app.run()