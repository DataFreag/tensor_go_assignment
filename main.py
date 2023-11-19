#import nessessary packages
import json
import whisper
import warnings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA


#set up warnings
warnings.filterwarnings('ignore')

def load_api_key(json_path: str) -> str:
    '''
    load API key from json file where API_KEY is stored
    Args:
        json_path (str): path file of json
    
    Returns:
        str: The API_KEY to load the q-a llm model
    '''
    with open(json_path) as key_file:
     return json.load(key_file)["API_KEY"]

def transcribe_audio(model: whisper, audio_path: str) -> str:
   '''
   transcribe audio using provided whisper model
   Args:
        model (whisper): the whisper ASR model
        audio_path (str): path to audio file for transcription
    
    Returns:
        str: the full transcription of audio
   '''
   result = model.transcribe(audio_path)
   return result["text"]

def make_embedder() -> HuggingFaceEmbeddings:
    '''
    create a hugging face embedding model

    Returns:
        HuggingFaceEmbeddings: the hugging face embedding model
    '''
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def make_qa_chain(api_key: str, retriever: VectorStoreRetriever) -> RetrievalQA:
    '''
    create a question-answering chain with ChatOpenAI LLM and retriever database
    
    Args:
        api_key (str): the API key for accessing ChatOpenAI model
        retriever (VectorStoreRetriever): the retriever object for document retrieval
    
    Returns:
        RetrievalQA: the question-answering chain
    '''
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key = api_key)
    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True
    )

def ask_question(qa_chain: RetrievalQA, question: str):
    '''
    ask a question using the previous question-answering chain

    Args:
        qa_chain (RetrievalQA): the question-answering chain
        question (str): the question to ask
    '''
    result = qa_chain({"query": question})
    print(f"Q: {result['query'].strip()}")
    print(f"A: {result['result'].strip()}\n")

#Load API key from JSON file
api_key = load_api_key('J:\Project\key.json')       #TODO: specify the JSON file path

#specify audio path
audio_path = r"J:\Project\Sample\lesmis_0002.wav"   #TODO: specify the audio path

#load whisper pretrained model
model = whisper.load_model("tiny")

#transcribe audio and printing
transcription = transcribe_audio(model, audio_path)
print(transcription)

#split transcription into chunks for document generation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.create_documents([transcription])

#create hugging face embedding model
hf = make_embedder()

#create chroma vector to store documents
db = Chroma.from_documents(texts, hf)

#using chroma as retriever vector database
retriever = db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 1})

#create a question-answering chain
qa_chain = make_qa_chain(api_key,retriever)


#asking to summarize the passage and print the result
q = "Summarize the passage.."
ask_question(qa_chain, q)


#asking to translate to a language and print the result
q = "Translate to russian.."    #TODO: specify the target transcription language
ask_question(qa_chain, q)