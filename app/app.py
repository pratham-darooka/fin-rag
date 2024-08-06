# next 3 lines for python <3.10
__import__('pysqlite3')
import sys

import nltk.downloader
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#nltk
import nltk
nltk.download('averaged_perceptron_tagger')

import os
import glob
from dotenv import load_dotenv

load_dotenv()

from tempfile import NamedTemporaryFile

import chainlit as cl
from chainlit.types import AskFileResponse
from loguru import logger
import chromadb
from chromadb.config import Settings

from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from prompt import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE, PARSING_INSTRUCTIONS

namespaces = set()
KNOWLEDGE_DIRECTORY = 'knowledge'

def process_file() -> list:
    # Process and save data in the user session
    pdf_files = glob.glob(os.path.join(KNOWLEDGE_DIRECTORY, '*.pdf'))

    # set up parser
    parser = LlamaParse(
        result_type="markdown",  # "markdown" and "text" are available
        parsing_instructions=PARSING_INSTRUCTIONS,
    )

    # use SimpleDirectoryReader to parse our file
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=pdf_files, file_extractor=file_extractor).load_data()
    
    all_docs = []
    for document in documents:
        document.metadata['source'] = document.metadata['file_name']
        all_docs.append(document.to_langchain_format())

    text_splitter = RecursiveCharacterTextSplitter(            
        chunk_size=2048,            
        chunk_overlap=256       
        )
    
    docs = text_splitter.split_documents(all_docs)
    
    for i, doc in enumerate(docs):            
        doc.metadata["source"] = f"chunk_{i}::" + doc.metadata["source"]

    if not docs:            
            raise ValueError("PDF file parsing failed.")
    
    return docs

def create_search_engine() -> VectorStore:    
    # # Process and save data in the user session
    # pdf_files = glob.glob(os.path.join(KNOWLEDGE_DIRECTORY, '*.pdf'))
    # logger.info(pdf_files)
    
    # docs = []

    # for file in pdf_files:
    #     docs.extend(process_file(file))    

    docs = process_file()

    cl.user_session.set("docs", docs)

    encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Initialize Chromadb client and settings, reset to ensure we get a clean search engine    
    client = chromadb.EphemeralClient()    
    client_settings = Settings(        
        allow_reset=True,        
        anonymized_telemetry=False    
        )    
    search_engine = Chroma(        
        client=client,       
        client_settings=client_settings    
        )    
    search_engine._client.reset()
    search_engine = Chroma.from_documents(        
        client=client,        
        documents=docs,        
        embedding=encoder,        
        client_settings=client_settings    
        )
    return search_engine

@cl.on_chat_start
async def start():           
    try:        
        msg = cl.Message(content=f"Hello! Loading documents...")   
        await msg.send()
        search_engine = await cl.make_async(create_search_engine)() 
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()        
        raise SystemError
    
    llm = ChatGoogleGenerativeAI(        
        model='gemini-pro',        
        temperature=0,        
        streaming=True,        
        convert_system_message_to_human=True    
        )
    
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(        
        llm=llm,        
        chain_type="stuff",        
        retriever=search_engine.as_retriever(max_tokens_limit=4097),
        memory=memory,
        chain_type_kwargs={            
                "prompt": PROMPT,            
                "document_prompt": EXAMPLE_PROMPT        
                },    
        )
    msg.content = f"Documents loaded! How can I help you?"  

    await msg.update()
    
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):    
    chain = cl.user_session.get("chain")  
        
    cb = cl.AsyncLangchainCallbackHandler()    
    
    response = await chain.acall(message.content, callbacks=[cb])    
    
    answer = response["answer"]    
    sources = response["sources"].strip()    
    source_elements = []
    
    # Get the documents from the user session   
    docs = cl.user_session.get("docs")    
    metadatas = [doc.metadata for doc in docs]    
    all_sources = [m["source"] for m in metadatas]

    # Adding sources to the answer    
    if sources:        
        found_sources = []
        # Add the sources to the message        
        for source in sources.split(","):            
            source_name = source.strip()        
    
            # Get the index of the source            
            try:               
                index = all_sources.index(source_name)    
            except ValueError:                
                continue            
            text = docs[index].page_content            
            found_sources.append(source_name)            
            
            # Create the text element referenced in the message            
            source_elements.append(cl.Text(content=text, name=source_name))
        if found_sources:            
            answer += f"\nSources: {', '.join(found_sources)}"       
        else:            
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()

if __name__ == "__main__":
    store = create_search_engine()