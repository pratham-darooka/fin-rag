# next 3 lines for python <3.10
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
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
from prompt import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE, PARSING_INSTRUCTIONS

namespaces = set()
KNOWLEDGE_DIRECTORY = 'knowledge'

def llama_parse_documents(directory, file):
    logger.info('API_KEY: ' + os.getenv('LLAMA_CLOUD_API_KEY'))
    src = os.path.join(directory, file)
    destination = os.path.join(directory, file.split('.')[0] + '.md')
    logger.info(f'Parsing {src} into {destination} using LlamaParse')
    
    if not os.path.exists(destination):
        with open(destination, "w", encoding='utf-8') as text_file:
            document = LlamaParse(
                    result_type="markdown",
                    parsing_instructions=PARSING_INSTRUCTIONS,
                ).load_data(src)
            parsed_text = "".join([i.text for i in document])
            text_file.write(parsed_text)
            logger.success(f'{destination} created.')

            langchain_doc = [d.to_langchain_format() for d in document]

            return langchain_doc
    else:
        logger.info("Found in cache!")
        return []

def process_file(*, file: AskFileResponse) -> list:   
    if file.type != "application/pdf":        
        raise TypeError("Only PDF files are supported")
    
    try:
        documents = llama_parse_documents(KNOWLEDGE_DIRECTORY, file.path)    
        logger.info(f'LlamaParse has finished')
        if documents == []:
            markdown_file_path = os.path.join(KNOWLEDGE_DIRECTORY, file.path.split('.')[0] + '.md')
            logger.info(markdown_file_path)
            loader = UnstructuredMarkdownLoader(markdown_file_path)
            
            documents = loader.load()
    except Exception as e:
        logger.error(f'!! Error: {e}')
        loader = PDFPlumberLoader(os.path.join(KNOWLEDGE_DIRECTORY, file.path))

        documents = loader.load()
    logger.info(f'Document loaded')

    text_splitter = RecursiveCharacterTextSplitter(            
        chunk_size=2048,            
        chunk_overlap=256       
        )
    
    docs = text_splitter.split_documents(documents)
    logger.info("Document splitting finished")

    try:
        for i, doc in enumerate(docs):            
            doc.metadata["source"] = f"chunk_{i}"
    except:
        pass
    
    if not docs:            
            raise ValueError("PDF file parsing failed.")
    return docs

def create_search_engine(*, file: AskFileResponse) -> VectorStore:    
    # Process and save data in the user session    
    docs = process_file(file=file)    

    logger.info("Creating search engine")
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
    logger.success("Search engine created.")
    return search_engine

@cl.on_chat_start
async def start():   
    # res = await cl.AskUserMessage(content="Enter Gemini API Key:", timeout=10).send()
    # if res:       
    #     await cl.Message(            
    #         content=f"Your Gemini API Key is processed: {res['content']}",       
    #         ).send()
    
    files = None    
    while files is None:        
        files = await cl.AskFileMessage(            
            content=WELCOME_MESSAGE,           
            accept=["application/pdf"],            
            max_size_mb=5,        
            ).send()
    file = files[0]    
    msg = cl.Message(content=f"Processing `{file.name}`...")    
    await msg.send()
    
    try:        
        search_engine = await cl.make_async(create_search_engine)(file=file)    
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
    msg.content = f"`{file.name}` processed. You can now ask questions!"  

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