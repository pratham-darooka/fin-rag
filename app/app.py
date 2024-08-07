# next 3 lines for python <3.10
__import__('pysqlite3')
import sys
import uuid
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#nltk data
import nltk
nltk.download('averaged_perceptron_tagger')

import os
import glob
from dotenv import load_dotenv

load_dotenv()

# from tempfile import NamedTemporaryFile

import chainlit as cl
from chainlit.types import AskFileResponse
from chainlit.input_widget import Select, Switch, Slider
from loguru import logger
import chromadb
from chromadb.config import Settings

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
# from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from prompt import EXAMPLE_PROMPT, PROMPT, PARSING_INSTRUCTIONS

namespaces = set()
message_history = ChatMessageHistory()

KNOWLEDGE_DIRECTORY = 'knowledge'
PERSIST_DIRECTORY = os.path.join(KNOWLEDGE_DIRECTORY, "db")

def get_file_name_from_path(file_path: str) -> str:
    return os.path.basename(file_path).split('.')[0]

def process_file() -> list:
    # Process and save data in the user session
    pdf_files = glob.glob(os.path.join(KNOWLEDGE_DIRECTORY, '*.pdf'))
    md_files = glob.glob(os.path.join(KNOWLEDGE_DIRECTORY, '*.md'))
    
    cached = {}
    for kb in pdf_files:
        cached[kb] = get_file_name_from_path(kb) in [get_file_name_from_path(cache) for cache in md_files]    
    
    logger.info(cached)

    input_files = [k for k, v in cached.items() if not v]
    logger.info(f"Found new documents: {input_files}") if len(input_files) > 0 else logger.info("No new documents found")

    if os.getenv('RESET_CHROMA') != 'False':
        for md_file in md_files:
            os.remove(md_file)

        logger.warning("Initializing DB")

        # set up parser
        parser = LlamaParse(
            result_type="markdown",
            parsing_instructions=PARSING_INSTRUCTIONS,
        )

        # use SimpleDirectoryReader to parse our file
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(input_files=pdf_files, file_extractor=file_extractor).load_data()

        for document in documents:
            md_file_path = os.path.join(KNOWLEDGE_DIRECTORY, get_file_name_from_path(document.metadata['file_name']) + '.md')

            with open(md_file_path, 'a') as md_file:
                md_file.write(document.text + '\n\n')
    elif len(input_files) > 0:
        logger.warning("Running incremental parsing for new documents.")
        # set up parser
        parser = LlamaParse(
            result_type="markdown",
            parsing_instructions=PARSING_INSTRUCTIONS,
        )

        # use SimpleDirectoryReader to parse our file
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            input_files=[os.path.join(KNOWLEDGE_DIRECTORY, f"{get_file_name_from_path(file)}.md") for file in pdf_files]
            ).load_data().extend(
                SimpleDirectoryReader(
                    input_files=input_files,
                    file_extractor=file_extractor
                    ).load_data()
                    )

        for document in documents:
            md_file_path = os.path.join(KNOWLEDGE_DIRECTORY, get_file_name_from_path(document.metadata['file_name']) + '.md')
            with open(md_file_path, 'a') as md_file:
                md_file.write(document.text + '\n\n')
        
        os.environ['INCREMENTAL_DB_UPDATE'] = True
    else:
        logger.success("Cache found.")
        # use SimpleDirectoryReader to parse our file
        documents = SimpleDirectoryReader(input_files=[os.path.join(KNOWLEDGE_DIRECTORY, f"{get_file_name_from_path(file)}.md") for file in pdf_files]).load_data()
    
    all_docs = []
    for document in documents:
        document.metadata['source'] = get_file_name_from_path(document.metadata['file_name'])
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
    # Process and save data in the user session
    docs = process_file()

    logger.info("Creating search engine.")
    cl.user_session.set("docs", docs)

    encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Initialize Chromadb client and settings, reset to ensure we get a clean search engine    
    # client = chromadb.EphemeralClient()    
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)  
    client_settings = Settings(        
        allow_reset=True,        
        anonymized_telemetry=False,
        is_persistent=True,
    )    

    if (not os.path.exists(PERSIST_DIRECTORY)) or (os.getenv('RESET_CHROMA') != 'False') or os.environ['INCREMENTAL_DB_UPDATE']:
        logger.warning("Resetting Chroma DB.")

        # Create a list of unique ids for each document based on the content
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]
        unique_ids = list(set(ids))

        # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
        seen_ids = set()
        unique_docs = [doc for doc, id in zip(docs, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

        # Add the unique documents to the database
        search_engine = Chroma.from_documents(        
            client=client,        
            documents=unique_docs,        
            embedding=encoder,        
            client_settings=client_settings,
            ids=unique_ids,
            )
        
        logger.info("Search engine created.")
    else:
        logger.warning("Using persisted Chroma DB.")
        search_engine = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=encoder,
            client=client,
            client_settings=client_settings,
            )
        
        logger.info("Search engine created from cache.")

    return search_engine

@cl.on_chat_start
async def start():           
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="LLM Provider",
                values=["Gemini", "Groq", "OpenAI", "Anthropic"],
                initial_index=0,
            ),
            Switch(
                id="Streaming", 
                label="Stream Tokens", 
                initial=True
                ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    ).send()

    try:        
        msg = cl.Message(content=f"Hello! Loading documents...")   
        await msg.send()
        search_engine = await cl.make_async(create_search_engine)() 
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()        
        raise SystemError
    
    if settings['Model'] == 'Gemini':
        llm = ChatGoogleGenerativeAI(        
            model='gemini-pro',        
            temperature=settings['Temperature'],        
            streaming=settings['Streaming'],        
            convert_system_message_to_human=True    
            )
    else:
        llm = ChatGoogleGenerativeAI(        
            model='gemini-pro',        
            temperature=settings['Temperature'],        
            streaming=settings['Streaming'],        
            convert_system_message_to_human=True    
            )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=search_engine.as_retriever(
                # max_tokens_limit=4097,
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.5, 
                    "k": 5
                    },
            ), llm=llm
        )

    chain = RetrievalQAWithSourcesChain.from_chain_type(        
        llm=llm,        
        chain_type="stuff",        
        # retriever=search_engine.as_retriever(
        #     max_tokens_limit=4097,
        #     search_type="similarity_score_threshold",
        #     search_kwargs={"k": 5},
        #     ),
        retriever=retriever_from_llm,
        memory=memory,
        chain_type_kwargs={            
                "prompt": PROMPT,            
                "document_prompt": EXAMPLE_PROMPT
                },    
        )
    msg.content = f"Documents loaded! How can I help you?"  

    await msg.update()
    
    cl.user_session.set("chain", chain)

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    logger.critical(settings)

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
            source_name = get_file_name_from_path(source.strip())

            # Get the index of the source            
            try:             
                index = all_sources.index(source_name)    
            except ValueError:     
                logger.error("Source not found")     
                # logger.info(source_name)      
                continue            

            text = docs[index].page_content            
            found_sources.append(source_name)            
            
            # Create the text element referenced in the message            
            source_elements.append(cl.Text(content=text, name=source_name))
        if found_sources:            
            answer += f"\nSources: {', '.join(found_sources)}"       
        else:
            pass            
            # answer += "\nError finding sources."

    logger.success(message_history.messages)

    ans = cl.Message(content="", elements=source_elements)
    for token in answer:
        await ans.stream_token(token)

    await ans.send()    
    # await cl.Message(content=answer, elements=source_elements).send()

if __name__ == "__main__":
    # do process files, change env var reset chroma, do process files again and see difference in document parsing
    process_file()