try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

# nltk data
import nltk
nltk.download('averaged_perceptron_tagger')

from dotenv import load_dotenv

load_dotenv()

 
import os
import uuid
import shutil
import glob

import chainlit as cl
from icecream import ic
from prompt import PARSING_INSTRUCTIONS
from loguru import logger

import chromadb
from chromadb.config import Settings

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

KNOWLEDGE_DIRECTORY = 'knowledge'
PERSIST_DIRECTORY = os.path.join(KNOWLEDGE_DIRECTORY, 'db', 'chroma')


def get_file_name_from_path(file_path: str) -> str:
    return os.path.basename(file_path).split('.')[0]

def process_file() -> list:
    # Process and save data in the user session
    pdf_files = ic(glob.glob(os.path.join(KNOWLEDGE_DIRECTORY, '*.pdf')))
    md_files = ic(glob.glob(os.path.join(KNOWLEDGE_DIRECTORY, '*.md')))
    
    logger.info(f"Parsing files: {pdf_files}")

    cached = {}
    for kb in pdf_files:
        cached[kb] = get_file_name_from_path(kb) in [get_file_name_from_path(cache) for cache in md_files]    
    
    logger.info(f"Cached: {cached}")

    input_files = [k for k, v in cached.items() if not v]
    logger.info(f"Found new documents: {input_files}") if len(input_files) > 0 else logger.info("No new documents found")
    
    documents = []

    if os.getenv('RESET_CHROMA') != 'False':
        logger.critical("Removing parsing cache.")
        for md_file in md_files:
            os.remove(md_file)

        logger.warning("Initializing parsing")

        # set up parser
        parser = LlamaParse(
            result_type="markdown",
            parsing_instructions=PARSING_INSTRUCTIONS,
            verbose=False,
            show_progress=True,
        )

        # use SimpleDirectoryReader to parse our file
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(input_files=pdf_files, file_extractor=file_extractor).load_data()
        
        for document in documents:
            md_file_path = os.path.join(KNOWLEDGE_DIRECTORY, get_file_name_from_path(document.metadata['file_name']) + '.md')

            with open(md_file_path, encoding='utf-8', mode='a') as md_file:
                md_file.write(document.text + '\n')
        
        logger.success("Cached documents")
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
                        input_files=input_files,
                        file_extractor=file_extractor
                    ).load_data()
        
        for document in documents:
            md_file_path = os.path.join(KNOWLEDGE_DIRECTORY, get_file_name_from_path(document.metadata['file_name']) + '.md')
            with open(md_file_path, encoding='utf-8', mode='a') as md_file:
                md_file.write(document.text + '\n\n')

        # updated_md_files = [os.path.join(KNOWLEDGE_DIRECTORY, f"{get_file_name_from_path(file)}.md") for file in pdf_files]

        # existing_docs = SimpleDirectoryReader(
        #                         input_files=updated_md_files
        #                     ).load_data()

        # documents.extend(existing_docs)
        
        os.environ['INCREMENTAL_DB_UPDATE'] = 'True'
    else:
        logger.success("Cache found.")
        # use SimpleDirectoryReader to parse our file
        # documents = SimpleDirectoryReader(input_files=[os.path.join(KNOWLEDGE_DIRECTORY, f"{get_file_name_from_path(file)}.md") for file in pdf_files]).load_data()
        
    loader = DirectoryLoader(KNOWLEDGE_DIRECTORY, use_multithreading=True, show_progress=True, glob="**/*.md")
    all_documents = loader.load()

    all_docs = []
    for document in all_documents:
        try:
            document.metadata['source'] = get_file_name_from_path(document.metadata['file_name'])
            all_docs.append(document.to_langchain_format())
        except:
            document.metadata['source'] = get_file_name_from_path(document.metadata['source'])
            all_docs.append(document)

    text_splitter = RecursiveCharacterTextSplitter(            
        chunk_size=1024,            
        chunk_overlap=128       
        )
    
    docs = text_splitter.split_documents(all_docs)
    
    logger.success('Documents splitting completed.')

    for i, doc in enumerate(docs):            
        # TODO: issue - inconsistency between cached and non-cached document sources    
        doc.metadata["source"] = f"chunk_{i}::" + doc.metadata["source"]

    if not docs:            
        raise ValueError("PDF file parsing failed.")
    
    return docs

def create_search_engine() -> VectorStore:    
    # Process and save data in the user session
    docs = process_file()
    cl.user_session.set("docs", docs)

    encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    logger.info("Setting up vector store")

    if (not os.path.exists(PERSIST_DIRECTORY)) or (os.getenv('RESET_CHROMA') != 'False') or (os.getenv('INCREMENTAL_DB_UPDATE') != 'False'):
        if os.getenv('RESET_CHROMA') != 'False':
            logger.critical("Resetting Chroma DB.")
            shutil.rmtree(PERSIST_DIRECTORY)
            os.environ['RESET_CHROMA'] = 'False'
        else:
            pass

        # Initialize Chromadb client and settings, reset to ensure we get a clean search engine    
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)  
        client_settings = Settings(        
            allow_reset=True,        
            anonymized_telemetry=False,
            is_persistent=True,
        )    

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
        # Initialize Chromadb client and settings, reset to ensure we get a clean search engine    
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)  
        client_settings = Settings(        
            allow_reset=True,        
            anonymized_telemetry=False,
            is_persistent=True,
        )    
        logger.warning("Using persisted Chroma DB.")

        search_engine = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=encoder,
            client=client,
            client_settings=client_settings,
            )
        
        logger.info("Search engine created from cache.")

    return search_engine
