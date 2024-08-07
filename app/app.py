# next 3 lines for python <3.10
# __import__('pysqlite3')
# import sys
# import uuid
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# nltk data
# import nltk
# nltk.download('averaged_perceptron_tagger')

import os
import glob
from dotenv import load_dotenv

load_dotenv()

from tempfile import NamedTemporaryFile
from typing import List

import chainlit as cl
from chainlit.types import AskFileResponse
from chainlit.input_widget import Select, Switch, Slider
from loguru import logger
import uuid
import chromadb
from chromadb.config import Settings

from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from prompt import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE, PARSING_INSTRUCTIONS, QUERY_PROMPT, CONDENSE_QUESTION_PROMPT

namespaces = set()
message_history = ChatMessageHistory()

KNOWLEDGE_DIRECTORY = 'knowledge'
PERSIST_DIRECTORY = os.path.join(KNOWLEDGE_DIRECTORY, "db")

def process_file() -> list:
    # Process and save data in the user session
    pdf_files = glob.glob(os.path.join(KNOWLEDGE_DIRECTORY, '*.pdf'))
    logger.info(f"Parsing files: {pdf_files}")
    
    if (os.getenv('RESET_CHROMA') != 'FALSE') or (glob.glob(os.path.join(KNOWLEDGE_DIRECTORY, '*.md')) == []):
        logger.warning("Cache not found.")
        # set up parser
        parser = LlamaParse(
            result_type="markdown",
            parsing_instructions=PARSING_INSTRUCTIONS,
        )

        # use SimpleDirectoryReader to parse our file
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(input_files=pdf_files, file_extractor=file_extractor).load_data()

        for document in documents:
            md_file_path = os.path.join(KNOWLEDGE_DIRECTORY, document.metadata['file_name'].split('.')[0] + '.md')
            with open(md_file_path, encoding='utf-8', mode='a') as md_file:
                md_file.write(document.text + '\n\n')
            logger.success(f"Parsed {document.metadata['file_name']} into {md_file_path}")

    else:
        # use SimpleDirectoryReader to parse our file
        documents = SimpleDirectoryReader(input_files=[f"{file.split('.')[0]}.md" for file in pdf_files]).load_data()
        
        logger.success("Cache found. Documents loaded.")
    
    all_docs = []
    for document in documents:
        document.metadata['source'] = document.metadata['file_name'].split('.')[0]
        all_docs.append(document.to_langchain_format())

    text_splitter = RecursiveCharacterTextSplitter(            
        chunk_size=2048,            
        chunk_overlap=256       
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
    # Initialize Chromadb client and settings, reset to ensure we get a clean search engine    
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)  
    client_settings = Settings(        
        allow_reset=True,        
        anonymized_telemetry=False,
        is_persistent=True,
    )    

    if (not os.path.exists(PERSIST_DIRECTORY)) or (os.getenv('RESET_CHROMA') != 'FALSE'):        
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
    else:
        search_engine = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=encoder,
            client=client,
            client_settings=client_settings,
            )
    logger.success("Vector store set up successfully")
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
                initial=0,
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    ).send()

    try:        
        msg = cl.Message(content=f"Hello! Loading documents...")   
        await msg.send()
        logger.info("Creating search engine")
        search_engine = await cl.make_async(create_search_engine)() 
        logger.success("Search engine created")
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()        
        raise SystemError
    
    if settings['Model'] == 'Gemini':
        logger.info(f"Using {settings['Model']} with temperature = {settings['Temperature']}.")
        llm = ChatGoogleGenerativeAI(        
            model='gemini-pro',        
            temperature=settings['Temperature'],        
            streaming=settings['Streaming'],        
            convert_system_message_to_human=True    
            )
    else:
        logger.info(f"Using {settings['Model']} with temperature = {settings['Temperature']}.")
        llm = ChatGroq(        
            model="mixtral-8x7b-32768",        
            temperature=settings['Temperature'],        
            streaming=settings['Streaming'],        
            )
            
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    # logger.critical(QUERY_PROMPT.format_prompt(chat_history=["meow"], question="{{question}}"))

    retriever = search_engine.as_retriever(
                max_tokens_limit=4097,
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.5, 
                    "k": 5
                    },
            )

    # Define the chain components
    # _inputs = RunnableParallel(
    #     standalone_question=RunnablePassthrough.assign(
    #         chat_history=lambda x: message_history.messages
    #     )
    #     | CONDENSE_QUESTION_PROMPT
    #     | llm
    #     | StrOutputParser(),
    # )

    # logger.critical(_inputs.invoke("question"))

    retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=retriever, 
            # prompt=QUERY_PROMPT.format(chat_history=message_history.messages, question="{question}"),
            llm=llm,
            include_original=True
        )

    chain = RetrievalQAWithSourcesChain.from_chain_type(        
        llm=llm,
        chain_type="stuff",        
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
    cl.user_session.set("settings", settings)
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
            source_name = source.strip().split('.')[0]             

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
    # logger.success(cl.chat_context.to_openai())

    ans = cl.Message(content="", elements=source_elements)
    for token in answer:
        await ans.stream_token(token)

    await ans.send()    

# @cl.set_starters
# async def set_starters():
#     return [
#         cl.Starter(
#             label="Microsoft's Operating Margin",
#             message="What is Microsoft's Operating Margin in 2023?",
#             ),
#         cl.Starter(
#             label="Apple's Net iPhone Sales",
#             message="What is the latest net sales for Apple's iphones?",
#             ),
#         cl.Starter(
#             label="Uber's CEO",
#             message="Who is Uber's CEO?",
#             ),
#         ]

if __name__ == "__main__":
    # do process files, change env var reset chroma, do process files again and see difference in document parsing
    pass