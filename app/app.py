import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

from loguru import logger
from rag_init import create_search_engine, get_file_name_from_path

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

from prompt import EXAMPLE_PROMPT, PROMPT

namespaces = set()
message_history = ChatMessageHistory()
cl.user_session.set('search_engine_created', False)

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
                max=1,
                step=0.1,
            ),
        ]
    ).send()

    if cl.user_session.get('search_engine_created'):
        try:        
            logger.info("Creating search engine")
            search_engine = await cl.make_async(create_search_engine)() 
            cl.user_session.set("search_engine", search_engine)

        except Exception as e:
            await cl.Message(content=f"Error: {e}").send()        
            raise SystemError

    logger.success("### Chatbot ready!")
    
    await setup_agent(settings)  
    
@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    logger.critical(f"Config: {settings}")

    if settings['Model'] == 'Gemini':
        llm = ChatGoogleGenerativeAI(        
            model='gemini-pro',        
            temperature=settings['Temperature'],        
            streaming=settings['Streaming'],        
            convert_system_message_to_human=True    
            )
    else:
        llm = ChatGroq(        
            model="llama3-8b-8192",        
            temperature=settings['Temperature'],        
            streaming=settings['Streaming'],        
            )
    
    logger.info(f"Using {settings['Model']} with temperature = {settings['Temperature']}.")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    search_engine = cl.user_session.get("search_engine")

    retriever = search_engine.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.5, 
                    "k": 5
                    },
            )
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=retriever,
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

    cl.user_session.set("chain", chain)

    logger.info("App setup using latest config.")

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

try:
    @cl.set_starters
    async def set_starters():
        return [
            cl.Starter(
                label="Microsoft's Operating Margin",
                message="What is Microsoft's Operating Margin in 2023?",
                ),
            cl.Starter(
                label="Apple's Net iPhone Sales",
                message="What is the latest net sales for Apple's iphones?",
                ),
            cl.Starter(
                label="Uber's CEO",
                message="Who is Uber's CEO?",
                ),
            ]
except:
    logger.warning("Please upgrade chainlit version.")

if __name__ == "__main__":
    # from langchain_community.document_loaders import TextLoader, DirectoryLoader

    # loader = DirectoryLoader('knowledge', use_multithreading=True, show_progress=True, glob="**/*.md")
    # docs = loader.load()
    
    # docs[0].metadata['source'] = get_file_name_from_path(docs[0].metadata['source'])
    # print(docs[0].metadata['source'])
    # print(docs[1].metadata['source'])

    # do process files, change env var reset chroma, do process files again and see difference in document parsing
    # with open('test-11.txt', encoding='utf-8', mode='w') as text_file1:
    #     text_file1.write(str(process_file()))

    # os.environ['RESET_CHROMA'] = 'True'
    
    # with open('test-21.txt', encoding='utf-8', mode='w') as text_file2:
    #     text_file2.write(str(process_file()))
    pass