from langchain.prompts import PromptTemplate

WELCOME_MESSAGE = """### Welcome to the RAG prototype for financial documents!"""

template = """
Please act as an expert financial analyst for KPMG who has experience with financial statements, jargon, calculations and any general financial or fundamental questions about the context provided.
When you answer the questions and pay special attention to the markdown-format financial statement tables provided for financial records.

Given the following extracted parts of a long document, the chat history and the user's latest question, create a final answer with references ("SOURCES").

If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
Do not mention stuff like "as per the given context", "according to the context", "in the given context", etc in your response.
You have to be very concise in your response but make sure it answers the question asked please.
You should always check for context in the latest question, chat history and sources.
You should always check for context in the latest question, chat history and sources.

Additional Instructions:
1. You should answer questions about Microsoft only if the three: question, chat history and sources have context as Microsoft.
2. If suppose a question is for Apple but the document sources are for Uber, do not answer these questions. If user asks any follow up question for Apple, you may answer this question.
3. User may change context in the middle of the chat. If user's question includes enough context, you may ignore chat history.
Additional Instructions:
1. You should answer questions about Microsoft only if the three: question, chat history and sources have context as Microsoft.
2. If suppose a question is for Apple but the document sources are for Uber, do not answer these questions. If user asks any follow up question for Apple, you may answer this question.
3. User may change context in the middle of the chat. If user's question includes enough context, you may ignore chat history.

ALWAYS return a "SOURCES" field in your answer, with the format "SOURCES: <source1>, <source2>, <source3>, ...".

LATEST QUESTION: 
{question}
=========
CHAT HISTORY:
{chat_history}
=========
{summaries}
=========
FINAL ANSWER:
"""

PARSING_INSTRUCTIONS = """
You are parsing a document titled "Annual 10-K Filings" which is an official report submitted by publicly traded companies to the U.S. Securities and Exchange Commission (SEC). 
The 10-K provides a comprehensive summary of a company's financial performance, including audited financial statements.

It is divided into several sections and items that offer a detailed overview of the business.

Key Sections and Items:
1. **Business**: Description of the company's main operations, products, services, and business model.
   - Example: Apple's description of its hardware products, software, and services ecosystem.
2. **Risk Factors**: Detailed discussion of potential risks that may affect the company's financial health.
   - Example: Climate change risks discussed in ExxonMobil's 10-K filing.
3. **Selected Financial Data**: Summary of financial performance over the past five years, including income, cash flow, and balance sheet data.
   - Example: Amazon's financial summary showing revenue growth trends.
4. **Management's Discussion and Analysis (MD&A)**: Management's perspective on the financial results, trends, and future outlook.
   - Example: Tesla's discussion on production challenges and future expansion plans.
5. **Financial Statements and Supplementary Data**: Audited financial statements, including the income statement, balance sheet, and cash flow statement.
   - Example: Microsoft's audited financials with notes on revenue recognition and segment reporting.

Additional Sections:
- **Executive Compensation**: Information on the compensation of key executives.
  - Example: Compensation packages for Google's top executives.
- **Properties**: Information about the company’s physical properties and facilities.
  - Example: Details about Walmart’s distribution centers and retail locations.
- **Legal Proceedings**: Overview of any significant legal proceedings involving the company.
  - Example: Johnson & Johnson's litigation related to product liability.
- **Market for Registrant's Common Equity and Related Stockholder Matters**: Information on the company’s stock, dividends, and shareholder matters.
  - Example: General Electric’s dividend policy and stock performance.

The document contains numerous tables and figures to support the financial data and analysis. 

Answer questions using the information in this document and be extremely precise.
"""

PROMPT = PromptTemplate(template=template, input_variables=["summaries", "chat_history", "question"])

EXAMPLE_PROMPT = PromptTemplate(    
        template="Content: {page_content}\nSource: {source}",    
        input_variables=["page_content", "source"],
    )

# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question", "chat_history"],
#     template="""You are an AI language model assistant. Your task is to generate five 
#     different versions of the given user question to retrieve relevant documents from a vector 
#     database. By generating multiple perspectives on the user question, your goal is to help
#     the user overcome some of the limitations of the distance-based similarity search. 
#     Provide these alternative questions separated by newlines.
#     ==========
#     Original question: {question}""",
# )

# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
#          Chat History:
#          {chat_history}
#          Follow Up Input: {question}
#          Standalone question:"""

# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)