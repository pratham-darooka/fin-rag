from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate

import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rag_init import process_file
# generator with openai models
generator_llm = ChatGoogleGenerativeAI(        
            model='gemini-pro',        
            temperature=1,         
            convert_system_message_to_human=True    
            )
critic_llm = ChatGoogleGenerativeAI(        
            model='gemini-pro',        
            temperature=1,         
            convert_system_message_to_human=True    
            )
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)


if __name__ == "__main__":
    documents = process_file()

    for document in documents:
        document.metadata['filename'] = document.metadata['source']

    testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
    ts_df = testset.to_pandas()
    ts_df.to_excel(os.path.join('test','ragas_test_queries.xlsx'), index=False)

    result = evaluate(
        ts_df["eval"],
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )
    res_df = result.to_pandas()
    res_df.to_excel(os.path.join('test','ragas_test_scores.xlsx'), index=False)