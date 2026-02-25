from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ-API-KEY")
llm = ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-120b", temperature=0.1,max_tokens=1024)


def rag_simple(query,retriever,llm=llm,top_k=5):
    
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['document'] for doc in results]) if results else""
    if not context:
        return "No relevant information found in the documents."
    
    
    ##generate the answer using GROQ LLM
    prompt = f"""Use the following context to answer the question concisely
    Context : {context}
    
    Question: {query}
    Answer"""
    # print("------------------------------------------------------------------\n" + prompt)
    # print("------------------------------------------------------------------\n")
    response =llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    query = input("Enter your query: ")
    response = rag_simple(query, retriever, llm)
    print("Response:\n", response)