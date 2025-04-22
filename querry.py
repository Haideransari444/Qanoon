import os
from dotenv import load_dotenv
# Updated imports for LangChain 0.2+
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Path to your saved FAISS index
VECTOR_STORE_PATH = r"E:\legeslative bot\vectordb"

# Load the embeddings model - same model used for creating embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the vector store with security flag set to allow pickle deserialization
print("Loading vector store...")
vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH, 
    embeddings,
    allow_dangerous_deserialization=True  # Only use this if you trust the source of your vector store
)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve 5 most relevant documents
)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
)

# Create a custom prompt template that includes instructions to cite sources
template = """
You are a Pakistani legal expert assistant. Use only the provided legal documents to answer the user's question.
If the documents do not contain relevant information, answer using your knowledge of Pakistani law.

Legal Documents:
{context}

User’s Question:
{question}

Your response must follow these rules:

    Provide a clear, explainatory, and fact-based answer using appropriate legal terminology.

    Do not reference the Quran or Sunnah in your answer.

    Base your response strictly on the provided legal documents or authentic Pakistani law knowledge.

    Do not include document citations within the body of the answer.

    Do not infer or fabricate information not explicitly present in the legal documents.

    Exclude any documents that are unrelated to the question.

    At the end of your answer, list all relevant sources in the following format:

        If the document is a .txt file, include only the file name (e.g., judgment.txt)

        If the document is a .pdf file, include the file name and page number (e.g., constitution.pdf, Page 5)
        Present these sources under a clearly labeled "Sources:" section.

    If no documents are relevant, omit the "Sources" section altogether.
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create the chain
def format_docs(docs):
    return "\n\n".join(f"Document: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

def get_legal_answer(question):
    """Get an answer to a legal question with citations"""
    try:
        answer = qa_chain.invoke(question)
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Interactive chat loop
def main():
    print("Pakistani Legal Assistant")
    print("Ask me questions about Pakistani law, and I'll provide answers with citations.")
    print("Type 'exit' to quit.")
    
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ["exit", "quit", "bye"]:
            print("Thank you for using the Pakistani Legal Assistant!")
            break
            
        answer = get_legal_answer(user_question)
        print("\nLegal Response:")
        print(answer)

if __name__ == "__main__":
    main()