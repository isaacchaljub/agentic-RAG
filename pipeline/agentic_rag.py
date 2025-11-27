import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Task, Crew, LLM
from numpy.linalg import norm
from numpy import dot
import numpy as np

#Load the environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

global_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#Create two LLMs, one to get the question and search for the answer in documents, the other one 
# to be the agent and search online if the documents don't contain the answer.
# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2,
)


crew_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.7
)

#Create a LLM function that answers whether the question can be answered by the documents or not.
# def can_answer_question()

def can_answer_question_from_documents(query, context):
    """Router function to determine if we can answer from local knowledge"""
    prompt = '''Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer. 
Output Format:
    - Answer: Yes/No
Study the below examples and based on that, respond to the last question. 
Examples:
    Input: 
        Text: The capital of France is Paris.
        User Question: What is the capital of France?
    Expected Output:
        Answer: Yes
    Input: 
        Text: The population of the United States is over 330 million.
        User Question: What is the population of China?
    Expected Output:
        Answer: No
    Input:
        User Question: {query}
        Text: {text}
'''
    formatted_prompt = prompt.format(text=context, query=query)
    response = llm.invoke(formatted_prompt)
    return response.content.strip().lower() == "yes"

#Create a function that searches the internet for the answer to the question.
def setup_web_scraping_agent():
    """Setup the web scraping agent and related components"""
    search_tool = SerperDevTool()  # Tool for performing web searches
    scrape_website = ScrapeWebsiteTool()  # Tool for extracting data from websites
    
    # Define the web search agent
    web_search_agent = Agent(
        role="Expert Web Search Agent",
        goal="Identify and retrieve relevant web data for user queries",
        backstory="An expert in identifying valuable web sources for the user's needs",
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )
    
    # Define the web scraping agent
    web_scraper_agent = Agent(
        role="Expert Web Scraper Agent",
        goal="Extract and analyze content from specific web pages identified by the search agent",
        backstory="A highly skilled web scraper, capable of analyzing and summarizing website content accurately",
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )
    
    # Define the web search task
    search_task = Task(
        description=(
            "Identify the most relevant web page or article for the topic: '{topic}'. "
            "Use all available tools to search for and provide a link to a web page "
            "that contains valuable information about the topic. Keep your response concise."
        ),
        expected_output=(
            "A concise summary of the most relevant web page or article for '{topic}', "
            "including the link to the source and key points from the content."
        ),
        tools=[search_tool],
        agent=web_search_agent,
    )
    
    # Define the web scraping task
    scraping_task = Task(
        description=(
            "Extract and analyze data from the given web page or website. Focus on the key sections "
            "that provide insights into the topic: '{topic}'. Use all available tools to retrieve the content, "
            "and summarize the key findings in a concise manner."
        ),
        expected_output=(
            "A detailed summary of the content from the given web page or website, highlighting the key insights "
            "and explaining their relevance to the topic: '{topic}'. Ensure clarity and conciseness."
        ),
        tools=[scrape_website],
        agent=web_scraper_agent,
    )
    
    # Define the crew to manage agents and tasks
    crew = Crew(
        agents=[web_search_agent, web_scraper_agent],
        tasks=[search_task, scraping_task],
        verbose=1,
        memory=False,
    )
    return crew

def get_web_content(query):
    """Get content from web scraping"""
    crew = setup_web_scraping_agent()
    result = crew.kickoff(inputs={"topic": query})
    return result.raw


# Load all documents from the documents directory
def load_all_documents(documents_directory="documents"):
    """Load all PDF documents from the directory"""
    all_documents = []
    for file in os.listdir(documents_directory):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(documents_directory, file))
            documents = loader.load()
            all_documents.extend(documents)
    return all_documents

# Create the vector store from the documents
def create_vector_database(documents):
    """Create a vector store from the documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    #Create the embeddings
    embeddings = global_embeddings
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# Initialize the vectorstore (call this once, then reuse it)
def initialize_vectorstore(documents_directory="documents"):
    """Load all documents and create a single vectorstore"""
    all_documents = load_all_documents(documents_directory)
    if not all_documents:
        raise ValueError(f"No PDF documents found in the documents directory: {documents_directory}")
    vectorstore = create_vector_database(all_documents)
    return vectorstore

# Create the RAG pipeline
def create_rag_pipeline(query, vectorstore=None):
    """Create the RAG pipeline - searches the vectorstore for the query"""
    if vectorstore is None:
        vectorstore = initialize_vectorstore()
    local_content = vectorstore.similarity_search(query, k=5)
    return " ".join([doc.page_content for doc in local_content])

#Use an LLM to generate the final answer
def generate_final_answer(context, query):
    """Generate final answer using LLM"""
    messages = [
        (
            "system",
            "You are a helpful assistant. Use the provided context to answer the query accurately.",
        ),
        ("system", f"Context: {context}"),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content

#Use caching to store answers and retrieve if similar queries are 
# aked again, relieving the LLM usage and reducing the cost.

query_cache = []
max_cache_size = 20
similarity_threshold = 0.85

def cosine_similarity(query_embedding, cached_embedding):
    return dot(query_embedding, cached_embedding) / (norm(query_embedding) * norm(cached_embedding))

#Process the query and answer according to the context
def process_query(query, vector_db):
    """Main function to process user query"""
    print(f"Processing query: {query}")

    # Step 0: Check if the query is in the cache
    if len(query_cache) != 0:
        embedded_query = global_embeddings.embed_query(query)
        similarity_scores = [cosine_similarity(embedded_query, cached_item[0]) for cached_item in query_cache]
        if np.max(similarity_scores) > similarity_threshold:
            return query_cache[np.argmax(similarity_scores)][2]
#No cache, continue to the next step
    
    # Step 1: Get initial context from local documents
    local_context = create_rag_pipeline(query, vector_db)
    print("Retrieved initial context from local documents")
    
    # Step 2: Check if we can answer from local knowledge
    can_answer_locally = can_answer_question_from_documents(query, local_context)
    print(f"Can answer locally: {can_answer_locally}")
    
    # Step 3: Get context either from local DB or web
    if can_answer_locally:
        context = local_context
        print("Using context from local documents")
    else:
        context = get_web_content(query)
        print("Retrieved context from web scraping")
    
    # Step 4: Generate final answer
    answer = generate_final_answer(context, query)

    # Step 5: Add the embeddings, query and answer to the cache
    query_cache.append((global_embeddings.embed_query(query), query, answer))

    # Step 6: Prune the cache if it's too large
    if len(query_cache) > max_cache_size:
        query_cache.pop(0)

    return answer

#Main function to run the RAG pipeline
def main():
    """Main function to run the RAG pipeline"""
    #Path to the documents directory
    documents_directory = "documents"

    #Create the vector database
    vector_db = initialize_vectorstore(documents_directory)

    #Define the query
    # query = "What is Agentic RAG?"
    query = "What is the highest city in Colombia?"

    #Answer the query
    answer = process_query(query, vector_db)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()