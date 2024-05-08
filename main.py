from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core import prompts
from langchain.chains import LLMChain
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from dotenv import find_dotenv, load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import textwrap

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    

    

    prompt = prompts.PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs



video_url = input("Give a Youtube URL \n")
db = create_db_from_youtube_video_url(video_url)

query = "What is the historical context?"
response, docs = get_response_from_query(db, query)
print(textwrap.fill(response, width=85))

while True:
    print()
    query = input("Do you have any questions? Or type quit to quit\n")
    if query == "quit":
        break
    
    response, docs = get_response_from_query(db, query)
    print(textwrap.fill(response, width=85))