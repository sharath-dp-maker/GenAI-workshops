import os
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize PDF Knowledge Base
embedder = SentenceTransformerEmbedder(model="sentence-transformers/all-MiniLM-L12-v2", dimensions=384)
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],  # Example PDF URL
    vector_db=PgVector(table_name="recipes", db_url=db_url, search_type=SearchType.hybrid, embedder=embedder),
)

# Load the knowledge base (comment out after the first run)
knowledge_base.load(recreate=True, upsert=True)

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    # Get Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    model = 'llama3-8b-8192'

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    print("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

    system_prompt = 'You are a friendly conversational chatbot with knowledge about recipes and cooking.'
    conversational_memory_length = 5  # Number of previous messages the chatbot will remember

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    while True:
        user_question = input("Ask a question: ")

        # If the user has asked a question
        if user_question:
            # Construct a chat prompt template using various components
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=system_prompt
                    ),  # Persistent system prompt included at the start of the chat.

                    MessagesPlaceholder(
                        variable_name="chat_history"
                    ),  # Placeholder for actual chat history

                    HumanMessagePromptTemplate.from_template(
                        "{human_input}"
                    ),  # Template for the current user's input
                ]
            )

            # Create a conversation chain using the LangChain LLM (Language Learning Model)
            conversation = LLMChain(
                llm=groq_chat,  # The Groq LangChain chat object
                prompt=prompt,  # The constructed prompt template
                verbose=False,   # Set to True for verbose output (for debugging)
                memory=memory,  # The conversational memory object
            )

            # Get response from Groq chat model
            response = conversation.predict(human_input=user_question)

            # Perform the query on the vector database (PgVector) for knowledge base
            # Using the vector database directly to find relevant information for the question
            knowledge_response = knowledge_base.vector_db.query(user_question)  # Querying the PgVector DB
            
            if knowledge_response:
                response += f"\n\nHere's something I found in my knowledge base:\n{knowledge_response}"

            print("Chatbot:", response)

if __name__ == "__main__":
    main()
