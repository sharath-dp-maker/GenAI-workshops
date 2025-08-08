import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector2,SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from phi.llm.groq import Groq
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import os
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()

# Set Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Database URL
#db_url = "postgresql+psycopg://postgres:Welcome2021@rds1.devst.xfactrs.com:5432/sharath_ai"
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Initialize the embedder
embedder = SentenceTransformerEmbedder(model="sentence-transformers/all-MiniLM-L12-v2", dimensions=384)

# Create an SQLAlchemy engine
engine = create_engine(db_url)

# Test the database connection
def test_database_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
    except OperationalError as e:
        print("❌ Error connecting to the database:")
        print(e)
        raise

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path: str) -> str:
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Define the PDF directory
pdf_directory = r"C:\Users\91855\Synthesis Systems\xfactrs - Documents\80 References\50 SQL"

# Get list of PDF files in the directory
pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

# Extract text from all PDF files
pdf_texts = [extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]

# Create a knowledge base from the extracted text
knowledge_base = PDFKnowledgeBase(
    path=pdf_directory,  # Specify the directory containing your PDFs
    texts=pdf_texts,     # Provide the extracted text
    vector_db=PgVector2(
        collection="LearnSQL",  # Use "collection" instead of "table_name"
        db_url=db_url,
        embedder=embedder,
    ),
)


# Load the knowledge base (only on first run or when updating)
knowledge_base.load(recreate=True, upsert=True)

# Define the storage for the assistant
storage = PgAssistantStorage(table_name="LearnSQL", db_url=db_url)

def LearnSQL(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    # Test the database connection before proceeding
    test_database_connection()

    # If not a new session, retrieve the existing run ID
    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if existing_run_ids:
            run_id = existing_run_ids[0]
            print(f"🔍 Continuing Run: {run_id}")
        else:
            print("🔍 No existing run found. Starting a new session.")

    # Initialize the assistant
    assistant = Assistant(
        run_id=run_id,
        llm=Groq(model="llama3-8b-8192", name="Groq", embedder=embedder),
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,  # Show tool calls in the response
        search_knowledge=True,  # Enable searching the knowledge base
        read_chat_history=True,  # Enable reading chat history
    )

    # Print the run ID for new sessions
    if run_id is None:
        run_id = assistant.run_id
        print(f"🚀 Started New Run: {run_id}")

    # Start the CLI app
    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(LearnSQL)