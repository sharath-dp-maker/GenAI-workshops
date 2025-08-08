from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the embedder
embedder = SentenceTransformerEmbedder(model="sentence-transformers/all-MiniLM-L12-v2", dimensions=384)

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
knowledge_base = PDFUrlKnowledgeBase(
    # Read PDF from this URL
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Store embeddings in the `ai.recipes` table
    vector_db=PgVector(table_name="recipes", db_url=db_url, search_type=SearchType.hybrid, embedder=embedder),
)
# Load the knowledge base: Comment after first run
knowledge_base.load(recreate=True, upsert=True)

agent = Agent(
    model=Groq(id="llama3-8b-8192"),
    knowledge=knowledge_base,
    # Enable RAG by adding references from AgentKnowledge to the user prompt.
    add_context=True,
    # Set as False because Agents default to `search_knowledge=True`
    search_knowledge=False,
    markdown=True,
    # debug_mode=True,
)
agent.print_response("How do I make chicken and galangal in coconut milk soup")