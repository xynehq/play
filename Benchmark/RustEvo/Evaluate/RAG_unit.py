import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores.utils import filter_complex_metadata
# Suppress deprecation warnings for a cleaner output
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Load environment variables from .env file
load_dotenv()

def select_chat_model(model="OpenAI",version=""):
    import getpass
    import os

    if model=="Groq":
        if not os.environ.get("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

        from langchain.chat_models import init_chat_model

        llm = init_chat_model("llama3-8b-8192", model_provider="groq")

    if model=="OpenAI":
        if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = ""

        
        from langchain.chat_models import init_chat_model

        llm = init_chat_model("gpt-4o-mini", model_provider="openai")


    if model=='Anthropic':
        if not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

        from langchain.chat_models import init_chat_model

        llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

    if model=='Azure':
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    if model == "Google Vertex":
        # Ensure your VertexAI credentials are configured

        from langchain.chat_models import init_chat_model

        llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
    if model == "AWS":
        # Ensure your AWS credentials are configured

        from langchain.chat_models import init_chat_model

        llm = init_chat_model("anthropic.claude-3-5-sonnet-20240620-v1:0", model_provider="bedrock_converse")

    if model == "NVIDIA":
        if not os.environ.get("NVIDIA_API_KEY"):
            os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter API key for NVIDIA: ")

        from langchain.chat_models import init_chat_model

        llm = init_chat_model("meta/llama3-70b-instruct", model_provider="nvidia")

    if model == "Together AI":
        if not os.environ.get("TOGETHER_API_KEY"):
            os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter API key for Together AI: ")

        from langchain.chat_models import init_chat_model

        llm = init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together")

    return llm

def select_embeddings_model(model="OpenAI"):
    import getpass
    import os

    if model == "OpenAI":
        os.environ["OPENAI_API_KEY"] = ""


        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", )

        # from langchain_openai import OpenAIEmbeddings
        
        # # 查看 OpenAIEmbeddings 类的参数，是否支持 openai_api_base
        # embeddings = OpenAIEmbeddings(
        #     model="text-embedding-3-large",
        #     openai_api_base="https://api.agicto.cn/v1"
        # )

    if model == "Azure":
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

        from langchain_openai import AzureOpenAIEmbeddings

        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    if model == "Google":
        from langchain_google_vertexai import VertexAIEmbeddings
        embeddings = VertexAIEmbeddings(model="text-embedding-004")

    if model == "AWS":
        from langchain_aws import BedrockEmbeddings
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    if model =="HuggingFace":
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    if model == "Ollama":
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="llama3")

    return embeddings

def get_RAG_document(query: str="",embedding_model="OpenAI", chat_model='OpenAI' ) -> str:
    from pprint import pprint
    # Assuming your JSON data is in a file named 'data.json'
    from langchain_community.document_loaders import JSONLoader

    file_path = "./data/final_docs.json"

    def metadata_func(record: dict, metadata: dict) -> dict:
        if "crate" not in record:
            metadata["name"] = record.get("name")
            metadata["from_version"] = record.get("from_version")
            metadata["to_version"] = record.get("to_version")
            metadata["module"] = record.get("module")
            metadata["type"] = record.get("type")
            metadata["signature"] = record.get("signature")
            metadata["documentation"] = record.get("documentation")
            # metadata["examples"] = record.get("examples")
            metadata["source_code"] = record.get("source_code")
            # 转换任何复杂类型为字符串
            for key, value in list(metadata.items()):
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
                elif value is None:
                    metadata[key] = ""
        else:
            metadata["crate"] = record.get("crate")
            metadata["name"] = record.get("name")
            metadata["from_version"] = record.get("from_version")
            metadata["to_version"] = record.get("to_version")
            metadata["module"] = record.get("module")
            metadata["type"] = record.get("type")
            metadata["signature"] = record.get("signature")
            metadata["documentation"] = record.get("documentation")
            metadata["source_code"] = record.get("source_code")            
            for key, value in list(metadata.items()):
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
                elif value is None:
                    metadata[key] = ""
        return metadata

    loader = JSONLoader(
        file_path="./data/final_docs.json",
        jq_schema=".[]",  # Iterate over each object in the array
        metadata_func=metadata_func,
        text_content=False  # Preserve JSON object as string in page_content
    )

    docs = loader.load()

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    # filtered_splits = [filter_complex_metadata(doc) for doc in all_splits]
    embeddings = select_embeddings_model(model=embedding_model)

    from langchain_chroma import Chroma

    # Assuming 'embeddings' is your embedding function and 'all_splits' is your list of documents
    vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

    # Check if the vector store already has documents
    existing_ids = vector_store.get()["ids"]  # Get the list of document IDs in the store

    if not existing_ids:
        # If no documents exist, embed and save
        ids = vector_store.add_documents(documents=all_splits)
        # ids = vector_store.add_documents(documents=filtered_splits)
        print(f"Added documents with IDs: {ids}")
    else:
        # If documents exist, just load (vector_store is already initialized)
        # print(f"Loaded existing store with document IDs: {existing_ids}")
        print(f"Loaded existing store with document IDs")

    from langchain.prompts import ChatPromptTemplate

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context: {context}\nQuestion: {question}")

    # Your existing code
    question = query
    retrieved_docs = vector_store.similarity_search(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt_value = prompt.invoke({"question": question, "context": docs_content})
    prompt_str = prompt_value.to_string()

    llm=select_chat_model(model=chat_model)

    try:
        answer = llm.invoke(prompt_str)
    except AttributeError:
        # Fallback for older LangChain versions or misconfigured LLMs
        answer = llm(prompt_str)
    #print("\n\nanswer:", answer)
    return answer

def test_get_RAG_document():
    context=get_RAG_document("what is hash_one API","OpenAI","OpenAI")
    print(context)



