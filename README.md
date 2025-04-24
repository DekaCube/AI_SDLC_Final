### Project Overview 
TODO

### Running Locally
TODO

### Getting a NGC API Key
TODO




### Prompts

Model Used `Claude 3.7 Sonnet Thinking`

### Code block 1

```
I'm starting a GenAI project in a Jupyter notebook and need to install the necessary packages. Can you provide the installation code for the following specific libraries:
- langchain
- langchain-community
- langchain-nvidia-ai-endpoints
- gradio (for creating the user interface)
- rich (for better console output)
- arxiv (for loading academic papers)
- pymupdf (for PDF document processing)
- faiss-cpu (for vector storage and similarity search)

I want to create a document retrieval and summarization system that integrates with NVIDIA AI endpoints.
```

### Code Block 2

```
Make me code to load the value NVIDIA_API_KEY from a .env file. Store the api key in an OS variable called NVIDIA_API_KEY
```

```
Refactor this to raise a value error if the key is not found.
```


### Code Block 3
```
I'm working on a GenAI project using NVIDIA's AI endpoints through LangChain. I need to see what specific models are available for my project through NVIDIA's API integration. Can you write the exact code I need to import the ChatNVIDIA class from the langchain_nvidia_ai_endpoints module and then display all the available models that I can use with this integration? I just need the import statement and the method call to list all models, nothing else.

Only display the first 20 models
```


### Code Block 4
```Create a comprehensive implementation for a document summarization system using LangChain and NVIDIA's AI endpoints. I need a complete Python code block that:

1. Sets up a clearly marked Document Summarization section with appropriate imports from:
   - langchain_core modules (RunnableLambda, RunnableAssign, ChatPromptTemplate, StrOutputParser, PydanticOutputParser)
   - langchain modules (RecursiveCharacterTextSplitter, document loaders for both UnstructuredFileLoader and ArxivLoader)
   - pydantic (BaseModel, Field) for structured data modeling
   - typing (List) for type hints
   - IPython.display (clear_output) for notebook interface management
   - functools (partial) for function manipulation
   - rich library components (Console, Style, Theme) for enhanced console output

2. Defines a Pydantic data model called DocumentSummaryBase with three fields:
   - running_summary: string field starting empty, with description explaining it should be updated not overridden
   - main_ideas: list of strings, limited to 3 maximum important points from the document
   - loose_ends: list of strings representing open questions, limited to 3 maximum

3. Implements console formatting with rich, creating a Console object and custom green NVIDIA-branded style (#76B900)

4. Contains document loading code that:
   - Includes commented-out code for UnstructuredFileLoader as a general-purpose option
   - Uses ArxivLoader to fetch the GraphRAG paper (ID: 2404.16130)

5. Configures a RecursiveCharacterTextSplitter with:
   - chunk_size of 1200 characters
   - chunk_overlap of 100 characters
   - comprehensive separators list including newlines, periods, semicolons, commas, spaces, and empty strings
   - Includes commented-out preprocessing code to handle potential formatting issues

6. Creates a detailed ChatPromptTemplate for summarization that:
   - Instructs to generate a running technical summary
   - Emphasizes not losing information when updating the knowledge base
   - Directs information flow from chunks to main ideas/loose ends to running summary
   - Includes placeholders for format instructions and input
   - Emphasizes following formatting precisely

Ensure the code is well-structured with appropriate section headers, comments explaining key components, and follows best practices for LangChain development.```
