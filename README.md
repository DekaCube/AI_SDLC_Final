### Project Overview 

This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) system using NVIDIA AI endpoints through the LangChain framework. The system is designed for advanced document processing, summarization, and knowledge extraction with generative AI capabilities.

The project leverages an existing NVIDIA RAG project that a team member had previously worked on in parallel this semester, which significantly accelerated development through one-shot prompting techniques. This approach allowed us to rapidly implement complex RAG components while customizing the solution to our specific requirements

### Running Locally

To set up your local environment:

1. **Install virtualenv** for Python environment management:
   ```bash
   pip install virtualenv
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Create environment
   virtualenv .venv
   
   # Activate environment
   source .venv/bin/activate 
   ```

3. **Install VSCode Jupyter extension**:
   - Open VSCode Extensions (Ctrl+Shift+X)
   - Search for "Jupyter" and install
   - Reload VSCode when prompted

4. **Create a .env file for API key**:
   ```bash
   # Create empty .env file in project root
   touch .env
   
   # Add your NVIDIA API key to the file
   echo "NVIDIA_API_KEY=your_api_key_here" > .env
   ```
   - Replace `your_api_key_here` with your actual NVIDIA NGC API key
   - This file will be loaded by the application to authenticate API requests

### Prompts

Model Used `Claude 3.7 Sonnet Thinking`

#### Code block 1

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

I want to create a document retrieval and summarizatiosystem that integrates with NVIDIA AI endpoints.
```

#### Code Block 2

```
Make me code to load the value NVIDIA_API_KEY from a .env file. Store the api key in an OS variable called NVIDIA_API_KEY
```

```
Refactor this to raise a value error if the key is not found.
```


#### Code Block 3
```
I'm working on a GenAI project using NVIDIA's AI endpoints through LangChain. I need to see what specific models are available for my project through NVIDIA's API integration. Can you write the exact code I need to import the ChatNVIDIA class from the langchain_nvidia_ai_endpoints module and then display all the available models that I can use with this integration? I just need the import statement and the method call to list all models, nothing else.

Only display the first 20 models
```


#### Code Block 4
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
```

#### Code Block 5

```
I'm building a RAG system using LangChain and NVIDIA AI endpoints, and I need a robust function for extracting structured information from LLM outputs using Pydantic models. The function should be called `RExtract` and will be a crucial component of my summarization system.

Please write a function that:

1. Takes three parameters:
   - `pydantic_class`: A Pydantic model class that defines the structure of the expected output
   - `llm`: A language model that will generate text
   - `prompt`: A prompt template to send to the LLM

2. Creates a PydanticOutputParser for the given Pydantic class

3. Uses RunnableAssign to inject format instructions into the prompt context, by creating a lambda that calls parser.get_format_instructions()

4. Implements a preprocessing function called `preparse` that handles common formatting issues in LLM outputs by:
   - Adding missing opening and closing braces if they don't exist
   - Replacing escaped underscores with regular underscores
   - Replacing newlines with spaces
   - Fixing escaped brackets by replacing "\\]" with "]" and "\\[" with "["
   - Include a commented-out print statement for debugging

5. Returns a complete LangChain runnable chain that:
   - First injects format instructions
   - Then applies the prompt
   - Then sends to the LLM
   - Then preprocesses the output with the preparse function
   - Finally parses the result with the Pydantic parser

The function should have a docstring explaining that it's a "Runnable Extraction module" that "Returns a knowledge dictionary populated by slot-filling extraction".

This function will be used in a document summarization system to extract structured information like running summaries, main ideas, and open questions from document chunks.
```

#### Code Block 6

```
I'm continuing to build my RAG system with NVIDIA AI endpoints and LangChain, and now I need a document summarization pipeline that processes chunks of text incrementally, building up a comprehensive summary.

Please write a function called `RSummarizer` that:

1. Implements an incremental document summarizer that processes a list of document chunks sequentially, building on previously extracted information
   
2. Takes four parameters:
   - `knowledge`: A Pydantic model instance (like DocumentSummaryBase) representing the initial state
   - `llm`: A language model for text generation
   - `prompt`: A prompt template for summarization instructions
   - `verbose`: A boolean flag to control progress output (default False)
   
3. Creates a nested function called `summarize_docs` that:
   - Uses RunnableAssign to create a parsing chain that utilizes the previously defined RExtract function
   - Initializes a state dictionary with the provided knowledge object
   - Processes each document in sequence, updating the state with new information
   - Validates that 'info_base' exists in the state after each update
   - If verbose=True, prints progress information showing how many documents have been processed
   - Stores the current summary in a global variable called `latest_summary` for recovery in case of crashes
   - Uses clear_output() to keep the notebook clean during processing
   - Returns the final info_base containing the accumulated summary data

4. Makes the function return a RunnableLambda wrapping the summarize_docs function

After defining the function, include code that:

1. Creates a variable called `latest_summary` initialized to an empty string to store the most recent summary

2. Sets up a ChatNVIDIA model using the "mistralai/mistral-7b-instruct-v0.3" model with max_tokens set to 4096

3. Pipes the model through a StrOutputParser to get clean string output

4. Creates a summarizer instance using:
   - An empty DocumentSummaryBase object
   - The configured LLM
   - The previously defined summary_prompt
   - verbose=True to show progress

5. Invokes the summarizer on the first 10 chunks of the document (docs_split[:10])

This function should build on the previously defined RExtract function and work with the DocumentSummaryBase model and summary_prompt template that have already been created.
```

#### Code Block 7

```
I'm continuing to build my RAG (Retrieval-Augmented Generation) system with NVIDIA AI endpoints and need to set up the core document processing pipeline. Please provide a complete implementation for the main RAG section of my code with the following specific components:

1. Create a section header with a clear visual separator using "#" characters and titled "Retrieval-Augmented Generation (RAG)"

2. Include ALL necessary imports grouped by functionality:
   - Standard library: json
   - UI: gradio as gr
   - NVIDIA specific: ChatNVIDIA and NVIDIAEmbeddings from langchain_nvidia_ai_endpoints
   - Vector storage: FAISS from langchain_community.vectorstores and IndexFlatL2 from faiss
   - Document processing: RecursiveCharacterTextSplitter, ArxivLoader, UnstructuredPDFLoader
   - In-memory storage: InMemoryDocstore from langchain_community.docstore.in_memory
   - Document transformation: LongContextReorder
   - LangChain components: ChatPromptTemplate, StrOutputParser, RunnableLambda, RunnableAssign
   - Utilities: itemgetter from operator, partial from functools
   - Rich console formatting: Console, Style, and Theme from rich.console/style/theme

3. Set up NVIDIA-branded console output with:
   - Console() instance
   - Style with NVIDIA's official green color (#76B900) and bold text
   - A partial function called "pprint" that uses this style

4. Configure the embedding model using NVIDIAEmbeddings:
   - Use "nvidia/nv-embed-v1" model
   - Set truncate parameter to "END"
   - Include a commented-out line showing how to get available models

5. Set up LLM using ChatNVIDIA:
   - Use "mistralai/mixtral-8x22b-instruct-v0.1" model
   - Include a commented-out line showing how to get available models

6. Create two utility functions:
   - RPrint: A function that takes an optional preface parameter and returns a RunnableLambda that:
     * Prints the input with the NVIDIA styling
     * Returns the input unchanged (for chain composition)
     * Include appropriate docstring
   
   - docs2str: A function that takes docs and optional title parameters and:
     * Formats document chunks into a readable string format
     * Includes source attribution with "[Quote from {title}]" prefix when metadata is available
     * Include appropriate docstring

7. Create a document transformer using LongContextReorder wrapped in a RunnableLambda

8. Configure a RecursiveCharacterTextSplitter with:
   - chunk_size of 1000
   - chunk_overlap of 100
   - Comprehensive separators list: ["\n\n", "\n", ".", ";", ",", " "]

9. Load a specific collection of academic papers using ArxivLoader:
   - Print "Loading Documents" before loading
   - Include exactly 9 papers with their IDs:
     * 1706.03762 (Attention Is All You Need)
     * 1810.04805 (BERT)
     * 2005.11401 (RAG)
     * 2205.00445 (MRKL)
     * 2310.06825 (Mistral)
     * 2306.05685 (LLM-as-a-Judge)
     * 2210.03629 (ReAct)
     * 2112.10752 (Latent Stable Diffusion)
     * 2103.00020 (CLIP)
   - Add comments next to each paper explaining what it is

10. Process the loaded papers by:
    - Converting to JSON and truncating at the "References" section if present
    - Include appropriate comments explaining this process

11. Split the documents into chunks:
    - Print "Chunking Documents" before splitting
    - Use list comprehension to apply the text splitter to each document
    - Filter out chunks shorter than 200 characters using nested list comprehensions

12. Create metadata and summary information:
    - Initialize a string variable "doc_string" with "Available Documents:" 
    - Extract and append paper titles to this string with bullet points
    - Create a separate list for document metadata
    - Combine the doc_string and metadata into "extra_chunks"

13. Print summary information about the loaded documents:
    - Print the doc_string using the NVIDIA-styled pprint
    - Loop through each document and print:
      * Document index
      * Number of chunks
      * Metadata using the NVIDIA-styled pprint
      * Empty line between documents

Please include descriptive comments throughout the code to explain key components and techniques being used. The code should follow LangChain best practices and be ready to integrate with the document processing pipeline we've been building.
```

#### Code Block 8
```I need code to implement the vector store construction phase of my RAG system using FAISS and NVIDIA embeddings. Please create a code cell that:

1. Uses the %%time magic command to measure execution performance

2. Includes a clear header comment labeled "Constructing Your Document Vector Stores"

3. Creates vector stores from both:
   - The metadata/summary chunks (stored in `extra_chunks`) using FAISS.from_texts()
   - All document chunks (stored in `docs_chunks`) using FAISS.from_documents()
   - Combine these into a single list called `vecstores`

4. Implements a utility function named `default_FAISS()` that:
   - Creates an empty FAISS vectorstore with proper configuration
   - Uses the existing embedder to determine vector dimensions automatically
   - Sets up with IndexFlatL2, InMemoryDocstore, empty index_to_docstore_id dict
   - Sets normalize_L2=False
   - Includes appropriate docstring

5. Implements an aggregation function named `aggregate_vstores()` that:
   - Takes a list of vectorstores
   - Efficiently merges them into a single vectorstore
   - Include comments explaining how it works with the default_FAISS utility

6. Creates a final `docstore` variable by applying the aggregation function to the vecstores list

7. Prints a confirmation message showing the total number of document chunks in the final docstore

8. Include appropriate comments explaining key steps and any unintuitive optimizations

The code should follow best practices for vector store construction in LangChain and be optimized for memory efficiency when dealing with large document collections.
```

```
Fix this error

AttributeError Traceback (most recent call last)
File <timed exec>:80

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\langchain_core\vectorstores\base.py:837, in VectorStore.from_documents(cls, documents, embedding, **kwargs)
820 @classmethod
821 def from_documents(
822 cls,
(...)
825 **kwargs: Any,
826 ) -> Self:
827 """Return VectorStore initialized from documents and embeddings.
828
829 Args:
(...)
835 VectorStore: VectorStore initialized from documents and embeddings.
836 """
--> 837 texts = [d.page_content for d in documents]
838 metadatas = [d.metadata for d in documents]
840 if "ids" not in kwargs:
```

```
Fix this error

---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
File <timed exec>:99

File <timed exec>:55, in aggregate_vstores(vecstores)

AttributeError: 'InMemoryDocstore' object has no attribute 'dict'
```

```
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
File <timed exec>:100

File <timed exec>:65, in aggregate_vstores(vecstores)

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\langchain_community\vectorstores\faiss.py:385, in FAISS.add_embeddings(self, text_embeddings, metadatas, ids, **kwargs)
    383 # Embed and create the documents.
    384 texts, embeddings = zip(*text_embeddings)
--> 385 return self.__add(texts, embeddings, metadatas=metadatas, ids=ids)

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\langchain_community\vectorstores\faiss.py:301, in FAISS.__add(self, texts, embeddings, metadatas, ids)
    297 _len_check_if_sized(texts, ids, "texts", "ids")
    299 _metadatas = metadatas or ({} for _ in texts)
    300 documents = [
--> 301     Document(id=id_, page_content=t, metadata=m)
    302     for id_, t, m in zip(ids, texts, _metadatas)
    303 ]
    305 _len_check_if_sized(documents, embeddings, "documents", "embeddings")
    307 if ids and len(ids) != len(set(ids)):

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\langchain_core\documents\base.py:289, in Document.__init__(self, page_content, **kwargs)
    286 """Pass page_content in as positional or named arg."""
    287 # my-py is complaining that page_content is not defined on the base class.
    288 # Here, we're relying on pydantic base class to handle the validation.
--> 289 super().__init__(page_content=page_content, **kwargs)

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\langchain_core\load\serializable.py:130, in Serializable.__init__(self, *args, **kwargs)
    128 def __init__(self, *args: Any, **kwargs: Any) -> None:
    129     """"""  # noqa: D419
--> 130     super().__init__(*args, **kwargs)

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pydantic\main.py:253, in BaseModel.__init__(self, **data)
    251 # `__tracebackhide__` tells pytest and some other tools to omit this function from tracebacks
    252 __tracebackhide__ = True
--> 253 validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    254 if self is not validated_self:
    255     warnings.warn(
    256         'A custom validator is returning a value other than `self`.\n'
    257         "Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.\n"
    258         'See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.',
    259         stacklevel=2,
    260     )

ValidationError: 1 validation error for Document
metadata
  Input should be a valid dictionary [type=dict_type, input_value=Document(id='73e3ea24-6fc... Document 9 - CLIP\n'), input_type=Document]
    For further information visit https://errors.pydantic.dev/2.11/v/dict_type
```

### Code Block 9

```
Create a code block for the conversational RAG interface of my project with the following specific components:

Model Configuration:

Set up the embedding model using NVIDIAEmbeddings with "nvidia/nv-embed-v1" and truncate="END" parameter
Configure the instruction LLM using ChatNVIDIA with "mistralai/mixtral-8x7b-instruct-v0.1" model
Include commented-out alternative options:
A commented line showing how to list available models with ChatNVIDIA.get_available_models()
A commented alternative model option using meta/llama-3.1-8b-instruct
Conversation Memory Store:

Create a conversation store named "convstore" using the previously defined default_FAISS() utility function with the embedder
Memory Management Function:

Implement a function called save_memory_and_get_output that:
Takes a dictionary with 'input' and 'output' keys and a vector store
Stores both user and agent messages as embedded texts in the provided vector store
Formats them as "User previously responded with {input}" and "Agent previously responded with {output}"
Returns the output value from the dictionary
Include appropriate docstring explaining functionality
Welcome Message:

Create a multi-line initial greeting message variable called "initial_msg" that:
Introduces the agent as a document chat helper
References the available documents using the existing doc_string variable
Asks how it can help the user
Chat Prompt Template:

Create a ChatPromptTemplate with system and user messages
System message should:
Define the chatbot's role as a document assistant
Reference the user's input with {input} placeholder
Include sections for conversation history retrieval with {history} placeholder
Include document context with {context} placeholder
Give specific instructions to answer only from retrieval and maintain conversational tone
User message should use the {input} placeholder
Chain Implementation:

Create a stream_chain that pipes:
The chat_prompt through RPrint() for debugging
Then through the instruct_llm
Finally through StrOutputParser()
Create a retrieval_chain that:
Starts with a dictionary containing the input
Adds conversation history by retrieving from convstore, reordering with long_reorder, and formatting with docs2str
Adds document context by retrieving from docstore, reordering, and formatting
Ends with RPrint() for debugging visualization
Chat Generator Function:

Implement a function called chat_gen that:
Takes parameters for message, history (default empty list), and return_buffer (default True)
Creates an empty buffer variable
First invokes the retrieval chain to get context
Then streams tokens from the stream chain
Accumulates tokens in the buffer
Uses yield to return either the growing buffer or individual tokens based on return_buffer parameter
Saves the conversation to memory after completion
Test Implementation:

Create a test_question variable with "Tell me about RAG!"
Include a loop that calls chat_gen with the test question and return_buffer=False
Print each token without a line break using end=''
The code should be compatible with a LangChain-based RAG system that uses NVIDIA AI endpoints, and should properly integrate with the previously defined utility functions and vectorstores.
```

```
Fix this error
AttributeError                            Traceback (most recent call last)
Cell In[25], line 139
    136 print("\nTesting chat implementation with query:", test_question)
    137 print("\nResponse:")
--> 139 for token in chat_gen(test_question, return_buffer=False):
    140     print(token, end='')

Cell In[25], line 118
    115 buffer = ""
    117 # Perform retrieval to get context
--> 118 context = retrieval_chain.invoke(message)
    120 # Stream responses from the LLM
    121 for token in stream_chain.stream(context):
    ```

    ```
    Fix this error
    TypeError                                 Traceback (most recent call last)
Cell In[26], line 140
    137 print("\nTesting chat implementation with query:", test_question)
    138 print("\nResponse:")
--> 140 for token in chat_gen(test_question, return_buffer=False):
    141     print(token, end='')

Cell In[26], line 119
    115 buffer = ""
    117 # Perform retrieval to get context
    118 # FIX: Call retrieval_chain as a function rather than using .invoke()
--> 119 context = retrieval_chain(message)
    121 # Stream responses from the LLM
    122 for token in stream_chain.stream(context):

Cell In[26], line 83
     81 def retrieval_chain(user_input):
     82     return (
---> 83         {"input": user_input}
     84         | RunnableAssign({
     85             "history": lambda _: docs2str(
     86                 long_reorder.invoke(
     87                     convstore.similarity_search(user_input, k=5)
     88                 ),
     89                 title="Conversation History"
     90             ),
     91             "context": lambda x: docs2str(
     92                 long_reorder.invoke(
     93                     docstore.similarity_search(x["input"], k=8)
     94                 ),
     95                 title="Document Context"
     96             ),
     97             "input": itemgetter("input")
     98         })
     99         | RPrint("Retrieved Context")
    100     )

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\langchain_core\runnables\base.py:586, in Runnable.__ror__(self, other)
    576 def __ror__(
    577     self,
    578     other: Union[
   (...)
    583     ],
    584 ) -> RunnableSerializable[Other, Output]:
    585     """Compose this Runnable with another object to create a RunnableSequence."""
--> 586     return RunnableSequence(coerce_to_runnable(other), self)

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\langchain_core\runnables\base.py:5910, in coerce_to_runnable(thing)
   5908     return RunnableLambda(cast("Callable[[Input], Output]", thing))
   5909 if isinstance(thing, dict):
-> 5910     return cast("Runnable[Input, Output]", RunnableParallel(thing))
   5911 msg = (
   5912     f"Expected a Runnable, callable or dict."
   5913     f"Instead got an unsupported type: {type(thing)}"
   5914 )
   5915 raise TypeError(msg)

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\langchain_core\runnables\base.py:3565, in RunnableParallel.__init__(self, steps__, **kwargs)
   3562 merged = {**steps__} if steps__ is not None else {}
   3563 merged.update(kwargs)
   3564 super().__init__(  # type: ignore[call-arg]
-> 3565     steps__={key: coerce_to_runnable(r) for key, r in merged.items()}
   3566 )

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\langchain_core\runnables\base.py:5915, in coerce_to_runnable(thing)
   5910     return cast("Runnable[Input, Output]", RunnableParallel(thing))
   5911 msg = (
   5912     f"Expected a Runnable, callable or dict."
   5913     f"Instead got an unsupported type: {type(thing)}"
   5914 )
-> 5915 raise TypeError(msg)
```