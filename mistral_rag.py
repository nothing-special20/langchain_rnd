# python3 -m pip install transformers

# pytorch for CPU
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# python3 -m pip install langchain
# python3 -m pip install nest-asyncio
# python3 -m pip install python-dotenv
# python3 -m pip install playwright
# python3 -m playwright install
# python3 -m playwright install-deps
# python3 -m pip install html2text
# python3 -m pip install sentence-transformers
# python3 -m pip install faiss-cpu
# python3 -m pip install accelerate
# python3 -m pip install optimum

import copy
import datetime
import nest_asyncio

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.docstore.document import Document

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# read the environment variables from the .env file
from dotenv import dotenv_values

# print(load_dotenv())
config = dotenv_values(".env")
HUGGINGFACEHUB_API_TOKEN = config["HUGGINGFACEHUB_API_TOKEN"]

#################################################################
# Tokenizer
#################################################################

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

device_type = "cpu"

model_config = transformers.AutoConfig.from_pretrained(
    model_name,
    device_map=device_type,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

prompt_template = """
### [INST] 
Instruction: Answer the question based on your 
fantasy football knowledge. Here is context to help:

{context}

### QUESTION:
{question} 

[/INST]
 """

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map="auto",
)
model.to_bettertransformer()


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


print(print_number_of_trainable_model_parameters(model))

text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

nest_asyncio.apply()

# Articles to index
articles = [
    "https://www.fantasypros.com/2023/11/rival-fantasy-nfl-week-10/",
    "https://www.fantasypros.com/2023/11/5-stats-to-know-before-setting-your-fantasy-lineup-week-10/",
    "https://www.fantasypros.com/2023/11/nfl-week-10-sleeper-picks-player-predictions-2023/",
    "https://www.fantasypros.com/2023/11/nfl-dfs-week-10-stacking-advice-picks-2023-fantasy-football/",
    "https://www.fantasypros.com/2023/11/players-to-buy-low-sell-high-trade-advice-2023-fantasy-football/",
]

# Scrapes the blogs above
loader = AsyncChromiumLoader(articles)
docs = loader.load()

docs_copy = copy.deepcopy(docs)

# Converts HTML to plain text
html2text = Html2TextTransformer()

_docs_transformed = html2text.transform_documents(docs_copy)

docs_transformed = []

for doc in _docs_transformed:
    try:
        metadata = copy.deepcopy(doc.metadata)
        page_content = copy.deepcopy(doc.page_content)
        page_content = page_content.split("* Apps")[1]
        page_content = page_content.replace(".\n\n", "\n~\n")
        page_content = page_content.replace("\n\n", " ")
        page_content = page_content.replace("\n~\n", "\n\n")
        # replace individual newline characters with a space, but leave double newlines or more alone
        # doc.page_content = re.sub('\n{1,99}', ' ', doc.page_content)
        doc = Document(page_content=page_content, metadata=metadata)
        docs_transformed.append(doc)
    except:
        print("error")

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(
    chunked_documents,
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
)

retriever = db.as_retriever()

start_time = datetime.datetime.now()

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

response = rag_chain.invoke("Who is CeeDee Lamb?")

print(response["text"])

print("Total time taken:\t", datetime.datetime.now() - start_time)

print("")
