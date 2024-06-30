from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from prompt_template import memory_prompt_template
from langchain.prompts import PromptTemplate
from langchain_community.llms.ctransformers import CTransformers
import yaml
from langchain.vectorstores import Chroma
import chromadb

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def create_llm(model_path=config['model_path'], model_type=config["model_type"], model_config=config['model_config']):
    llm = CTransformers(
        model=model_path, model_type=model_type, config=model_config)
    return llm


def create_embeddings(embeddings_path=config['embeddings_path']):
    return HuggingFaceEmbeddings(model_name=embeddings_path)


def load_vectordb(embeddings):

    persistent_client = chromadb.PersistentClient("chroma_db")

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="collection_name",
        embedding_function=embeddings,
    )

    return langchain_chroma


def load_pdf_chat_chain(chat_history):
    return pdfChatChain(chat_history)


def load_retrieval_chain(llm, memory, vector_db):
    return RetrievalQA.from_llm(llm=llm, memory=memory, retriever=vector_db.as_retriever())


class pdfChatChain:

    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)
        self.vector_db = load_vectordb(create_embeddings())

        llm = create_llm()
        self.llm_chain = load_retrieval_chain(llm, self.memory, self.vector_db)

    def run(self, user_input):
        return self.llm_chain.run(query=user_input, history=self.memory.chat_memory.messages, stop=["Human:"])


def create_chat_memory(chat_historyhuman):
    return ConversationBufferWindowMemory(memory_key='history', chat_memory=chat_historyhuman, k=4)


def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)


def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

    # return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)


class chatChain:
    def __init__(self, chat_history, model_path=config['model_path'], model_type=config['model_type']):
        self.memory = create_chat_memory(chat_history)
        llm = create_llm(model_path=model_path, model_type=model_type)
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)

    def run(self, user_input):
        return self.llm_chain.run(human_input=user_input, history=self.memory.chat_memory.messages, stop="Human:")


def load_normal_chain(chat_history):
    return chatChain(chat_history)
