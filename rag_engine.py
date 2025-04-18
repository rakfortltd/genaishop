import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoModelForSeq2SeqLM

class RAGEngine:
    def __init__(self, pdf_dir="data", embedding_model="sentence-transformers/all-MiniLM-L6-v2", llm_model="mistralai/Mistral-7B-Instruct-v0.1"):
        self.pdf_dir = pdf_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.qa_chain = None

    def load_documents(self):
        documents = []
        for file in os.listdir(self.pdf_dir):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.pdf_dir, file))
                documents.extend(loader.load())
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(documents)

    def load_embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.embedding_model)

    def load_local_llm(self):
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )

        return HuggingFacePipeline(pipeline=pipe)

    def setup(self):
        docs = self.load_documents()
        embeddings = self.load_embeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = self.load_local_llm()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    def ask(self, question: str):
        if not self.qa_chain:
            raise ValueError("RAG pipeline not initialized. Call setup() first.")
        result = self.qa_chain(question)
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "N/A") for doc in result["source_documents"]]
        }
