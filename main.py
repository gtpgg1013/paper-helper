import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")

if __name__ == "__main__":
    # pdf_path = "paper-helper/1706.03762.pdf"
    pdf_path = "paper-helper/2310.16944v1.pdf"
    pdf_name = os.path.basename(pdf_path)
    index_name = f"{pdf_name}_faiss_index"

    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_TOKEN)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_name)

    new_vectorstore = FAISS.load_local(index_name, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(max_tokens=1024, openai_api_key=OPENAI_API_TOKEN),
        chain_type="stuff",
        retriever=new_vectorstore.as_retriever(),
    )
    # res = qa.run("Give me the gist of ReAct in 3 sentences")

    res = qa.run(
        """아래 조건과 형식으로 ZEPHYR의 METHOD와 수식에 대해 설명해줘.
1. retriever를 사용할 것
2. 참고한 문서와 페이지 수를 포함할 것
3. 5줄 이상으로 자세하게 설명할 것

###형식 : {참고한 문서와 페이지 수}, {설명 주제}
{내용}
    
###설명 : """
    )

    print(res)
