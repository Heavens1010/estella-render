from flask import Flask, request, jsonify
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-api-key")
SYSTEM_PROMPT = """You are Estella, a virtual assistant designed to help relief teachers using Utopia Education’s platform..."""

app = Flask(__name__)
vectorstore = FAISS.load_local("faiss_utopia_knowledge", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False
)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "")
    try:
        response = qa_chain.run(user_question)
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Estella Agent is running!", 200

if __name__ == "__main__":
    app.run()
