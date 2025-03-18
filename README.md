# RAG Chatbot with LangChain, Pinecone, and OpenAI on Streamlit

## Prerequisites
* Python 3.11 to 3.12.9 (DO NOT USE PYTHON 3.13)
* Pinecone account (https://www.pinecone.io/)
* OpenAI Platform account (https://platform.openai.com/)

<h2>Installation</h2>
1. Clone the repository:

```
git clone https://github.com/czhaoyiii/RAG_Chatbot.git
cd RAG_Chatbot
```

2. Create a virtual environment

```
python -m venv venv
```

3. Activate the virtual environment

```
Windows: venv\Scripts\Activate
Mac: source venv/bin/activate
```

4. Install libraries

```
pip install -r requirements.txt
```

5. Create API keys

* Create an API key on Pinecone
* Create an API key for OpenAI

6. Add API keys to .env file

* Rename .env_example to .env
* Add the API keys for Pinecone and OpenAI to the .env file
* Change the PINECONE_INDEX_NAME to a suitable name

<h3>Executing the scripts</h3>

1. Open a terminal in VS Code

2. Execute the following command:

```
python ingestion.py
streamlit run chatbot_rag.py
```
