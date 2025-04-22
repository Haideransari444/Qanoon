```
# 🇵🇰 Pakistani Legal Advocate Assistant

A fully terminal-based AI chatbot powered by LangChain and Gemini, trained on Pakistani legal texts including the Constitution, Penal Code, and major court rulings. The assistant provides formal legal guidance using only the provided data with proper legal citations.

---

## 📌 Features

- ✅ Offline-ready FAISS vector store for legal documents  
- ✅ Uses Gemini LLM API via LangChain  
- ✅ Custom structured prompt for strict legal formatting  
- ✅ Pure terminal interface (no frontend needed)  
- ✅ Cites only relevant legal content (e.g., Sections, Articles, Clauses, PLD Cases)  
- ✅ No hallucinations — answers only from loaded documents  
- ✅ Easily extendable with more legal documents  

---

## 📁 Project Structure

.
├── backend/
│   ├── query.py               # Main terminal chatbot script
│   ├── vectordb/              # Folder containing FAISS index files
│   │   ├── index.faiss
│   │   └── index.pkl
├── .env                       # Contains your Gemini API key
├── requirements.txt           # All required Python packages
└── README.md

---

## 🚀 Quickstart

### 1. Clone the repo

git clone https://github.com/your-username/legal-advocate-assistant.git
cd legal-advocate-assistant

### 2. Install dependencies

pip install -r requirements.txt

### 3. Add your Gemini API key

Create a `.env` file in the root directory:

GOOGLE_API_KEY=your-gemini-api-key

### 4. Ensure vector DB exists

Place `index.faiss` and `index.pkl` inside the `backend/vectordb/` folder. If you don’t have them, run a script to build the vector index using LangChain and your legal documents.

### 5. Run the chatbot

cd backend
python query.py

---

## 💬 Prompt Template Used

You are a Pakistani legal expert assistant. Using ONLY the following legal documents, answer the user's question.
If you don't know the answer or the documents don't provide enough information, say so.

When citing legal authorities, use only formal legal citation format including:
- Section numbers (e.g., "Section 34")
- Article numbers (e.g., "Article 15")
- Chapter references (e.g., "Chapter IV")
- Specific clauses (e.g., "Clause 2(a)")
- Case citations (e.g., "PLD 2005 SC 193")
- Statute dates (e.g., "Companies Act, 2017")

DO NOT include document filenames, page numbers, or any reference to the source documents provided to you.

Legal Documents:
{context}

User's Question: {question}

Your answer must:
1. Be clear, concise, and factual
2. Include only formal legal citations as described above
3. NOT make up or infer information not present in the documents
4. Present information in a structured, organized format with headings where appropriate

---

## 📦 Dependencies

- langchain
- faiss-cpu
- python-dotenv
- google-generativeai
- tqdm
- bcrypt (if adding login functionality)

Install with:

pip install -r requirements.txt

---

## 🤝 Contributing

Pull requests and suggestions are welcome. If you find a bug or want to add more legal documents, feel free to fork and contribute.

---

## ⚖️ Disclaimer

This tool is for **educational and informational purposes only.** It does not substitute real legal advice. Always consult a licensed legal practitioner.

---

## 🧠 Built with

- LangChain → https://github.com/langchain-ai/langchain
- Gemini API → https://ai.google.dev
- FAISS → https://github.com/facebookresearch/faiss
```
