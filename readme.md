

## Financial Literacy AI Chatbot

An **AI-powered financial literacy assistant** built using **Streamlit** and **Retrieval-Augmented Generation (RAG)**.
It helps users understand personal finance topics like savings, investments, insurance, and budgeting — through intelligent, context-aware chat.

---

### Features

* **Conversational Memory:** The chatbot remembers previous messages, so each response relates to the user’s past queries (like ChatGPT).
* **Retrieval-Augmented Generation (RAG):** Answers are grounded in trusted financial knowledge sources using vector embeddings.
* **Multi-Provider Support:** Works with both **OpenAI GPT** and **Google Gemini** APIs.
* **Interactive UI (Streamlit):** Clean chat interface with history tracking and user input boxes.
* **Secure Configuration:** API keys and environment variables are handled through `.env` and Streamlit secrets.

---

###  Tech Stack

| Component    | Technology Used             |
| ------------ | --------------------------- |
| Frontend     | Streamlit                   |
| Backend      | Python (RAG pipeline)       |
| Embeddings   | OpenAI / Gemini             |
| Vector Store | FAISS or Chroma             |
| Memory       | Streamlit Session State     |
| Deployment   | Streamlit Cloud / Localhost |

---

### Project Structure

```
chatbot/
│
├── src/
│   ├── app.py               # Main Streamlit app
│   ├── rag_pipeline.py      # RAG pipeline logic
│   ├── embeddings.py        # Embedding model utilities
│   ├── vectorstore.py       # Vector database functions
│   ├── verifier.py          # Optional answer verifier
│   └── utils/               # Helper scripts
│
├── data/                    # Knowledge base for financial literacy
├── .env                     # API keys and environment variables
├── requirements.txt          # Dependencies
├── .gitignore
└── README.md
```

---

###  Example Queries

* “What is the difference between a mutual fund and an ETF?”
* “How can I start investing with a low salary?”
* “Explain compound interest with a real-life example.”
* “What’s the safest way to build an emergency fund?”

---

###  Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/financial-literacy-chatbot.git
   cd financial-literacy-chatbot
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API keys**
   Create a `.env` file:

   ```
   OPENAI_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   PROVIDER=openai
   ```

5. **Run the app**

   ```bash
   streamlit run src/app.py
   ```

---

###  Future Improvements

* Add voice input/output for accessibility
* User authentication and personalized dashboards
* Expand dataset with localized finance data
* Integrate charts and financial calculators

---
