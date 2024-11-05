# PDF-chatbot

# PDF Chatbot with AI-Powered Q&A

This project is a **PDF-based chatbot** that allows users to upload a PDF document and interact with it through AI-powered Q&A. By utilizing Google’s Generative AI API and LangChain’s document processing capabilities, this app responds to user questions by retrieving relevant information from the PDF and generating answers based on the content.

### Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Setup and Requirements](#setup-and-requirements)
4. [How It Works](#how-it-works)
5. [Usage](#usage)
6. [Future Enhancements](#future-enhancements)

---

### Project Overview

The **PDF Chatbot** is an interactive application where users can upload a PDF file and ask questions about its content. The app extracts and processes the text within the PDF, generates embeddings, and stores them in a FAISS vector database. When a question is asked, it finds relevant document sections and uses Google Generative AI to generate a context-aware response.

### Features

- **PDF Upload and Processing:** Users can upload PDF files, which are then parsed into text data.
- **Contextual Q&A:** Finds relevant sections within the document and provides accurate, context-driven responses.
- **Real-time AI Interaction:** Utilizes Google Generative AI for natural language understanding and response generation.
- **User-Friendly Interface:** Built with Streamlit, offering an intuitive and easy-to-use chatbot experience.

### Setup and Requirements

To run this project, ensure you have the following dependencies installed:

```python
!pip install streamlit google-generativeai langchain-google-genai transformers torch faiss-cpu
```

### Environment Variables

Create an `.env` file with your Google API key for secure access:

```plaintext
GOOGLE_API_KEY="your-google-api-key-here"
```

### How It Works

1. **PDF Processing:** After the PDF is uploaded, the `process_pdf` function reads the document, splits it into pages, and generates embeddings using Hugging Face's `all-MiniLM-L6-v2` model.
2. **Vector Storage with FAISS:** The embeddings are stored in a FAISS vector database, allowing for efficient similarity search.
3. **Question-Answer Interaction:** When the user asks a question, relevant paragraphs from the PDF are retrieved, forming a context that is passed to Google Generative AI.
4. **AI Response Generation:** The Google Generative AI API processes the context and question to generate a coherent response.

### Usage

1. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

2. **Upload a PDF:** Use the file uploader to select a PDF for analysis.
3. **Ask a Question:** Enter your question about the document content, and the chatbot will respond with contextually relevant information.
4. **Review Relevant Content:** The app also displays paragraphs from the PDF that are relevant to the question.

### Future Enhancements

- **Support for Additional File Types:** Add compatibility for DOCX and TXT files.
- **Enhanced Embedding Models:** Experiment with larger, more accurate embedding models.
- **Multi-Language Support:** Enable language translation to make the app accessible to a broader audience.
