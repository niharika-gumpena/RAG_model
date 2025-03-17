# PDF-Based Conversational AI using LangChain & Google Gemini

This project implements a conversational AI system that processes PDF documents, generates embeddings, and answers questions based on the document's content using Google's Gemini models.

## Features
✅ Extract text from uploaded PDFs  
✅ Chunk text data for efficient processing  
✅ Generate embeddings using Google Generative AI models  
✅ Use FAISS for fast vector search  
✅ Interactive conversational chatbot for Q&A from PDFs  

## Prerequisites
Before running the project, ensure you have the following:
- Python 3.11 or higher
- Required libraries (install using the command below)

```bash
pip install langchain langchain_google_genai faiss-cpu PyPDF2 python-dotenv google-generativeai
```

## Environment Variables
Create a `.env` file in the root directory with the following keys:
```
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Add your API keys in the `.env` file.

4. Run the Python script:
```bash
python main.py
```

5. Upload your desired PDF files when prompted.

## Usage
1. Upload your PDF files.
2. The text will be extracted and chunked for efficient searching.
3. Ask your questions in the terminal.
4. Type `exit` to end the session.

## Example Output
```
📂 Upload your PDF files
✅ PDF processed successfully!
Ask a question (or type 'exit' to quit): What is the main topic of the PDF?
Reply: The document discusses AI advancements in healthcare.
```

## Troubleshooting
- **404 Error for Models:** If you encounter a `404 models not found` error, change the model name in the code to one of the available options:
  - `models/gemini-1.5-pro-latest`
  - `models/gemini-1.5-flash-latest`

- **TypeError for `configure()` method:** Remove the `api_version` argument in the `genai.configure()` method.


## Acknowledgments
- [LangChain](https://python.langchain.com/)
- [Google Generative AI](https://cloud.google.com/generative-ai)
- [FAISS](https://faiss.ai/)

