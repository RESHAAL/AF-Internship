# 🤖 CSR Gujarat AI Assistant

An AI-powered CSR Intelligence Assistant built using **Python, Streamlit, Gemini AI, TF-IDF Semantic Search, and RapidFuzz** to help NGOs, researchers, and CSR professionals explore CSR opportunities and company information across Gujarat.

## 📌 Project Overview

Finding relevant CSR partners and initiatives from large datasets can be time-consuming. This project simplifies the process by providing a conversational AI assistant that can answer questions about companies, CSR sectors, initiatives, directors, CSR heads, and locations using Gujarat CSR data.

The chatbot combines semantic search, typo-tolerant matching, and Gemini AI to deliver accurate and user-friendly responses.

---

## 🚀 Features

### 🔍 Smart Search

* Search companies using natural language.
* Semantic search using TF-IDF and Cosine Similarity.
* Retrieves the most relevant companies based on user queries.

### ✨ Typo-Tolerant Queries

The chatbot can understand misspelled inputs such as:

* Adnai → Adani
* Ahemdabad → Ahmedabad
* Helthcare → Healthcare

### 🤖 Gemini AI Integration

* Generates professional summaries from retrieved CSR records.
* Provides human-readable responses instead of raw database entries.

### 📋 Company Information Cards

Displays:

* Company Name
* Sector
* City
* CSR Areas
* CSR Initiatives
* Director Information
* CSR Head Information
* Website

### 📊 Analytics Dashboard

Interactive dashboard showing:

* Company distribution by city
* Sector distribution
* Top CSR focus areas
* Dataset statistics

### 💡 Suggested Questions

Predefined questions help users quickly explore the dataset.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Google Gemini API
* Pandas
* Scikit-Learn
* RapidFuzz
* Plotly
* OpenPyXL

---

## 📂 Dataset

This project uses CSR company information from the Gujarat dataset containing:

* Company Details
* Contact Information
* CSR Sectors
* CSR Initiatives
* Director Details
* CSR Head Details
* Location Information

Only Gujarat data is used for chatbot responses.

---

## 📁 Project Structure

```text
CSR_GUJARAT_AI/
│
├── app.py
├── Gujarat.xlsx
├── requirements.txt
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-link>
cd CSR_GUJARAT_AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add Gemini API Key

Open `app.py` and replace:

```python
API_KEY = "YOUR_API_KEY"
```

with your Gemini API key.

---

## ▶️ Run Application

```bash
python -m streamlit run app.py
```

The application will open automatically in your browser.

---

## 💬 Example Queries

* Companies working in education
* Healthcare CSR initiatives
* Companies in Ahmedabad
* Women empowerment projects
* Renewable energy companies
* Show sustainability initiatives
* CSR Head of Adani Green Energy
* Companies working in rural development

---

## 🧠 How It Works

1. User submits a query.
2. TF-IDF semantic search identifies relevant records.
3. RapidFuzz handles spelling mistakes and fuzzy matching.
4. Top matching companies are retrieved.
5. Gemini AI generates a contextual summary.
6. Results are displayed with detailed company information.

This approach follows a Retrieval-Augmented Generation (RAG) workflow.

---

## 🎯 Future Improvements

* Multi-state CSR support
* PDF report generation
* Voice-based interaction
* NGO recommendation engine
* CSR opportunity matching
* Advanced filtering options

---

## 👩‍💻 Author

Developed as part of a Technical Internship project focused on building AI-powered tools for CSR data exploration and NGO collaboration.
