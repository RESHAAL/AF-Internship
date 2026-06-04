# CSR GUJARAT AI ASSISTANT
# PART 1 - IMPORTS + DATA LOADING + SEARCH ENGINE

import streamlit as st
import pandas as pd
import numpy as np
from google import genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import fuzz

import plotly.express as px

# GEMINI CONFIGURATION

API_KEY = "AQ.Ab8xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

client = genai.Client(api_key=API_KEY)

# STREAMLIT PAGE CONFIG

st.set_page_config(
    page_title="CSR Gujarat AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# LOAD DATA

@st.cache_data
def load_data():

    df = pd.read_excel("Gujarat.xlsx")

    # Remove unwanted columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Replace NaN
    df = df.fillna("")

    return df


df = load_data()

# CREATE SEARCHABLE TEXT

@st.cache_data
def create_search_text(df):

    combined_text = []

    for _, row in df.iterrows():

        text = " ".join([
            str(row["Name of Company"]),
            str(row["Sector"]),
            str(row["Contact"]),
            str(row["Email Id"]),
            str(row["Address"]),
            str(row["City"]),
            str(row["Website"]),
            str(row["CSR Sectors"]),
            str(row["Types of CSR Initiatives"]),
            str(row["Director's Name"]),
            str(row["Director's Contact"]),
            str(row["Director's Email"]),
            str(row["CSR Head Name"]),
            str(row["CSR Head Contact"]),
            str(row["CSR Head Email"]),
            str(row["Comments"])
        ])

        combined_text.append(text)

    return combined_text


search_text = create_search_text(df)

# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(search_text)

# FUZZY SEARCH

def fuzzy_search(query, top_k=10):

    scores = []

    query = query.lower()

    for idx, text in enumerate(search_text):

        score = fuzz.token_sort_ratio(
            query,
            text.lower()
        )

        scores.append((idx, score))

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:top_k]

# SEMANTIC SEARCH USING TF-IDF

def semantic_search(query, top_k=10):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for idx in top_indices:

        results.append(
            (
                idx,
                similarities[idx]
            )
        )

    return results

# HYBRID SEARCH
# TF-IDF + RAPIDFUZZ

def hybrid_search(query, top_k=10):

    semantic_results = semantic_search(query, top_k=20)

    fuzzy_results = fuzzy_search(query, top_k=20)

    scores = {}

    # Semantic score weight
    for idx, score in semantic_results:

        scores[idx] = scores.get(idx, 0) + score * 70

    # Fuzzy score weight
    for idx, score in fuzzy_results:

        scores[idx] = scores.get(idx, 0) + score * 0.30

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_matches = ranked[:top_k]

    matched_rows = []

    for idx, score in top_matches:

        row = df.iloc[idx].to_dict()

        row["match_score"] = round(score, 2)

        matched_rows.append(row)

    return matched_rows

# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

# PART 2 - GEMINI RAG ENGINE

def build_context(results):

    context = ""

    for i, row in enumerate(results, start=1):

        context += f"""

Company {i}

Name: {row.get('Name of Company','')}

Sector: {row.get('Sector','')}

City: {row.get('City','')}

Address: {row.get('Address','')}

Website: {row.get('Website','')}

CSR Sectors: {row.get('CSR Sectors','')}

CSR Initiatives: {row.get('Types of CSR Initiatives','')}

Director Name: {row.get("Director's Name",'')}

Director Contact: {row.get("Director's Contact",'')}

Director Email: {row.get("Director's Email", '')}

CSR Head Name: {row.get('CSR Head Name','')}

CSR Head Contact: {row.get('CSR Head Contact','')}

CSR Head Email: {row.get('CSR Head Email','')}

Comments: {row.get('Comments','')}

---------------------------------------------------

"""

    return context


# GEMINI RESPONSE GENERATION

def generate_ai_response(user_query, search_results):

    context = build_context(search_results)

    prompt = f"""
You are an expert CSR Assistant.

You MUST answer ONLY using the information provided below.

Rules:

1. Use only Gujarat CSR dataset information.
2. Do not invent companies.
3. Do not hallucinate data.
4. If information is unavailable, say so.
5. Give a concise professional summary.
6. Mention key companies involved.
7. Mention common CSR focus areas if relevant.

CSR DATA:

{context}

USER QUESTION:

{user_query}

Provide a professional NGO-style response.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"""
Unable to generate AI response.

Error:
{str(e)}
"""


# COMPLETE SEARCH PIPELINE

def get_answer(user_query):

    results = hybrid_search(
        user_query,
        top_k=10
    )

    ai_answer = generate_ai_response(
        user_query,
        results
    )

    return ai_answer, results


# COMPANY CARD DISPLAY

def display_company_card(company):

    company_name = company.get(
        "Name of Company",
        "Not Available"
    )

    city = company.get(
        "City",
        "Not Available"
    )

    sector = company.get(
        "Sector",
        "Not Available"
    )

    csr_sector = company.get(
        "CSR Sectors",
        "Not Available"
    )

    initiatives = company.get(
        "Types of CSR Initiatives",
        "Not Available"
    )

    website = company.get(
        "Website",
        "Not Available"
    )

    csr_head = company.get(
        "CSR Head Name",
        "Not Available"
    )

    director = company.get(
        "Director's Name",
        "Not Available"
    )

    score = company.get(
        "match_score",
        0
    )

    st.markdown(
        f"""
### 🏢 {company_name}

**📍 City:** {city}

**🏭 Sector:** {sector}

**🎯 CSR Areas:** {csr_sector}

**💡 Initiatives:** {initiatives}

**👤 Director:** {director}

**🤝 CSR Head:** {csr_head}

**🌐 Website:** {website}

**⭐ Match Score:** {score}
"""
    )

    st.divider()


# TOP RESULT SUMMARY

def display_top_results(results):

    st.subheader(
        "📋 Top Matching Companies"
    )

    for company in results:

        display_company_card(
            company
        )


# QUICK SEARCH HELPER

def search_and_answer(query):

    ai_response, results = get_answer(
        query
    )

    return ai_response, results

# PART 3 - ANALYTICS DASHBOARD

def show_dashboard():

    st.header("📊 CSR Gujarat Analytics Dashboard")

    # KPIs

    total_companies = len(df)

    total_cities = df["City"].nunique()

    total_sectors = df["Sector"].nunique()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🏢 Companies",
            total_companies
        )

    with col2:
        st.metric(
            "📍 Cities",
            total_cities
        )

    with col3:
        st.metric(
            "🏭 Sectors",
            total_sectors
        )

    st.divider()

    # TOP CITIES

    st.subheader("📍 Top Cities")

    city_counts = (
        df["City"]
        .astype(str)
        .value_counts()
        .head(10)
        .reset_index()
    )

    city_counts.columns = [
        "City",
        "Count"
    ]

    fig_city = px.bar(
        city_counts,
        x="City",
        y="Count",
        title="Top 10 Cities"
    )

    st.plotly_chart(
        fig_city,
        use_container_width=True
    )

    # CSR SECTORS

    st.subheader("🎯 Top CSR Areas")

    csr_text = (
        df["CSR Sectors"]
        .astype(str)
        .str.cat(sep=",")
    )

    sector_list = []

    for item in csr_text.split(","):

        item = item.strip()

        if len(item) > 2:
            sector_list.append(item)

    sector_df = pd.DataFrame(
        sector_list,
        columns=["Sector"]
    )

    sector_counts = (
        sector_df["Sector"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    sector_counts.columns = [
        "CSR Area",
        "Count"
    ]

    fig_sector = px.bar(
        sector_counts,
        x="CSR Area",
        y="Count",
        title="Top CSR Areas"
    )

    st.plotly_chart(
        fig_sector,
        use_container_width=True
    )

    # COMPANY DISTRIBUTION

    st.subheader("🏭 Sector Distribution")

    sector_dist = (
        df["Sector"]
        .astype(str)
        .value_counts()
        .head(10)
        .reset_index()
    )

    sector_dist.columns = [
        "Sector",
        "Count"
    ]

    fig_pie = px.pie(
        sector_dist,
        names="Sector",
        values="Count",
        title="Sector Distribution"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


# SIDEBAR

def sidebar_panel():

    st.sidebar.title(
        "🤖 CSR Gujarat Assistant"
    )

    st.sidebar.markdown(
        "---"
    )

    st.sidebar.subheader(
        "💡 Suggested Questions"
    )

    suggestions = [

        "Companies working in education",

        "Healthcare CSR initiatives",

        "Companies in Ahmedabad",

        "Women empowerment projects",

        "Renewable energy companies",

        "Companies working in environment",

        "Skill development initiatives",

        "CSR head of Adani Green Energy",

        "Companies working in rural development",

        "Top sustainability initiatives"

    ]

    for item in suggestions:

        st.sidebar.markdown(
            f"• {item}"
        )

    st.sidebar.markdown(
        "---"
    )

    st.sidebar.info(
        """
This chatbot answers only from
the Gujarat CSR dataset.

Powered by:
- Gemini 2.5 Flash
- TF-IDF Search
- RapidFuzz
- Streamlit
"""
    )


# DATASET OVERVIEW

def dataset_overview():

    with st.expander(
        "📄 View Gujarat Dataset"
    ):

        st.dataframe(
            df,
            use_container_width=True
        )

# PART 4 - MAIN APPLICATION

# Sidebar
sidebar_panel()

# HEADER

st.title("🤖 CSR Gujarat AI Assistant")

st.markdown("""
Ask questions about:

- Companies
- CSR Activities
- CSR Heads
- Directors
- Education Initiatives
- Healthcare Projects
- Sustainability Programs
- Ahmedabad Companies
- Women Empowerment Programs

The assistant answers ONLY from the Gujarat CSR dataset.
""")

st.divider()

# MODE SELECTION

mode = st.sidebar.radio(
    "Choose View",
    [
        "💬 Chat Assistant",
        "📊 Analytics Dashboard"
    ]
)

# DASHBOARD MODE

if mode == "📊 Analytics Dashboard":

    show_dashboard()

    dataset_overview()

# CHAT MODE

else:

    # Chat History Display

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # User Input

    prompt = st.chat_input(
        "Ask a CSR-related question..."
    )

    if prompt:

        # Store user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        # Display User Message

        with st.chat_message("user"):

            st.markdown(prompt)

        # Assistant Response

        with st.chat_message("assistant"):

            with st.spinner(
                "Analyzing Gujarat CSR data..."
            ):

                try:

                    ai_response, results = search_and_answer(
                        prompt
                    )

                    # Show Gemini Summary

                    st.subheader(
                        "🤖 AI Summary"
                    )

                    st.markdown(
                        ai_response
                    )

                    st.divider()

                    # Show Company Cards

                    display_top_results(
                        results
                    )

                    # Save response to history

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": ai_response
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Error: {str(e)}"
                    )

# FOOTER

st.markdown("---")

st.caption(
    """
CSR Gujarat AI Assistant

Built with:
Gemini 2.5 Flash • Streamlit • TF-IDF • RapidFuzz

Answers are generated only from Gujarat CSR data.
"""
)
