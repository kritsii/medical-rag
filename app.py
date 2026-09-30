import os
import tomllib
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv


REQUEST_TIMEOUT_SECONDS = 120
SECRETS_FILE = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
REQUIRED_SETTINGS = ("GROQ_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")
REQUEST_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def configure_environment() -> list[str]:
    load_dotenv()
    if SECRETS_FILE.is_file():
        with SECRETS_FILE.open("rb") as secrets_file:
            secrets = tomllib.load(secrets_file)
        for key in REQUIRED_SETTINGS + ("GROQ_MODEL",):
            value = secrets.get(key)
            if value and not os.getenv(key):
                os.environ[key] = str(value)

    if not os.getenv("PINECONE_INDEX_NAME"):
        os.environ["PINECONE_INDEX_NAME"] = "diabetes-rag"
    return [key for key in REQUIRED_SETTINGS[:2] if not os.getenv(key)]


def run_research(query: str) -> tuple[list[dict], dict | None]:
    from retrieval_agent import retrieve
    from synthesis_agent import synthesize

    chunks = retrieve(query)
    result = synthesize(query, chunks) if chunks else None
    return chunks, result


def pubmed_url(chunk: dict) -> str:
    pmid = str(chunk.get("pmid", "")).strip()
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{quote_plus(pmid)}/"

    search_term = str(chunk.get("doi") or chunk.get("title") or "").strip()
    return f"https://pubmed.ncbi.nlm.nih.gov/?term={quote_plus(search_term)}"


st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --ink: #f4f8fb;
        --muted: #b8c7d4;
        --brand: #66cdd7;
        --brand-dark: #14344a;
        --wash: #0b1726;
        --line: #294158;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: #0b1726;
        color: #f4f8fb;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #102a43;
        border-right: 0;
    }
    [data-testid="stAppViewContainer"] [data-testid="stMain"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMain"] li,
    [data-testid="stAppViewContainer"] [data-testid="stMain"] label,
    [data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stMain"] .stMarkdown {
        color: #e5edf4;
    }
    [data-testid="stSidebar"] * { color: #e8f3f5; }
    [data-testid="stSidebar"] .stButton button {
        text-align: left;
        background: transparent;
        border: 1px solid rgba(220, 241, 244, .16);
        color: #e8f3f5;
        border-radius: 10px;
        padding: .55rem .7rem;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(102, 205, 215, .16);
        border-color: #66cdd7;
    }
    .brand-mark {
        display: flex; align-items: center; gap: 13px; margin: .4rem 0 2.1rem;
    }
    .brand-icon {
        width: 42px; height: 42px; display: grid; place-items: center;
        border-radius: 13px; background: linear-gradient(145deg, #66cdd7, #16839a);
        color: white; font-size: 24px; box-shadow: 0 8px 20px rgba(0,0,0,.16);
    }
    .brand-name { font: 800 1.03rem 'Manrope', sans-serif; letter-spacing: -.02em; }
    .brand-subtitle { color: #9fc3ca; font-size: .72rem; margin-top: 2px; }
    .hero h1 { color: #f4f8fb; font: 800 2.35rem 'Manrope', sans-serif; letter-spacing: -.045em; margin-bottom: .3rem; }
    .hero p { color: var(--muted); font-size: 1rem; margin-top: 0; }
    .eyebrow { color: var(--brand); font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .query-card, [data-testid="stForm"] {
        background: #162b43; border: 1px solid #284761;
        border-radius: 18px; padding: 1.1rem 1.25rem .85rem;
        box-shadow: 0 10px 28px rgba(7, 24, 41, .16);
    }
    [data-testid="stTextInput"] label { color: #f4f8fb; font-weight: 600; }
    [data-testid="stTextInput"] input {
        border: 1px solid #41637b; border-radius: 10px; background: #0f2034;
        color: #ffffff; caret-color: #66cdd7; padding: .72rem .85rem;
        box-shadow: none;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextInput"] input:focus-visible {
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
    [data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within {
        border: 1px solid #66cdd7 !important;
        box-shadow: 0 0 0 1px #66cdd7 !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within > div,
    [data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within > div {
        border-color: #66cdd7 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: #b6c6d2; opacity: 1;
    }
    [data-testid="stForm"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stBaseButton-primary"] {
        background: #16839a !important;
        background-image: none !important;
        border: 1px solid #66cdd7 !important;
        color: #fff !important;
        font-weight: 700;
    }
    [data-testid="stForm"] button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: #0f6f82 !important;
        background-image: none !important;
        border-color: #a0e9ee !important;
        color: #fff !important;
    }
    .section-title { color: #f4f8fb; font: 700 1.28rem 'Manrope', sans-serif; margin-top: 1.8rem; }
    [data-testid="stExpander"] {
        background: #14263a; border: 1px solid #294158; border-radius: 12px;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        color: #e5edf4;
    }
    [data-testid="stExpander"] a, [data-testid="stMain"] a { color: #78d5df; }
    [data-testid="stAlert"] { background: #14263a; color: #f4f8fb; }
    .st-key-answer_panel {
        background: #14263a; border: 1px solid #294158; border-radius: 16px;
        padding: 1rem 1.35rem; margin: .5rem 0 1.2rem;
        box-shadow: 0 10px 28px rgba(0,0,0,.15);
    }
    .disclaimer { color: #9fb0bf; font-size: .8rem; margin-top: 2.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">
          <div class="brand-icon">✚</div>
          <div><div class="brand-name">MedResearch</div><div class="brand-subtitle">Evidence, made clearer</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Chat history")
    if st.session_state["history"]:
        for index, item in enumerate(reversed(st.session_state["history"])):
            label = item["query"]
            if len(label) > 38:
                label = label[:35] + "..."
            if st.button(label, key=f"history_{item['id']}", use_container_width=True):
                st.session_state["query_input"] = item["query"]
                st.session_state["research_result"] = item["result"]
                st.session_state["research_error"] = None
                st.rerun()
        if st.button("Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.session_state.pop("research_result", None)
            st.rerun()
    else:
        st.caption("Your recent research questions will appear here.")
    st.markdown("---")
    st.caption("Research aid only — not a substitute for professional medical advice.")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Medical literature companion</div>
      <h1>Ask better questions.<br>Find stronger evidence.</h1>
      <p>Search indexed research papers and get a cited, easy-to-read synthesis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

missing_settings = configure_environment()
if missing_settings:
    st.error(
        "Missing required configuration: "
        + ", ".join(missing_settings)
        + ". Add them to your local .env or .streamlit/secrets.toml file, "
        "or configure them as Railway environment variables."
    )
    st.stop()

with st.form("research_query"):
    query = st.text_input(
        "Research question",
        key="query_input",
        placeholder="e.g. What are risk factors for diabetes?",
    )
    submitted = st.form_submit_button("Search papers and generate answer", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a research question to continue.")
    else:
        try:
            with st.spinner("Retrieving papers and synthesizing an answer..."):
                future = REQUEST_EXECUTOR.submit(run_research, query.strip())
                chunks, result = future.result(timeout=REQUEST_TIMEOUT_SECONDS)
            st.session_state["research_result"] = (chunks, result)
            st.session_state["research_error"] = None
            st.session_state["history"] = [
                item for item in st.session_state["history"] if item["query"] != query.strip()
            ]
            st.session_state["history"].append(
                {
                    "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
                    "query": query.strip(),
                    "result": (chunks, result),
                }
            )
        except TimeoutError:
            st.session_state["research_result"] = None
            st.session_state["research_error"] = (
                f"The request took longer than {REQUEST_TIMEOUT_SECONDS} seconds. "
                "Please try again."
            )
        except Exception as exc:
            st.session_state["research_result"] = None
            st.session_state["research_error"] = (
                f"The research request failed ({type(exc).__name__}): {exc}"
            )

if st.session_state.get("research_error"):
    st.error(st.session_state["research_error"])

research_result = st.session_state.get("research_result")
if research_result is not None:
    chunks, result = research_result
    if not chunks:
        st.info("No relevant papers were found. Try rephrasing your question.")
    else:
        with st.container(key="answer_panel"):
            st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
            if result:
                st.markdown(result["answer"])
            else:
                st.info("Papers were retrieved, but no answer could be generated.")

        st.markdown(
            f'<div class="section-title">Retrieved papers <span style="color:#66cdd7">({len(chunks)})</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Citation numbers match the [1], [2], etc. references in the answer.")
        for index, chunk in enumerate(chunks, start=1):
            title = chunk.get("title") or "Untitled paper"
            doi = str(chunk.get("doi") or "").strip()
            with st.expander(f"[{index}] {title}"):
                if doi and doi.upper() != "N/A":
                    st.markdown(f"**DOI:** [{doi}](https://doi.org/{quote_plus(doi, safe='/')})")
                else:
                    st.write("**DOI:** Not available")
                st.markdown(f"**PubMed:** [View paper]({pubmed_url(chunk)})")
                if chunk.get("score") is not None:
                    st.caption(f"Retrieval score: {chunk['score']:.3f}")
                if chunk.get("text"):
                    st.write(chunk["text"])

st.markdown(
    '<div class="disclaimer">Always verify important findings with the original publication and a qualified healthcare professional.</div>',
    unsafe_allow_html=True,
)
