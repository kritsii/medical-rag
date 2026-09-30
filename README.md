# Medical Research Assistant

This Streamlit app searches the configured Pinecone index for relevant research
paper excerpts and uses Groq to synthesize an answer with numbered references.
It is a research aid, not a substitute for professional medical advice.

## Run locally

1. Use Python 3.12 and create/activate a virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Add `GROQ_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME` to
   `.streamlit/secrets.toml` (or a local `.env` file). The secrets file is
   git-ignored. The index name defaults to `diabetes-rag`.
4. Start the app: `streamlit run app.py`.
5. Open the local URL printed by Streamlit and ask a research question.

The first query may take longer while the FastEmbed model is
downloaded. Requests have a 120-second UI timeout; answer generation also has a
60-second Groq request timeout.

## Railway deployment checklist

- [ ] Push the application code to a GitHub repository. Never commit `.env` or
  `.streamlit/secrets.toml`.
- [ ] In Railway, create a project and deploy from the GitHub repository.
- [ ] Set these service variables in Railway (not in the repository):
  - `GROQ_API_KEY`
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_NAME` (for example, `diabetes-rag`)
  - Optionally, `GROQ_MODEL` (defaults to `qwen/qwen3.8-27b`)
- [ ] Deploy using the included `Procfile`; Railway supplies the `$PORT`.
- [ ] Wait for the deployment to finish, then open the generated public URL.
- [ ] Test a relevant question and verify that papers, inline citation numbers,
  DOI links, and PubMed links are displayed.
- [ ] Test a question with no matching papers and confirm the no-results message.
- [ ] Check Railway logs and verify the app reports missing configuration or
  upstream/API failures clearly if a service variable is incorrect.
