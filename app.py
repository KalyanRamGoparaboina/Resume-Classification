import streamlit as st
import pandas as pd
import pickle
import re
from pathlib import Path
import docx
import PyPDF2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

st.set_page_config(page_title="Resume Classifier AI", layout="wide")

@st.cache_resource
def load_resources():
    model_path = Path("model.pkl")
    vectorizer_path = Path("tfidf.pkl")
    le_path = Path("label_encoder.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    return model, tfidf, le

def extract_text(file):
    suffix = Path(file.name).suffix.lower()
    text = ""
    try:
        if suffix == ".docx":
            doc = docx.Document(file)
            text = " ".join([p.text for p in doc.paragraphs])
        elif suffix == ".pdf":
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif suffix == ".txt":
            text = file.read().decode("utf-8")
    except Exception as e:
        return f"Error: {e}"
    return text.strip()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    st.title("📄 Resume Intelligence & Multi-Model Analytics")
    
    tabs = st.tabs(["Dashboard & EDA", "Model Performance", "Resume Classifier", "Resume Ranker"])
    
    # Resources
    model, tfidf, le = load_resources()
    
    # 1. Dashboard & EDA Tab
    with tabs[0]:
        st.header("Exploratory Data Analysis (EDA)")
        if Path("processed_resumes.csv").exists():
            df = pd.read_csv("processed_resumes.csv")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Category Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df, x='category', palette='viridis', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
            with col2:
                st.subheader("Resume Word Count Analysis")
                df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df['word_count'], bins=15, kde=True, color='blue', ax=ax)
                st.pyplot(fig)
            
            st.subheader("Top Technical Keywords Across All Resumes")
            all_text = " ".join(df['cleaned_text'].astype(str))
            words = [w for w in all_text.split() if len(w) > 3] # Filter small words
            common_words = Counter(words).most_common(20)
            word_df = pd.DataFrame(common_words, columns=['Keyword', 'Count'])
            
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.barplot(data=word_df, x='Keyword', y='Count', palette='magma', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Training data not found.")

    # 2. Model Performance Tab
    with tabs[1]:
        st.header("Multi-Model Performance Metrics")
        if Path("model_performance.csv").exists():
            perf_df = pd.read_csv("model_performance.csv")
            
            # Identify Best Model (sorting by F1 Score then Accuracy)
            best_row = perf_df.sort_values(by=["F1 Score", "Accuracy"], ascending=False).iloc[0]
            st.success(f"🏆 **Top Performer: {best_row['Model']}** (F1 Score: {best_row['F1 Score']:.2f})")
            
            # Metric Selection
            metric = st.selectbox("Select Metric for Comparison", ["Accuracy", "Precision", "Recall", "F1 Score"])
            
            st.subheader(f"Model Comparison: {metric}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=perf_df, x='Model', y=metric, palette='coolwarm', ax=ax)
            plt.ylim(0, 1.1)
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.2f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')
            st.pyplot(fig)
            
            st.subheader("Detailed Metrics Table")
            st.table(perf_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score"]))
        else:
            st.warning("Performance metrics not found.")

    # 3. Resume Classifier Tab
    with tabs[2]:
        st.header("Bulk Resume Classification")
        st.info("Upload multiple resumes to classify them into categories automatically.")
        uploaded_files = st.file_uploader("Upload resumes (Multiple supported)...", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="bulk")
        
        if uploaded_files:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing: {file.name} ({i+1}/{len(uploaded_files)})")
                text = extract_text(file)
                cleaned = clean_text(text)
                features = tfidf.transform([cleaned])
                pred_encoded = model.predict(features)[0]
                prediction = le.inverse_transform([pred_encoded])[0]
                
                conf = 0
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    conf = max(probs)
                
                results.append({
                    "Filename": file.name,
                    "Predicted Category": prediction,
                    "Confidence": f"{conf:.2%}" if conf > 0 else "N/A"
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.success(f"✅ Successfully processed {len(uploaded_files)} resumes.")
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            
            # Allow downloading results
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Classification Report (CSV)",
                data=csv,
                file_name="resume_classification_report.csv",
                mime="text/csv",
            )
            
            # Individual details
            with st.expander("View Individual Detailed Reports"):
                selected_file = st.selectbox("Select a resume to view details", [f.name for f in uploaded_files])
                file_idx = [f.name for f in uploaded_files].index(selected_file)
                target_file = uploaded_files[file_idx]
                
                # Re-run for individual view (cached or simple)
                text = extract_text(target_file)
                cleaned = clean_text(text)
                features = tfidf.transform([cleaned])
                pred_encoded = model.predict(features)[0]
                prediction = le.inverse_transform([pred_encoded])[0]
                
                st.write(f"### Details for: {selected_file}")
                st.write(f"**Predicted Category:** {prediction}")
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    prob_df = pd.DataFrame({
                        "Category": le.classes_,
                        "Confidence": probs
                    }).sort_values(by="Confidence", ascending=False)
                    st.dataframe(prob_df, use_container_width=True)
                
                st.write("**Extracted Text Preview:**")
                st.text(text[:1000] + "..." if len(text) > 1000 else text)

    # 4. Resume Ranker Tab
    with tabs[3]:
        st.header("AI Ranker & Shortlisting")
        jd_text = st.text_area("Paste Job Description / Key Skills", height=150)
        
        st.subheader("Rank Resumes")
        rank_source = st.radio("Select Resume Source", ["Upload New Resumes", "Use Existing Database"])
        
        if rank_source == "Upload New Resumes":
            rank_files = st.file_uploader("Upload resumes to rank...", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="ranker")
            if st.button("Rank Uploaded Resumes"):
                if not jd_text:
                    st.error("Please provide a Job Description.")
                elif not rank_files:
                    st.error("Please upload resumes to rank.")
                else:
                    rank_results = []
                    jd_vec = tfidf.transform([clean_text(jd_text)])
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, f in enumerate(rank_files):
                        status_text.text(f"Ranking: {f.name} ({i+1}/{len(rank_files)})")
                        text = extract_text(f)
                        cleaned = clean_text(text)
                        vec = tfidf.transform([cleaned])
                        score = cosine_similarity(jd_vec, vec)[0][0]
                        
                        # Also predict category for extra info
                        pred_encoded = model.predict(vec)[0]
                        prediction = le.inverse_transform([pred_encoded])[0]
                        
                        rank_results.append({
                            "Filename": f.name,
                            "Category": prediction,
                            "Match Score": (score * 100).round(2)
                        })
                        progress_bar.progress((i + 1) / len(rank_files))
                    
                    status_text.success(f"✅ Successfully ranked {len(rank_files)} resumes.")
                    rank_df = pd.DataFrame(rank_results).sort_values(by="Match Score", ascending=False).reset_index(drop=True)
                    rank_df.index += 1 # 1st, 2nd, 3rd...
                    
                    # --- NEW TOP 3 PODIUM VIEW ---
                    st.divider()
                    st.header("🏆 Top Talent Podium")
                    
                    top_cols = st.columns(min(3, len(rank_df)))
                    medals = ["🥇", "🥈", "🥉"]
                    
                    for i in range(min(3, len(rank_df))):
                        with top_cols[i]:
                            candidate = rank_df.iloc[i]
                            st.markdown(f"""
                            <div style="border:1px solid #ddd; padding:20px; border-radius:15px; text-align:center; background-color: rgba(255, 255, 255, 0.05);">
                                <h1 style="margin:0;">{medals[i]}</h1>
                                <h3 style="margin:10px 0;">{candidate['Filename'][:20]}...</h3>
                                <div style="font-size: 24px; font-weight: bold; color: #00FF00;">{candidate['Match Score']}%</div>
                                <div style="color: #888;">{candidate['Category']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.divider()
                    st.subheader("📋 Complete Ranked Shortlist (Highest to Lowest)")
                    
                    # Displaying the dataframe with a highlight for the top row
                    def highlight_top(s):
                        return ['background-color: rgba(0, 255, 0, 0.1)' if s.name == 1 else '' for _ in s]
                    
                    st.dataframe(
                        rank_df.style.background_gradient(cmap="Greens", subset=["Match Score"])
                        .format({"Match Score": "{:.2f}%"}),
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = rank_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Shortlisted Ranking", csv, "shortlisted_candidates.csv", "text/csv")
        
        else:
            if Path("processed_resumes.csv").exists():
                df = pd.read_csv("processed_resumes.csv")
                if st.button("Calculate Best Candidates from Database"):
                    if jd_text:
                        jd_vec = tfidf.transform([clean_text(jd_text)])
                        resume_vecs = tfidf.transform(df['cleaned_text'].astype(str))
                        
                        scores = cosine_similarity(jd_vec, resume_vecs).flatten()
                        df['Match Score'] = (scores * 100).round(2)
                        
                        ranked_df = df[['file_name', 'category', 'Match Score']].sort_values(by='Match Score', ascending=False).reset_index(drop=True)
                        ranked_df.index += 1
                        
                        st.divider()
                        st.header("🏆 Database Top Matches")
                        
                        top_cols_db = st.columns(min(3, len(ranked_df)))
                        for i in range(min(3, len(ranked_df))):
                            with top_cols_db[i]:
                                candidate = ranked_df.iloc[i]
                                st.markdown(f"""
                                <div style="border:1px solid #ddd; padding:20px; border-radius:15px; text-align:center; background-color: rgba(255, 255, 255, 0.05);">
                                    <h1 style="margin:0;">{["🥇", "🥈", "🥉"][i]}</h1>
                                    <h3 style="margin:10px 0;">{candidate['file_name'][:20]}...</h3>
                                    <div style="font-size: 24px; font-weight: bold; color: #00FF00;">{candidate['Match Score']}%</div>
                                    <div style="color: #888;">{candidate['category']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.divider()
                        st.subheader("📋 Full Database Ranking")
                        st.dataframe(
                            ranked_df.style.background_gradient(cmap="Blues", subset=["Match Score"]),
                            use_container_width=True
                        )
                    else:
                        st.error("Please provide a Job Description.")
            else:
                st.warning("Database (processed_resumes.csv) not found.")

if __name__ == "__main__":
    main()
