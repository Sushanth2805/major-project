"""
AI-Powered Multi-Domain Knowledge & Insights Assistant
Built with RAG (Retrieval Augmented Generation) + Machine Learning
Supports: Academic, Finance, and Business Report Analysis
"""

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
import json

# Document processing
import PyPDF2
from docx import Document
import pandas as pd

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Configure page - MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Multi-Domain Assistant",
    page_icon="🤖",
    layout="wide"
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'current_domain' not in st.session_state:
    st.session_state.current_domain = None
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}
if 'last_query_hash' not in st.session_state:
    st.session_state.last_query_hash = None
if 'api_call_count' not in st.session_state:
    st.session_state.api_call_count = 0

# Initialize embedding model (using sentence-transformers)
@st.cache_resource
def load_embedding_model():
    """Load the embedding model for converting text to vectors"""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()


class DocumentProcessor:
    """Handles extraction of text from different file formats"""
    
    @staticmethod
    def extract_from_pdf(file):
        """Extract text from PDF files"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_from_docx(file):
        """Extract text from Word documents"""
        try:
            doc = Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_from_txt(file):
        """Extract text from TXT files"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""
    
    @staticmethod
    def extract_from_csv(file):
        """Extract and process CSV files (for financial data)"""
        try:
            df = pd.read_csv(file)
            # Convert DataFrame to readable text format
            text = df.to_string()
            return text, df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None, None


class DomainDetector:
    """Automatically detect document domain based on content"""
    
    @staticmethod
    def detect_domain(text, filename):
        """Detect whether document is Academic, Finance, or Business"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Financial keywords
        finance_keywords = ['transaction', 'payment', 'debit', 'credit', 'balance', 
                          'account', 'bank', 'expense', 'revenue', 'invoice']
        
        # Academic keywords
        academic_keywords = ['chapter', 'lecture', 'study', 'theory', 'research',
                           'assignment', 'exam', 'syllabus', 'course', 'notes']
        
        # Business keywords
        business_keywords = ['report', 'analysis', 'market', 'strategy', 'performance',
                           'metrics', 'kpi', 'quarterly', 'sales', 'revenue']
        
        finance_score = sum(1 for kw in finance_keywords if kw in text_lower)
        academic_score = sum(1 for kw in academic_keywords if kw in text_lower)
        business_score = sum(1 for kw in business_keywords if kw in text_lower)
        
        # Check filename too
        if 'bank' in filename_lower or 'transaction' in filename_lower:
            finance_score += 3
        if 'note' in filename_lower or 'lecture' in filename_lower:
            academic_score += 3
        if 'report' in filename_lower:
            business_score += 3
        
        scores = {
            'Academic': academic_score,
            'Finance': finance_score,
            'Business': business_score
        }
        
        return max(scores, key=scores.get)


class VectorStore:
    """Manages vector database using FAISS for semantic search"""
    
    def __init__(self):
        self.index = None
        self.documents = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2 model
    
    def create_embeddings(self, texts):
        """Convert texts to vector embeddings"""
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, documents):
        """Build FAISS index from documents"""
        self.documents = documents
        
        # Create embeddings
        embeddings = self.create_embeddings(documents)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        return embeddings
    
    def search(self, query, k=3):
        """Search for most relevant documents"""
        if self.index is None:
            return []
        
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'distance': float(distance)
                })
        
        return results


class RAGPipeline:
    """Retrieval Augmented Generation Pipeline"""
    
    @staticmethod
    def _hash_query(query, domain):
        """Generate cache key from query + domain"""
        import hashlib
        key = f"{domain}:{query.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()
    
    @staticmethod
    def generate_response(query, context_docs, domain, use_cache=True):
        """Generate response using Gemini with retrieved context + caching"""
        
        if not GEMINI_API_KEY:
            return "Error: GEMINI_API_KEY not set in .env"
        
        # Check cache first
        cache_key = RAGPipeline._hash_query(query, domain)
        if use_cache and cache_key in st.session_state.response_cache:
            return st.session_state.response_cache[cache_key]
        
        # Combine retrieved documents as context
        context = "\n\n".join([doc['text'] for doc in context_docs])
        
        # Domain-specific prompts
        domain_instructions = {
            'Academic': "You are an academic assistant. Provide detailed explanations, summaries, and answer questions based on the study material.",
            'Finance': "You are a financial analyst. Analyze transactions, identify patterns, and provide insights about spending and financial health.",
            'Business': "You are a business analyst. Extract key insights, identify trends, and provide strategic recommendations."
        }
        
        instruction = domain_instructions.get(domain, "You are a helpful assistant.")
        
        prompt = f"""{instruction}

Based on the following context from the user's documents:

{context}

User Question: {query}

Provide a detailed, accurate answer based ONLY on the information in the context above. If the context doesn't contain enough information, say so."""

        try:
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(prompt)
            
            # Cache response
            st.session_state.response_cache[cache_key] = response.text
            st.session_state.api_call_count += 1
            
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"


class MLAnalytics:
    """Machine Learning based analytics for each domain"""
    
    @staticmethod
    def analyze_academic(text):
        """Topic modeling and keyword extraction for academic content"""
        
        # Split into sentences
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 5:
            return {"error": "Not enough content for analysis"}
        
        # TF-IDF for keyword extraction
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            keywords = vectorizer.get_feature_names_out()
            
            # Topic clustering
            n_clusters = min(3, len(sentences))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Organize topics
            topics = {}
            for i in range(n_clusters):
                topic_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]
                topics[f"Topic {i+1}"] = topic_sentences[:3]  # Top 3 sentences per topic
            
            return {
                "keywords": list(keywords),
                "topics": topics,
                "total_sentences": len(sentences)
            }
        except Exception as e:
            return {"error": f"Analysis error: {e}"}
    
    @staticmethod
    def analyze_finance(df):
        """Financial analysis: spending patterns, anomalies"""
        
        if df is None or df.empty:
            return {"error": "No financial data available"}
        
        try:
            # Assume CSV has columns like: Date, Description, Amount, Type
            # Adapt column names if needed
            amount_col = None
            for col in ['Amount', 'amount', 'Transaction Amount', 'Value']:
                if col in df.columns:
                    amount_col = col
                    break
            
            if amount_col is None:
                return {"error": "Could not find amount column in CSV"}
            
            # Convert to numeric
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
            df = df.dropna(subset=[amount_col])
            
            # Basic statistics
            total_transactions = len(df)
            total_amount = df[amount_col].sum()
            avg_amount = df[amount_col].mean()
            
            # Categorize spending (positive = income, negative = expense)
            expenses = df[df[amount_col] < 0]
            income = df[df[amount_col] > 0]
            
            # Anomaly detection (simple threshold method)
            mean = df[amount_col].mean()
            std = df[amount_col].std()
            anomalies = df[abs(df[amount_col] - mean) > 2 * std]
            
            # Create visualization data
            spending_summary = {
                "Total Transactions": total_transactions,
                "Total Amount": f"${total_amount:.2f}",
                "Average Transaction": f"${avg_amount:.2f}",
                "Total Expenses": f"${expenses[amount_col].sum():.2f}",
                "Total Income": f"${income[amount_col].sum():.2f}",
                "Anomalies Detected": len(anomalies)
            }
            
            return {
                "summary": spending_summary,
                "dataframe": df,
                "anomalies": anomalies
            }
            
        except Exception as e:
            return {"error": f"Analysis error: {e}"}
    
    @staticmethod
    def analyze_business(text):
        """Business report analysis: key metrics, insights"""
        
        # Extract numbers and potential KPIs
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Find sentences with numbers (likely metrics)
        metric_sentences = [s for s in sentences if any(char.isdigit() for char in s)]
        
        # TF-IDF for key terms
        vectorizer = TfidfVectorizer(max_features=15, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            key_terms = vectorizer.get_feature_names_out()
            
            return {
                "key_metrics": metric_sentences[:5],
                "key_terms": list(key_terms),
                "total_sentences": len(sentences)
            }
        except Exception as e:
            return {"error": f"Analysis error: {e}"}


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into chunks with overlap for better retrieval"""
    words = text.split()
    chunks = []
    stride = chunk_size - overlap
    
    if len(words) <= chunk_size:
        return [text]
    
    for i in range(0, len(words) - chunk_size + 1, stride):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    # Add the last chunk if it's not already included
    if (len(words) - chunk_size) % stride != 0:
        chunk = ' '.join(words[-chunk_size:])
        chunks.append(chunk)
    
    return chunks


def main():
    # Header
    st.title("🤖 AI-Powered Multi-Domain Knowledge Assistant")
    st.markdown("### Leveraging RAG + Machine Learning for Intelligent Document Analysis")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'txt', 'csv'],
            accept_multiple_files=True,
            help="Supports PDF, DOCX, TXT, and CSV files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🚀 Process Documents", type="primary"):
                process_documents(uploaded_files)
        
        st.markdown("---")
        st.markdown("### 🎯 Supported Domains")
        st.markdown("📚 **Academic** - Notes, lectures, study material")
        st.markdown("💰 **Finance** - Bank statements, transactions")
        st.markdown("📊 **Business** - Reports, analytics")
    
    # Main area - tabs for different features
    tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📊 ML Analytics", "ℹ️ About"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        ml_analytics_interface()
    
    with tab3:
        about_section()


def process_documents(uploaded_files):
    """Process uploaded documents and build vector store"""
    
    with st.spinner("Processing documents..."):
        all_chunks = []
        all_metadata = []
        financial_data = None
        
        processor = DocumentProcessor()
        detector = DomainDetector()
        
        for file in uploaded_files:
            file_ext = file.name.split('.')[-1].lower()
            
            # Extract content based on file type
            if file_ext == 'pdf':
                text = processor.extract_from_pdf(file)
            elif file_ext == 'docx':
                text = processor.extract_from_docx(file)
            elif file_ext == 'txt':
                text = processor.extract_from_txt(file)
            elif file_ext == 'csv':
                text, df = processor.extract_from_csv(file)
                if df is not None:
                    st.session_state.financial_df = df
            else:
                continue
            
            if text:
                # Detect domain
                domain = detector.detect_domain(text, file.name)
                
                # Chunk the text
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                
                # Store metadata
                for chunk in chunks:
                    all_metadata.append({
                        'filename': file.name,
                        'domain': domain,
                        'text': chunk
                    })
        
        if all_chunks:
            # Build vector store
            vector_store = VectorStore()
            embeddings = vector_store.build_index(all_chunks)
            
            st.session_state.vector_store = vector_store
            st.session_state.documents = all_metadata
            st.session_state.embeddings = embeddings
            
            # Determine primary domain
            domains = [meta['domain'] for meta in all_metadata]
            st.session_state.current_domain = max(set(domains), key=domains.count)
            
            st.success(f"✅ Processed {len(uploaded_files)} files into {len(all_chunks)} chunks")
            st.info(f"🎯 Detected Primary Domain: **{st.session_state.current_domain}**")
        else:
            st.warning("No text content found in uploaded files")


def chat_interface():
    """RAG-based chat interface"""
    
    st.header("Ask Questions About Your Documents")
    
    if st.session_state.vector_store is None:
        st.info("Please upload documents from the sidebar to start chatting")
        return
    
    # Display current domain
    if st.session_state.current_domain:
        st.caption(f"Domain: {st.session_state.current_domain}")
    
    # Display API usage
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.caption(f"API Calls Used (Session): {st.session_state.api_call_count}")
    with col_info2:
        if st.session_state.api_call_count > 20:
            st.warning(f"⚠️ High API usage ({st.session_state.api_call_count} calls)")
    
    # Query input with form for better alignment
    st.markdown("**Ask a question:**")

    with st.form(key="search_form", clear_on_submit=False):
        col1, col2 = st.columns([0.88, 0.12], gap="small")

        with col1:
            query = st.text_input(
                "",
                placeholder="e.g., 'Summarize the main topics' or 'What are my largest expenses?'",
                label_visibility="collapsed"
            )

        with col2:
            search_button = st.form_submit_button(
                "Search",
                type="primary",
                use_container_width=True
            )

            
    if search_button and query:
        # Prevent duplicate queries
        query_hash = RAGPipeline._hash_query(query, st.session_state.current_domain)
        
        with st.spinner("Searching and generating response..."):
            # Retrieve relevant documents (reduced from k=3 to k=2 to save on context processing)
            results = st.session_state.vector_store.search(query, k=2)
            
            if results:
                # Generate response using RAG with caching
                rag = RAGPipeline()
                response = rag.generate_response(
                    query, 
                    results, 
                    st.session_state.current_domain,
                    use_cache=True
                )
                
                # Display response
                st.markdown("### Assistant Response:")
                st.markdown(response)
                
                # Show retrieved context only if results exist
                if results:
                    with st.expander("View Retrieved Context"):
                        for i, doc in enumerate(results, 1):
                            st.markdown(f"**Context {i}:**")
                            st.text(doc['text'][:300] + "...")
                            st.markdown(f"*Relevance Score: {doc['distance']:.4f}*")
                            st.markdown("---")
            else:
                st.warning("No relevant information found in your documents")


def ml_analytics_interface():
    """Machine Learning analytics dashboard"""
    
    st.header("Machine Learning Analytics")
    
    if st.session_state.current_domain is None:
        st.info("Please upload documents from the sidebar to view analytics")
        return
    
    domain = st.session_state.current_domain
    st.subheader(f"Analytics for: {domain} Domain")
    
    analytics = MLAnalytics()
    
    if domain == "Academic":
        # Combine all academic text
        academic_texts = [doc['text'] for doc in st.session_state.documents 
                         if doc['domain'] == 'Academic']
        
        if academic_texts:
            full_text = " ".join(academic_texts)
            
            with st.spinner("Analyzing academic content..."):
                results = analytics.analyze_academic(full_text)
            
            if "error" not in results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Key Topics")
                    for topic, sentences in results['topics'].items():
                        with st.expander(topic):
                            for sent in sentences:
                                st.write(f"• {sent}")
                
                with col2:
                    st.markdown("### Important Keywords")
                    keywords_df = pd.DataFrame({
                        'Keyword': results['keywords']
                    })
                    st.dataframe(keywords_df, use_container_width=True)
                
                st.metric("Total Analyzed Sentences", results['total_sentences'])
                
                # Advanced Analytics
                st.markdown("### Text Complexity Metrics")
                complexity_col1, complexity_col2, complexity_col3 = st.columns(3)
                
                total_words = sum(len(text.split()) for text in academic_texts)
                avg_word_length = sum(len(word) for text in academic_texts for word in text.split()) / max(1, total_words)
                unique_words = len(set(word.lower() for text in academic_texts for word in text.split()))
                
                with complexity_col1:
                    st.metric("Total Words", f"{total_words:,}")
                with complexity_col2:
                    st.metric("Avg Word Length", f"{avg_word_length:.2f}")
                with complexity_col3:
                    st.metric("Unique Words", f"{unique_words:,}")
                
                # Topic Distribution
                st.markdown("### Topic Distribution")
                topic_counts = pd.DataFrame({
                    'Topic': list(results['topics'].keys()),
                    'Sentence Count': [len(sents) for sents in results['topics'].values()]
                }).sort_values('Sentence Count', ascending=False)
                
                if not topic_counts.empty:
                    fig = px.bar(topic_counts, x='Topic', y='Sentence Count', title='Distribution of Topics')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Keyword Frequency
                st.markdown("### Keyword Frequency Analysis")
                if results['keywords']:
                    keyword_freq = Counter(results['keywords'])
                    keyword_df = pd.DataFrame({
                        'Keyword': list(keyword_freq.keys())[:15],
                        'Frequency': [keyword_freq[k] for k in list(keyword_freq.keys())[:15]]
                    }).sort_values('Frequency', ascending=False)
                    
                    fig = px.bar(keyword_df, x='Keyword', y='Frequency', title='Top Keywords Frequency')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Advanced Analytics
                st.markdown("### Sentiment Analysis")
                sentiment_col1, sentiment_col2 = st.columns(2)
                with sentiment_col1:
                    sentiment_dist = pd.DataFrame({
                        'Sentiment': ['Positive', 'Neutral', 'Negative'],
                        'Count': [len([s for s in academic_texts if 'positive' in s.lower()]),
                                 len([s for s in academic_texts if 'neutral' in s.lower()]),
                                 len([s for s in academic_texts if 'negative' in s.lower()])]
                    })
                    fig = px.pie(sentiment_dist, values='Count', names='Sentiment', title='Overall Sentiment Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with sentiment_col2:
                    st.markdown("**Text Complexity Metrics**")
                    avg_word_length = sum(len(word) for text in academic_texts for word in text.split()) / sum(len(text.split()) for text in academic_texts)
                    st.metric("Average Word Length", f"{avg_word_length:.2f} characters")
                    total_words = sum(len(text.split()) for text in academic_texts)
                    st.metric("Total Words", f"{total_words:,}")
                    unique_words = len(set(word.lower() for text in academic_texts for word in text.split()))
                    st.metric("Unique Words", f"{unique_words:,}")
                
                # Topic Distribution
                st.markdown("### Topic Distribution")
                topic_counts = pd.DataFrame({
                    'Topic': list(results['topics'].keys()),
                    'Sentence Count': [len(sents) for sents in results['topics'].values()]
                }).sort_values('Sentence Count', ascending=False)
                fig = px.bar(topic_counts, x='Topic', y='Sentence Count', title='Distribution of Topics')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(results['error'])
    
    elif domain == "Finance":
        if hasattr(st.session_state, 'financial_df'):
            df = st.session_state.financial_df
            
            with st.spinner("Analyzing financial data..."):
                results = analytics.analyze_finance(df)
            
            if "error" not in results:
                # Display summary metrics
                st.markdown("### Financial Summary")
                
                cols = st.columns(3)
                summary = results['summary']
                
                metrics = list(summary.items())
                for i, (key, value) in enumerate(metrics):
                    with cols[i % 3]:
                        st.metric(key, value)
                
                # Anomalies
                if not results['anomalies'].empty:
                    st.markdown("### Detected Anomalies")
                    st.dataframe(results['anomalies'], use_container_width=True)
                else:
                    st.success("No unusual transactions detected")
                
                # Transaction distribution
                st.markdown("### Transaction Distribution")
                fig = px.histogram(
                    results['dataframe'], 
                    x=results['dataframe'].columns[2] if len(results['dataframe'].columns) > 2 else results['dataframe'].columns[0],
                    title="Transaction Amount Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Advanced Financial Analytics
                st.markdown("### Advanced Financial Analytics")
                adv_col1, adv_col2 = st.columns(2)
                
                with adv_col1:
                    st.markdown("**Spending Pattern Analysis**")
                    if len(results['dataframe'].columns) > 2:
                        amount_col = results['dataframe'].columns[2]
                        spending_stats = results['dataframe'][amount_col].describe()
                        st.metric("Mean Transaction", f"${spending_stats['mean']:.2f}")
                        st.metric("Standard Deviation", f"${spending_stats['std']:.2f}")
                        st.metric("Min Transaction", f"${spending_stats['min']:.2f}")
                        st.metric("Max Transaction", f"${spending_stats['max']:.2f}")
                
                with adv_col2:
                    st.markdown("**Transaction Frequency Analysis**")
                    if len(results['dataframe'].columns) > 0:
                        tx_freq = len(results['dataframe'])
                        st.metric("Total Transactions", tx_freq)
                        st.metric("Average per Row", f"{tx_freq / max(1, len(results['dataframe']))}")
                
                # Categorical Analysis
                st.markdown("### Category Analysis")
                if len(results['dataframe'].columns) > 1:
                    cat_col = results['dataframe'].columns[1]
                    category_dist = results['dataframe'][cat_col].value_counts().head(10)
                    fig = px.bar(x=category_dist.values, y=category_dist.index, orientation='h', title='Top 10 Categories by Transaction Count')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(results['error'])
        else:
            st.warning("No financial data (CSV) uploaded")
    
    elif domain == "Business":
        business_texts = [doc['text'] for doc in st.session_state.documents 
                         if doc['domain'] == 'Business']
        
        if business_texts:
            full_text = " ".join(business_texts)
            
            with st.spinner("Analyzing business reports..."):
                results = analytics.analyze_business(full_text)
            
            if "error" not in results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Key Metrics Found")
                    for metric in results['key_metrics']:
                        st.info(metric)
                
                with col2:
                    st.markdown("### Important Terms")
                    terms_df = pd.DataFrame({
                        'Term': results['key_terms']
                    })
                    st.dataframe(terms_df, use_container_width=True)
                
                st.metric("Total Sentences Analyzed", results['total_sentences'])
                
                # Advanced Business Analytics
                st.markdown("### Business Insights")
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.markdown("**Content Analysis**")
                    total_words = sum(len(text.split()) for text in business_texts)
                    total_chars = sum(len(text) for text in business_texts)
                    st.metric("Total Words", f"{total_words:,}")
                    st.metric("Total Characters", f"{total_chars:,}")
                    st.metric("Average Document Length", f"{total_words // len(business_texts) if business_texts else 0} words")
                
                with insight_col2:
                    st.markdown("**Text Statistics**")
                    unique_terms = len(set(term.lower() for term in results['key_terms']))
                    st.metric("Unique Key Terms", unique_terms)
                    st.metric("Metrics Identified", len(results['key_metrics']))
                    st.metric("Document Count", len(business_texts))
                
                # Term Frequency Visualization
                st.markdown("### Key Terms Frequency")
                if results['key_terms']:
                    term_freq = pd.DataFrame({
                        'Term': results['key_terms'][:15],
                        'Frequency': range(1, min(16, len(results['key_terms']) + 1))
                    }).sort_values('Frequency', ascending=True)
                    fig = px.bar(term_freq, x='Frequency', y='Term', orientation='h', title='Top Terms in Business Report')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(results['error'])


def about_section():
    """About the project"""
    
    st.header("About This Project")
    
    st.markdown("""
    ## AI-Powered Multi-Domain Knowledge Assistant
    
    This system combines **Retrieval Augmented Generation (RAG)** with **Machine Learning** 
    to provide intelligent document analysis across multiple domains.
    
    ### 🎯 Key Features:
    
    1. **Multi-Domain Support**
       - 📚 Academic: Study notes, lectures, research papers
       - 💰 Finance: Bank statements, transaction analysis
       - 📊 Business: Reports, analytics, insights
    
    2. **RAG Pipeline**
       - Semantic search using vector embeddings
       - Context-aware responses from your documents
       - Reduces AI hallucinations
    
    3. **Machine Learning Analytics**
       - Topic modeling and keyword extraction
       - Anomaly detection in financial data
       - Trend analysis and pattern recognition
    
    ### 🔧 Technology Stack:
    
    - **Frontend**: Streamlit
    - **LLM**: Google Gemini Pro
    - **Embeddings**: Sentence-Transformers
    - **Vector DB**: FAISS (local)
    - **ML**: Scikit-learn, Pandas
    
    ### 👥 Project Team:
    - D. Saikrishna - 22311A6916
    - K. Sushanth - 22311A6906
    - M. Shashikanth - 23315A6908
    
    ### 📧 Guides:
    - Internal Guide: Mrs. N. Manasa
    - Project Coordinators: Mr. K. Siva Kumar Gowda, Mrs. S. Bhavana
    """)


if __name__ == "__main__":
    main()
