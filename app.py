app_code='''import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Download NLTK data (only needed first time)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, vectorizer, encoder
    except FileNotFoundError:
        st.error("Model files not found! Please train the model first.")
        return None, None, None

# Text preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = str(text).lower()
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = ' '.join(text.split())
    
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Prediction function
def predict_news(statement, model, vectorizer, encoder):
    cleaned = preprocess_text(statement)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    predicted_label = encoder.inverse_transform([prediction])[0]
    class_probabilities = dict(zip(encoder.classes_, prediction_proba))
    return predicted_label, class_probabilities, cleaned

# Main app
def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1E88E5;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">📰 Fake News Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze news credibility using Machine Learning</p>', unsafe_allow_html=True)
    
    model, vectorizer, encoder = load_models()
    
    if model is None:
        st.stop()
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This AI-powered tool analyzes news statements and predicts their credibility level.
        
        **Categories:**
        - ✅ True
        - 🟢 Mostly True
        - 🟡 Half True
        - 🟠 Barely True
        - 🔴 False
        - 🔥 Pants on Fire
        """)
        
        st.header("📊 Model Info")
        st.write(f"**Algorithm:** Logistic Regression")
        st.write(f"**Features:** TF-IDF")
        
        st.header("🎯 Tips")
        st.write("""
        - Enter complete sentences
        - Provide context when possible
        - Longer statements work better
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Enter News Statement")
        
        user_input = st.text_area(
            "Paste or type the news statement you want to verify:",
            height=150,
            placeholder="Example: The president announced a new policy today..."
        )
        
        analyze_button = st.button("🔎 Analyze Statement", type="primary", use_container_width=True)
        
        with st.expander("📝 Try Example Statements"):
            examples = [
                "The president signed a new bill into law today.",
                "Scientists discovered that the earth is flat.",
                "The unemployment rate decreased by 2% last quarter.",
            ]
            for i, example in enumerate(examples):
                if st.button(f"Example {i+1}", key=f"ex_{i}"):
                    user_input = example
    
    
    if analyze_button and user_input:
        with st.spinner("🔄 Analyzing statement..."):
            try:
                predicted_label, probabilities, cleaned_text = predict_news(
                    user_input, model, vectorizer, encoder
                )
                
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    emoji_map = {
                        'true': '✅',
                        'mostly-true': '🟢',
                        'half-true': '🟡',
                        'barely-true': '🟠',
                        'false': '🔴',
                        'pants-fire': '🔥'
                    }
                    
                    emoji = emoji_map.get(predicted_label, '❓')
                    st.markdown(f"### Predicted Credibility:")
                    st.markdown(f"# {emoji} **{predicted_label.upper()}**")
                
                max_prob = max(probabilities.values())
                if max_prob > 0.5:
                    confidence = "High"
                    color = "green"
                elif max_prob > 0.3:
                    confidence = "Moderate"
                    color = "orange"
                else:
                    confidence = "Low"
                    color = "red"
                
                st.markdown(f"**Model Confidence:** :{color}[{confidence} ({max_prob*100:.1f}%)]")
                
                st.subheader("📊 Probability Distribution")
                
                prob_df = pd.DataFrame({
                    'Category': list(probabilities.keys()),
                    'Probability': list(probabilities.values())
                }).sort_values('Probability', ascending=True)
                
                fig = px.bar(
                    prob_df,
                    x='Probability',
                    y='Category',
                    orientation='h',
                    text=prob_df['Probability'].apply(lambda x: f'{x*100:.1f}%'),
                    color='Probability',
                    color_continuous_scale='RdYlGn'
                )
                
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Probability",
                    yaxis_title="",
                    xaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📋 View Detailed Probabilities"):
                    prob_df_sorted = pd.DataFrame({
                        'Category': list(probabilities.keys()),
                        'Probability': [f"{v*100:.2f}%" for v in probabilities.values()]
                    })
                    st.dataframe(prob_df_sorted, use_container_width=True, hide_index=True)
                
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
    
    elif analyze_button and not user_input:
        st.warning("⚠️ Please enter a news statement to analyze.")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit</p>
            <p><small>⚠️ Always verify news from multiple reliable sources.</small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
'''

# Write to file
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print("✅ app.py created successfully!")
print("\n" + "="*60)
print("File location:", os.path.abspath('app.py'))
print("="*60)

