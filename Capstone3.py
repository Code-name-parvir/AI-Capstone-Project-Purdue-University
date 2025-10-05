# My first business intelligence app!
# I'm learning streamlit and AI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# trying to import langchain
try:
    from langchain.llms import OpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.evaluation.qa import QAEvalChain
    langchain_available = True
except:
    langchain_available = False

# setup page - copied from tutorial
st.set_page_config(page_title="My AI Business App", layout="wide")

# CSS I found online
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
    color: blue;
}
.metric-box {
    background-color: lightblue;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def load_data():
    # loading the sales data file
    # checking if file exists first
    
    if os.path.exists('sales_data.csv'):
        try:
            st.info("Loading sales_data.csv...")
            df = pd.read_csv('sales_data.csv')
            
            # making sure date column is datetime (learned this is important)
            df['Date'] = pd.to_datetime(df['Date'])
            
            st.success(f"Loaded {len(df)} records from sales_data.csv!")
            return df
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            st.stop()
    
    # if file not in folder, try upload option (option 2)
    st.sidebar.subheader("Data File Required")
    st.sidebar.warning("sales_data.csv not found in project folder!")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your sales_data.csv file", 
        type=['csv'],
        help="The CSV should have columns: Date, Product, Region, Sales, Customer_Age, Customer_Gender, Customer_Satisfaction"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
            st.sidebar.success(f"Loaded {len(df)} records!")
            return df
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded file: {e}")
            st.sidebar.write("Make sure your CSV has the right format!")
            st.stop()
    
    # no data available - stop the app
    st.error("Cannot run without data file!")
    st.write("**Please do one of the following:**")
    st.write("1. Place 'sales_data.csv' in the same folder as Capstone.py")
    st.write("2. Upload your sales_data.csv file using the sidebar")
    st.write("")
    st.write("**Required CSV columns:**")
    st.write("- Date")
    st.write("- Product") 
    st.write("- Region")
    st.write("- Sales")
    st.write("- Customer_Age")
    st.write("- Customer_Gender")
    st.write("- Customer_Satisfaction")
    st.stop()  # stops the app here if no data

# this function makes charts
def make_charts(data):
    # calculating basic metrics
    total_sales = data['Sales'].sum()
    avg_satisfaction = data['Customer_Satisfaction'].mean()
    avg_age = data['Customer_Age'].mean()
    total_customers = len(data)
    
    # showing metrics in boxes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-box"><h3>Total Sales</h3><h2>${:,.0f}</h2></div>'.format(total_sales), unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box"><h3>Avg Satisfaction</h3><h2>{:.2f}/5</h2></div>'.format(avg_satisfaction), unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box"><h3>Avg Age</h3><h2>{:.0f} years</h2></div>'.format(avg_age), unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-box"><h3>Customers</h3><h2>{:,}</h2></div>'.format(total_customers), unsafe_allow_html=True)
    
    st.write("")  # adding space
    st.write("")
    
    # making charts using plotly
    col1, col2 = st.columns(2)
    
    with col1:
        # sales over time chart
        st.subheader("Sales Over Time")
        data['Date'] = pd.to_datetime(data['Date'])
        monthly_data = data.groupby(data['Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly_data['Date'] = monthly_data['Date'].astype(str)
        
        fig1 = px.line(monthly_data, x='Date', y='Sales', title="Monthly Sales")
        st.plotly_chart(fig1, use_container_width=True)
        
        # product sales chart
        st.subheader("Sales by Product")
        product_sales = data.groupby('Product')['Sales'].sum().reset_index()
        fig2 = px.bar(product_sales, x='Product', y='Sales', title="Product Performance")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # regional pie chart
        st.subheader("Sales by Region")
        region_sales = data.groupby('Region')['Sales'].sum().reset_index()
        fig3 = px.pie(region_sales, values='Sales', names='Region', title="Regional Distribution")
        st.plotly_chart(fig3, use_container_width=True)
        
        # age distribution
        st.subheader("Customer Ages")
        fig4 = px.histogram(data, x='Customer_Age', title="Age Distribution", nbins=20)
        st.plotly_chart(fig4, use_container_width=True)

# model evaluation function (trying to use QAEvalChain like the project said, not working, something needs fixing)
def evaluate_model_responses(api_key):
    st.header("Model Evaluation")
    st.write("Testing how good my AI is at answering questions!")
    
    if not api_key:
        st.warning("Need API key for evaluation too!")
        return
        
    if not langchain_available:
        st.error("Can't evaluate without LangChain!")
        return
    
    try:
        # setup evaluation chain (copied from docs)
        llm = OpenAI(temperature=0, openai_api_key=api_key)
        eval_chain = QAEvalChain.from_llm(llm)
        
        # some made up test questions and expected answers
        test_qa_pairs = [
            {
                "query": "What is the total sales amount?",
                "answer": "Based on the data, total sales are significant across all product lines and regions.",
                "result": ""  
            },
            {
                "query": "Which product performs best?", 
                "answer": "All widget products show strong performance with balanced distribution.",
                "result": ""
            },
            {
                "query": "How satisfied are customers?",
                "answer": "Customer satisfaction averages around 3-4 out of 5 across all segments.", 
                "result": ""
            }
        ]
        
        st.write("Running evaluation tests...")
        
        # evaluate each question
        results = []
        for i, qa_pair in enumerate(test_qa_pairs):
            with st.spinner(f"Testing question {i+1}..."):
                try:
                    # evaluate the answer quality
                    eval_result = eval_chain.evaluate(
                        [qa_pair],
                        predictions=[qa_pair["answer"]]
                    )
                    results.append(eval_result[0] if eval_result else {"text": "No evaluation"})
                except Exception as e:
                    st.write(f"Error evaluating question {i+1}: {e}")
                    results.append({"text": "Error in evaluation"})
        
        # show results (basic way)
        st.subheader("Evaluation Results:")
        for i, (qa_pair, result) in enumerate(zip(test_qa_pairs, results)):
            st.write(f"**Question {i+1}:** {qa_pair['query']}")
            st.write(f"**Answer:** {qa_pair['answer']}")
            st.write(f"**Evaluation:** {result.get('text', 'No evaluation available')}")
            st.write("---")
        
        # simple scoring (not sure if this is right)
        passed_tests = sum(1 for r in results if 'correct' in str(r).lower() or 'good' in str(r).lower())
        total_tests = len(results)
        score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        st.metric("Model Performance Score", f"{score:.1f}%")
        
        if score > 70:
            st.success("Model seems to be working well!")
        elif score > 50:
            st.warning("Model is okay but could be better")
        else:
            st.error("Model needs improvement")
            
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        st.write("Something went wrong with the evaluation!")
def ai_chat(data, api_key):
    st.header("Chat with AI")
    st.write("Ask questions about the data!")
    
    if not api_key:
        st.warning("Need API key to work!")
        return
    
    if not langchain_available:
        st.error("LangChain not installed properly!")
        return
    
    # setting up memory
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # showing previous messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.write("You: " + message['content'])
        else:
            st.write("AI: " + message['content'])
    
    # user input
    user_input = st.text_input("Type your question:")
    
    if st.button("Send") and user_input:
        # add user message
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        
        try:
            # setup AI
            llm = OpenAI(temperature=0.5, openai_api_key=api_key)
            
            # create some context from data
            context = f"""
            Data Summary:
            - Total Sales: ${data['Sales'].sum():,.2f}
            - Average Satisfaction: {data['Customer_Satisfaction'].mean():.2f}/5.0
            - Number of customers: {len(data)}
            - Products: {', '.join(data['Product'].unique())}
            - Regions: {', '.join(data['Region'].unique())}
            - Age range: {data['Customer_Age'].min()} to {data['Customer_Age'].max()} years
            """
            
            # simple prompt
            prompt = f"""
            You are a business analyst. Use this data to answer questions:
            {context}
            
            Question: {user_input}
            Answer:
            """
            
            response = llm(prompt)
            
            # add AI response
            st.session_state.messages.append({'role': 'ai', 'content': response})
            
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.write("Something went wrong with the AI!")

# analytics section 
def show_analytics(data):
    st.header("Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Stats")
        st.write("Sales Statistics:")
        st.write(f"Mean: ${data['Sales'].mean():.2f}")
        st.write(f"Median: ${data['Sales'].median():.2f}")
        st.write(f"Max: ${data['Sales'].max():.2f}")
        st.write(f"Min: ${data['Sales'].min():.2f}")
        
        st.write("")
        st.write("Customer Stats:")
        st.write(f"Average Age: {data['Customer_Age'].mean():.1f}")
        st.write(f"Average Satisfaction: {data['Customer_Satisfaction'].mean():.2f}")
        
    with col2:
        st.subheader("Top Performers")
        # finding best product
        product_sales = data.groupby('Product')['Sales'].sum()
        best_product = product_sales.idxmax()
        st.write(f"Best Product: {best_product}")
        
        # finding best region  
        region_sales = data.groupby('Region')['Sales'].sum()
        best_region = region_sales.idxmax()
        st.write(f"Best Region: {best_region}")
        
        # gender split
        gender_counts = data['Customer_Gender'].value_counts()
        st.write("Customer Gender:")
        for gender, count in gender_counts.items():
            st.write(f"- {gender}: {count}")

# main function (this runs everything)
def main():
    # title
    st.markdown('<p class="big-font">My Business Intelligence App!</p>', unsafe_allow_html=True)
    st.write("This is my first AI business app!")
    
    # sidebar
    st.sidebar.title("Settings")
    st.sidebar.write("Control panel:")
    
    # loading data
    with st.spinner('Loading data...'):
        df = load_data()
    
    st.sidebar.success(f"Loaded {len(df)} records!")
    
    # filters
    st.sidebar.subheader("Filters")
    selected_region = st.sidebar.selectbox("Pick Region:", ['All'] + list(df['Region'].unique()))
    selected_product = st.sidebar.selectbox("Pick Product:", ['All'] + list(df['Product'].unique()))
    
    # applying filters
    filtered_data = df.copy()
    if selected_region != 'All':
        filtered_data = filtered_data[filtered_data['Region'] == selected_region]
    if selected_product != 'All':
        filtered_data = filtered_data[filtered_data['Product'] == selected_product]
    
    # API key input
    api_key = st.sidebar.text_input("OpenAI API Key:", type="password", help="Enter your API key")
    
    # tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Analytics", "AI Chat", "Model Evaluation"])
    
    with tab1:
        st.header("Dashboard")
        if len(filtered_data) > 0:
            make_charts(filtered_data)
        else:
            st.write("No data with current filters!")
    
    with tab2:
        st.header("Advanced Analytics")
        if len(filtered_data) > 0:
            show_analytics(filtered_data)
        else:
            st.write("No data to analyze!")
    
    with tab3:
        ai_chat(filtered_data, api_key)
    
    with tab4:
        evaluate_model_responses(api_key)
    
    # footer
    st.write("---")
    st.write("Made with Streamlit!")

# running the app
if __name__ == "__main__":
    main()