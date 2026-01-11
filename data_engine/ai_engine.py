from openai import OpenAI
import os
import pandas as pd
import json

# Ensure client is initialized only when needed or global if env is ready
# We will initialize inside function or globally if imported after load_dotenv
# But better to check key at runtime

def suggest_model_config(df: pd.DataFrame, user_goal: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OpenAI API Key not found. Please check .env file."}

    client = OpenAI(api_key=api_key)

    # Prepare context
    columns = list(df.columns)
    # Convert dtypes to string format for JSON serialization
    dtypes = {k: str(v) for k, v in df.dtypes.items()}
    
    # Get a small sample, handle NaNs for JSON
    sample = df.head(3).replace({float('nan'): None}).to_dict(orient='records')
    
    prompt = f"""
    I have a dataset with these columns: {columns}
    Data Types: {dtypes}
    Sample Data: {sample}
    
    User Goal: "{user_goal}"
    
    Based on the goal and data, suggest:
    1. The target column (must be one of the columns).
    2. The problem type (Regression or Classification).
    3. The best algorithm from this exact list: 
       [Linear Regression, Logistic Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, SVR, SVC, KNN, Naive Bayes].
       
    Rules:
    - If the target is numeric and goal implies predicting value -> Regression.
    - If target is categorical or few unique integers -> Classification.
    - If goal is vague, infer the most likely target.
    
    Return ONLY valid JSON in this format:
    {{
        "target_column": "col_name",
        "problem_type": "Regression" | "Classification",
        "algorithm": "Algorithm Name",
        "reasoning": "Brief explanation"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-nano", # Use 3.5-turbo for speed/cost, or gpt-4 if available
            messages=[
                {"role": "system", "content": "You are a data science expert helper."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Simple cleanup for markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        return json.loads(content.strip())
        
    except Exception as e:
        print(f"AI Error: {e}")
        return {"error": str(e)}
