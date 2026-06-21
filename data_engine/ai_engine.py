import pandas as pd
import numpy as np
import os
import json
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAnalyst:
    def __init__(self):
        self.provider = "rule_based"
        self.model = None
        
        # 1. Try OpenAI
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                self.provider = "gemini"
                logger.info("AI Analyst: Using Google Gemini (Free Tier)")
            except Exception as e:
                logger.error(f"Failed to init Gemini: {e}")

        elif self.openai_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_key)
                self.provider = "openai"
                self.model_name = "gpt-4o-mini" # Cost effective
                logger.info("AI Analyst: Using OpenAI")
            except Exception as e:
                logger.error(f"Failed to init OpenAI: {e}")
                
    def analyze_dataset(self, df: pd.DataFrame) -> dict:
        """
        Main entry point for dataset analysis.
        Returns a dictionary with 'summary', 'insights', 'recommendations'.
        """
        
        # prepares a lean context string to avoid token limits
        context = self._prepare_context(df)
        
        if self.provider == "gemini":
            return self._analyze_with_gemini(context)
        elif self.provider == "openai":
            return self._analyze_with_openai(context)
        else:
            return self._analyze_rule_based(df)

    def _prepare_context(self, df: pd.DataFrame) -> str:
        """Creates a text summary of the dataframe for the LLM."""
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        buffer = []
        buffer.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        buffer.append(f"Numeric Columns: {', '.join(numeric_cols[:10])}")
        buffer.append(f"Categorical Columns: {', '.join(cat_cols[:10])}")
        buffer.append(f"Missing Values: {df.isnull().sum().sum()}")
        
        # Sample Data
        sample = df.head(3).to_string(index=False)
        buffer.append(f"\nSample Data:\n{sample}")
        
        # Correlations (Top 3)
        if len(numeric_cols) > 1:
            corr_mat = df[numeric_cols].corr().abs()
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            high_corr = upper.stack().sort_values(ascending=False).head(3)
            buffer.append("\nTop Correlations:")
            for idx, val in high_corr.items():
                buffer.append(f"- {idx[0]} vs {idx[1]}: {val:.2f}")
                
        return "\n".join(buffer)

    def _analyze_with_gemini(self, context: str) -> dict:
        prompt = f"""
        Act as a Senior Data Analyst. Analyze this dataset summary:
        
        {context}
        
        Provide:
        1. An Executive Summary (2 sentences max).
        2. 3-5 Key Insights (bullet points).
        3. 2 Recommendations for next steps.
        
        Format output as JSON with keys: "summary", "insights" (list), "recommendations" (list).
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "")
            return json.loads(text)
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return self._fallback_response(context)

    def _analyze_with_openai(self, context: str) -> dict:
        prompt = f"""
        Act as a Senior Data Analyst. Analyze this dataset summary:
        
        {context}
        
        Provide JSON output with keys: "summary", "insights" (list of strings), "recommendations" (list of strings).
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI Error: {e}")
            return self._fallback_response(context)

    def _analyze_rule_based(self, df: pd.DataFrame) -> dict:
        """Advanced Rule-Based Expert System when no AI key is present."""
        insights = []
        recommendations = []
        
        # 1. Structure Analysis
        if df.shape[0] < 50:
            insights.append("Small dataset detected (under 50 rows). Results may not be statistically significant.")
            recommendations.append("Collect more data to improve reliability.")
        
        # 2. Missing Data
        missing_pct = df.isnull().mean().mean()
        if missing_pct > 0.1:
            insights.append(f"High missing data ({missing_pct:.1%}). This requires cleaning.")
            recommendations.append("Impute missing values using Mean/Median strategies.")
        
        # 3. Correlation Check
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr_mat = numeric_df.corr().abs()
            upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
            high_corrs = [column for column in upper.columns if any(upper[column] > 0.9)]
            if high_corrs:
                insights.append(f"Detected multicollinearity (redundancy) in: {', '.join(high_corrs)}.")
                recommendations.append("Consider dropping highly correlated features to avoid overfitting.")
        
        # 4. Outlier Check
        outlier_cols = []
        for col in numeric_df.columns[:5]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr)))]
            if not outliers.empty:
                outlier_cols.append(col)
        
        if outlier_cols:
            insights.append(f"Potential outliers found in {len(outlier_cols)} columns (e.g., {outlier_cols[0]}).")

        # Summary Generation
        summary = f"The dataset contains {df.shape[0]} records and {df.shape[1]} attributes. "
        if missing_pct == 0:
            summary += "It is clean with no missing values. "
        else:
            summary += "It requires data cleaning due to missing values. "
            
        return {
            "summary": summary,
            "insights": insights if insights else ["No significant anomalies detected."],
            "recommendations": recommendations if recommendations else ["Proceed with exploratory analysis."]
        }

    def recommend_model_config(self, df: pd.DataFrame, goal: str) -> dict:
        """
        Suggests model configuration based on dataset and user goal.
        """
        if self.provider == "gemini":
            return self._recommend_gemini(df, goal)
        elif self.provider == "openai":
            return self._recommend_openai(df, goal)
        else:
            return self._recommend_rule_based(df, goal)

    def _recommend_gemini(self, df: pd.DataFrame, goal: str) -> dict:
        prompt = f"""
        Act as a Machine Learning Engineer.
        Dataset Columns: {', '.join(df.columns.tolist())}
        User Goal: "{goal}"
        
        Suggest the best configuration to achieve this goal.
        Return JSON with keys:
        - "target_column": (string) exact column name from dataset
        - "problem_type": "regression" or "classification" or "clustering"
        - "algorithm": "random forest", "linear", "logistic", "kmeans", "decision tree"
        - "reasoning": (string) brief explanation of why.
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "")
            return json.loads(text)
        except Exception as e:
            logger.error(f"Gemini Config Error: {e}")
            return self._recommend_rule_based(df, goal)

    def _recommend_openai(self, df: pd.DataFrame, goal: str) -> dict:
        prompt = f"""
        Act as a Machine Learning Engineer.
        Dataset Columns: {', '.join(df.columns.tolist())}
        User Goal: "{goal}"
        
        Suggest the best configuration.
        Return JSON with keys: "target_column", "problem_type", "algorithm", "reasoning".
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI Config Error: {e}")
            return self._recommend_rule_based(df, goal)

    def _recommend_rule_based(self, df: pd.DataFrame, goal: str) -> dict:
        # 1. Detect Potential Target
        goal_lower = goal.lower()
        cols = df.columns.tolist()
        target_col = None
        
        # Keyword match
        for col in cols:
            if col.lower() in goal_lower:
                target_col = col
                break
        
        # Fallback to last column
        if not target_col: target_col = cols[-1]
            
        # 2. Analyze Target
        problem_type = "regression"
        if target_col in df.columns:
            series = df[target_col]
            if pd.api.types.is_numeric_dtype(series):
                if series.nunique() < 20 and series.dtype != 'float':
                     problem_type = "classification"
            else:
                problem_type = "classification"
        
        return {
            "target_column": target_col,
            "problem_type": problem_type,
            "algorithm": "random forest",
            "reasoning": f"Based on '{goal}', we selected '{target_col}'. As a {problem_type} problem, Random Forest is a robust choice (Rule-Based Fallback)."
        }

    def _fallback_response(self, context):
        return {
            "summary": "AI Analysis failed. Using fallback summary.",
            "insights": ["Could not generate deep insights."],
            "recommendations": ["Check API Keys."]
        }

    def process_query(self, df: pd.DataFrame, query: str) -> dict:
        """
        Classifies the query and processes it (Viz, Model, or Text).
        """
        prompt = f"""
        Act as a Data Assistant.
        Dataset Columns: {', '.join(df.columns.tolist())}
        User Query: "{query}"
        
        Classify intent and generate response JSON:
        1. If user wants a CHART/PLOT/VISUALIZATION:
           Type: "visualization"
           Data: {{ "chartType": "bar/line/scatter/pie", "x": "col_name", "y": "col_name", "title": "...", "description": "..." }}
        
        2. If user wants to TRAIN/BUILD MODEL/PREDICT:
           Type: "modeling"
           Data: {{ "target_column": "...", "problem_type": "...", "algorithm": "...", "reasoning": "..." }}
           
        3. If general question/insight:
           Type: "text"
           Data: {{ "response": "..." }}
           
        Return JSON with keys: "type", "data".
        """
        
        if self.provider == "gemini":
            return self._process_gemini(prompt, df)
        elif self.provider == "openai":
            return self._process_openai(prompt)
        else:
            return self._process_rule_based(df, query)

    def _process_gemini(self, prompt: str, df: pd.DataFrame) -> dict:
        try:
            response = self.model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "")
            return json.loads(text)
        except Exception as e:
            logger.error(f"Gemini Process Error: {e}")
            return self._process_rule_based(df, "Fallback")

    def _process_openai(self, prompt: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
             logger.error(f"OpenAI Process Error: {e}")
             return {"type": "text", "data": {"response": "AI Error. Using fallback."}}

    def _process_rule_based(self, df: pd.DataFrame, query: str) -> dict:
        # Simple heuristic fallback
        query_lower = query.lower()
        
        # 1. Visualization
        if any(w in query_lower for w in ['plot', 'chart', 'graph', 'show', 'visualize']):
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                return {
                    "type": "visualization",
                    "data": {
                        "chartType": "bar",
                        "x": df.columns[0] if not num_cols else df.columns[0], # Just picking something
                        "y": num_cols[0],
                        "title": f"{num_cols[0]} Analysis",
                        "description": "Rule-based fallback chart"
                    }
                }
        
        # 2. Modeling
        if any(w in query_lower for w in ['predict', 'model', 'train', 'classify']):
             # Use existing logic
             config = self._recommend_rule_based(df, query)
             return {"type": "modeling", "data": config}
             
        # 3. Text
        return {
            "type": "text",
            "data": {"response": f"I processed your query based on keyword rules. To get smarter insights, please add an API Key."}
        }

# Helper to be imported
def suggest_model_config(df: pd.DataFrame, goal: str):
    # Backward compatibility: Use the class instance
    analyst = AIAnalyst()
    return analyst.recommend_model_config(df, goal)

def process_ai_query(df: pd.DataFrame, query: str):
    analyst = AIAnalyst()
    return analyst.process_query(df, query)
