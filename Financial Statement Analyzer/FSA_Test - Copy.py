import os
import re
import openai
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from PyPDF2 import PdfReader
from flask import Flask, request, render_template

from openai import OpenAI

client = OpenAI(
  api_key="your-API-Key"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def summarize_financial_text(text):
    try:
        response = openai.Completion.create(
            model="gpt-4",
            prompt=f"Summarize key financial trends and figures:\n{text}",
            max_tokens=500
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

def parse_financial_data(text):
    revenue = re.search(r"Revenue:?\s*\$?([\d,]+)", text)
    profit = re.search(r"Profit:?\s*\$?([\d,]+)", text)
    data = {
        "Revenue": int(revenue.group(1).replace(',', '')) if revenue else 0,
        "Profit": int(profit.group(1).replace(',', '')) if profit else 0
    }
    return pd.DataFrame([data])

def calculate_yoy_changes(df, column):
    df[f'{column}_YoY'] = df[column].pct_change() * 100
    return df

def calculate_ratios(data):
    try:
        ratios = {
            "Gross Margin": data['Gross_Profit'] / data['Revenue'] if data['Revenue'] else None,
            "Operating Margin": data['Operating_Profit'] / data['Revenue'] if data['Revenue'] else None,
            "ROE": data['Net_Income'] / data['Equity'] if data['Equity'] else None,
            "Current Ratio": data['Current_Assets'] / data['Current_Liabilities'] if data['Current_Liabilities'] else None,
        }
        return ratios
    except KeyError as e:
        print(f"Missing data for ratio calculation: {e}")
        return None

def forecast_eps(historical_eps):
    x = np.arange(len(historical_eps)).reshape(-1, 1)
    y = np.array(historical_eps)
    model = LinearRegression().fit(x, y)
    next_year = len(historical_eps)
    predicted_eps = model.predict([[next_year]])
    return predicted_eps[0]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        text = extract_text_from_pdf(file)
        if text:
            summary = summarize_financial_text(text)
            financial_data = parse_financial_data(text)
            ratios = calculate_ratios({
                "Revenue": financial_data.at[0, "Revenue"],
                "Gross_Profit": 500000,  # Replace with real value
                "Operating_Profit": 200000,  # Replace with real value
                "Net_Income": 150000,  # Replace with real value
                "Equity": 700000,  # Replace with real value
                "Current_Assets": 300000,  # Replace with real value
                "Current_Liabilities": 150000,  # Replace with real value
            })
            return render_template("results.html", summary=summary, data=financial_data.to_html(), ratios=ratios)
        else:
            return "Error extracting text from the uploaded file."
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
