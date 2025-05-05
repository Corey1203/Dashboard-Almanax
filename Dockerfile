FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip  && pip install --no-cache-dir -r requirements.txt

COPY . .

# One-time SHAP explainer generation before app launch
RUN python generate_shap_explainer.py

CMD ["streamlit", "run", "app_v13.py"]