# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Purpose
# MAGIC
# MAGIC We will generate synthetic data for our Investment Advisor Multi-Agent Use Cases
# MAGIC
# MAGIC - Customer Demographic Data
# MAGIC - Investment Services and Products

# COMMAND ----------

# MAGIC %pip install openai -q -U
# MAGIC %pip install Faker -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text(name="catalog", label="Catalog", defaultValue="fins_genai")
dbutils.widgets.text(name="schema", label="Schema", defaultValue="agents")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import sys
import os
sys.path.append(os.path.abspath('.'))

spark.sql(f"USE CATALOG {catalog};")
spark.sql(f"USE SCHEMA {schema};")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Demographic Data

# COMMAND ----------

from faker_utils import FakeDemographicDataGenerator

config = {
    "name": None,
    "age": {"min": 18, "max": 90},
    "gender": ["Male", "Female", "Non-binary"],
    "email": None,
    "phone": None,
    "address": None,
    "city": None,
    "country": None,
    "income_level": ["Low", "Middle", "High"],
    "investment_experience": ["Beginner", "Intermediate", "Expert"],
    "risk_aversion": ["Low", "Medium", "High"],
    "investment_preference": ["Stocks", "Bonds", "Real Estate", "Cryptocurrency", "Mutual Funds"]
}

demographic_gen = FakeDemographicDataGenerator(config=config, num_records=100)
df_demographic = demographic_gen.generate()
df_demographic.head()

# COMMAND ----------

spark.createDataFrame(df_demographic).write \
    .mode("overwrite") \
    .saveAsTable("investment_customer_demographics")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create Investment Products

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Investment Product Offering

# COMMAND ----------

from openai import OpenAI
import os

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

CLIENT = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints"
)

# COMMAND ----------

import re
import json

def oneshot_prompt(prompt, model="databricks-meta-llama-3-3-70b-instruct"):
    response = CLIENT.chat.completions.create(
      model=model,
      max_tokens=2000,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
      ]
    )
    return response.choices[0].message.content 


def create_product(model="databricks-meta-llama-3-3-70b-instruct", k=10):
    product_prompt = \
    f"""
    Generate a list {k} finacial products or services a investment management company would offer to customer of different risk aversion and income levels. Return JSON key value pairs where key is the service name and value is the service description.
    """
    result = oneshot_prompt(product_prompt, model)
    match = re.search(r'```json(.*?)```', result, re.DOTALL)
    if match:
        result = match.group(1).strip()
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = {}
    else:
        return result
    return result

# COMMAND ----------

products = create_product()
products

# COMMAND ----------

from faker_utils import generate_product_data

products_tiers = ['self_managed', 'digital_advisor', 'personal_advisor']

df_products = generate_product_data(products, products_tiers)
df_products

# COMMAND ----------

spark.createDataFrame(df_products).write.mode('overwrite').saveAsTable('investment_products')

# COMMAND ----------


