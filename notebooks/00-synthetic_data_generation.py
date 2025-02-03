# Databricks notebook source
# MAGIC %pip install openai -q -U
# MAGIC %pip install Faker -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00-config

# COMMAND ----------

from openai import OpenAI
import os

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Demographic Data

# COMMAND ----------
from faker_utils import FakeDemographicDataGenerator

config = {
    "name": True,
    "age": {"min": 18, "max": 90},
    "gender": ["Male", "Female", "Non-binary"],
    "email": True,
    "phone": True,
    "address": True,
    "city": True,
    "country": True,
    "income_level": ["Low", "Middle", "High"],
    "investment_experience": ["Beginner", "Intermediate", "Expert"],
    "risk_aversion": ["Low", "Medium", "High"],
    "investment_preference": ["Stocks", "Bonds", "Real Estate", "Cryptocurrency", "Mutual Funds"]
}

df_demographic = FakeDemographicDataGenerator(config=config, num_records=100)
df_demographic.head()

# COMMAND ----------

spark.createDataFrame(df_demographic).write.mode("overwrite").saveAsTable("customer_demographics")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Generate transcripts
# MAGIC
# MAGIC ## Create topics

# COMMAND ----------

import re

def oneshot_prompt(prompt, model="databricks-meta-llama-3-1-70b-instruct"):
    response = client.chat.completions.create(
      model=model,
      max_tokens=300,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
      ]
    )
    return response.choices[0].message.content 


def create_topic(model="databricks-meta-llama-3-1-70b-instruct", k=10):
    topic_prompt = \
    f"""
    Generate a list {k} topics a customer will inquire the personal finance and credit department Experain about. Response only the topics
    """
    topics_txt = oneshot_prompt(topic_prompt, model)
    #topics_txt = topics_txt.split(':')[1].strip()
    topics = re.sub(r'\"', '', topics_txt)
    topics = [re.sub(r'^[0-9]+\.', '', t).strip() for t in topics.split('\n')]
    return topics

# COMMAND ----------

topics = create_topic()
topics

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Transcripts
# MAGIC
# MAGIC create 3 to 5 transcripts of different topic fo each user

# COMMAND ----------

import random
import pandas as pd

def create_transcripts(topics, 
                       name, 
                       account_id, 
                       topic_count=1,
                       model="databricks-meta-llama-3-1-70b-instruct"):
    
    selected_topics = random.sample(topics, topic_count)
    transcripts = []
    for topic in selected_topics:
        #print(f"Topic: {topic}")
        #print("---------------------------------")
        prompt_tempate = \
        f"""
        Create a realistic call transcript between a customer <customer> who has called the cusomer finance and credict company Experian <experian> to ask for a recommendation. The wealth management company's customer service agent will converse with the customer about the following 
        topics: {topic}.

        For this particular customer, they provides the follow information to the customer service agent: 

        - name: {name}
        - account id: {account_id}

        Generate the transcript using the following format:

        <customer>:
        <experian>:

        Be thoughtful of the customer's tone and the the cusomer finance and credict company Experian's response.

        Your response:
        """

        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': prompt_tempate} 
            ]

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        transcripts.append((name, account_id, topic, response.choices[0].message.content))
    return pd.DataFrame(transcripts, columns=['name', 'account_id', 'topic', 'transcript'])

# COMMAND ----------

dfs = []
for i, r in df_demographic.iterrows():
    tc = random.randint(1, 5)
    dfs.append(create_transcripts(topics=topics, 
                                  name=r['Name'], 
                                  account_id=r['Account_id'], 
                                  topic_count=tc))

# COMMAND ----------

df_transcript = pd.concat(dfs)
print(df_transcript.shape)
df_transcript.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta

# COMMAND ----------

import re

def fix_transcript(x):
    x = re.sub(r'^Here\sis.*\n\n', '', x)
    x = re.sub(r'\*\*(.*:)\*\*', r'\1', x)
    return x

# COMMAND ----------

import re

df_transcript['transcript'] = df_transcript['transcript'].apply(fix_transcript)
display(df_transcript)

# COMMAND ----------

from pyspark.sql.functions import expr, rand, current_date, to_timestamp, date_sub, concat, lit, lpad
df = spark.createDataFrame(df_transcript)

df_new = df.withColumn(
    "timestamp",
    to_timestamp(
        concat(
            expr("date_sub(current_date(), cast(rand() * 365 * 5 as int))"),
            lit(" "),
            lpad(expr("cast(rand() * 23 as int)"), 2, '0'),
            lit(":"),
            lpad(expr("cast(rand() * 59 as int)"), 2, '0'),
            lit(":"),
            lpad(expr("cast(rand() * 59 as int)"), 2, '0')
        )
    )
)

# COMMAND ----------

table_name = "call_transcripts"

# write to a delta table
df_new.write \
    .mode('overwrite') \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Product Data
# MAGIC
# MAGIC Experian Products and Services:
# MAGIC
# MAGIC * credit monitoring service
# MAGIC * credit reports ans scores
# MAGIC * credict education tools
# MAGIC * Fraud prevention services
# MAGIC * credict card recommendation
# MAGIC * analytics solutions

# COMMAND ----------

fake = Faker()
experian_products = {
    'credit_monitoring_service': 'credit monitoring service that sends alerts for changes to your credit report', 
    'credit_reports_and_scores': 'one-time or periodic credit reports from all three bureaus, FICO® Score, FICO® Score Planner, VantageScore®, and Score Tracker',
    'credict_education': 'A variety of credit education tools and resources to help individuals understand their creditworthiness and improve their credit scores',
    'Fraud_prevention_services': 'Fraud prevention services that include credit bureau alerts, identity monitoring, and other fraud prevention services',
    'retirement_planning': 'a 401(k)s retirement plan service for small business owners',
    'business_solutions': 'data, analytics, and technology products for businesses, including Experian Edge, Commercial Insights Hub, and Business Databases'
}
products_tiers = ['free', 'basic', 'premium']

products = []
for prod, des in experian_products.items():
    for tier in products_tiers:
        products.append({
            'product_id': fake.random_int(min=10000, max=99999),
            'product_name': prod,
            'tier': tier,
            'description': des
        })
df_products = pd.DataFrame(products)
df_products.head()

# COMMAND ----------

spark.createDataFrame(df_products).write.mode('overwrite').saveAsTable('products')

# COMMAND ----------

