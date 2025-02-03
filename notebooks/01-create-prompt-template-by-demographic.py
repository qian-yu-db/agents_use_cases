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
# MAGIC # Add Income Level and Age Range to Demographic Table

# COMMAND ----------

df_demo = spark.table("customer_demographics")
display(df_demo)

# COMMAND ----------

df_demo_pd = df_demo.toPandas()
df_demo_pd.head()

# COMMAND ----------

df_demo_pd['Education'].value_counts()

# COMMAND ----------

import random

def add_income(edu):
    if edu == 'Bachelors':
        return random.choice(['low', 'mid', 'high'])
    elif edu == 'Masters':
        return random.choice(['high', 'mid'])
    elif edu == 'High School':
        return random.choice(['low', 'mid'])
    else:
        return random.choice(['mid', 'high'])

df_demo_pd['Income'] = df_demo_pd['Education'].apply(add_income)
df_demo_pd.head()

# COMMAND ----------

df_new = spark.sql(
    """
    select
    *,
    case when age < 30 then '18 to 30' 
         when age < 45 then '30 to 45' 
         when age < 60 then '45 to 60'
         else '60 to 90'
     end as age_range
    from customer_demographics
    """
)
print(df_new.count())
display(df_new)

# COMMAND ----------

df_new.write \
    .mode('overwrite') \
    .option("overwriteSchema", "true") \
    .saveAsTable('customer_demographics')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Define Customer Segments
# MAGIC
# MAGIC * Gender
# MAGIC * Income
# MAGIC * Education Level

# COMMAND ----------

def generate_prompt_tempates(gender, 
                             income,
                             age_range, 
                             education,
                             model="databricks-meta-llama-3-1-70b-instruct"):
    
    prompt_template = f"""
    Generate a LLM prompt to be used in a RAG application for customer service based on the demographic of a {age_range} year old, {education} educated, {income} income {gender} gender.

    the prompte should enable LLM to give hyper personalized response with the given demographics.
    
    only return the prompt, do not explain
    """

    messages = [
        {'role': 'system', 'content': "You are a helpful assistant."},
        {'role': 'user', 'content': prompt_template} 
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

# COMMAND ----------

df_segments = spark.sql(
    """
    select
    distinct 
    gender,
    case when age < 30 then '18 to 30' 
         when age < 45 then '30 to 45' 
         when age < 60 then '45 to 60'
         else '60 to 90'
     end as age_range,
    income,
    education
    from customer_demographics
    """
)
print(df_segments.count())
display(df_segments)

# COMMAND ----------

df_segments_pd = df_segments.toPandas()
for index, row in df_segments_pd.iterrows():
    df_segments_pd.at[index, 'prompt_template'] = generate_prompt_tempates(row['gender'], 
                                                                           row['age_range'],
                                                                           row['income'], 
                                                                           row['education'])

# COMMAND ----------

display(df_segments_pd)

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, trim

#df = spark.createDataFrame(df_segments_pd)
df_new = df.withColumn("prompt_template", trim(regexp_replace("prompt_template", "Here is.*prompt:|Here is a potential prompt.*:", "")))
display(df_new)

# COMMAND ----------

df_new.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("prompt_personas")

# COMMAND ----------

# MAGIC %md
# MAGIC # Test UC function of create prompt tempates

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION qyu_test.experian.get_personalized_product_description(gender STRING, age_range STRING, income STRING, education STRING, occupation STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'This functions generate a persona description based on a customer gendar, age_range, income, education, and occupation.'
# MAGIC RETURN SELECT ai_query('databricks-meta-llama-3-1-70b-instruct',
# MAGIC   CONCAT("You are an specialist in personal finance and credit. Your goal is to write a description of product or service needs of a customer demographic persona in 2 to 3 sentences. The customer is ", age_range, " year old ", education, " educated ", gender, " who works as ", occupation, " and earns ", income, " income in a household.") 
# MAGIC   ) as persona_description;
# MAGIC
# MAGIC select get_personalized_product_description('Female', '30 to 45', 'low', 'Bachelor', 'music teacher')

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC *
# MAGIC from products

# COMMAND ----------

