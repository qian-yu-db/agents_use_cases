# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install crewai crewai_tools langchain_community -q
# MAGIC %pip install openai mlflow json2html markdownify nbconvert nbformat -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import yaml
import sys
from crewai import Agent, Task, Crew

sys.path.append(os.path.abspath('./tools/'))
sys.path.append(os.path.abspath('.'))

catalog = "fins_genai"
schema = "agents"
volume = "ml_documents"
model = "wine_quality"

os.environ["DATABRICKS_API_KEY"] = dbutils.secrets.get("databricks_token_qyu", "qyu_rag_sp_token")
os.environ["DATABRICKS_API_BASE"] = (
    f"{dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()}/serving-endpoints"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup Crew
# MAGIC
# MAGIC * environment and input params are written to `./config/env.yaml`
# MAGIC * agents and tasks are defined in `./config/agents.yaml` and `./config/tasks.yaml`
# MAGIC * custom tools defined in `./tools/custom_tools.py`
# MAGIC * crew is defined in `./crew.py`
# MAGIC

# COMMAND ----------

import yaml

env_config = {
    "catalog": catalog,
    "schema": schema,
    "ml_model": model,
    "llm_endpoint" : "databricks/databricks-meta-llama-3-3-70b-instruct"
}

with open("config/env.yaml", "w") as file:
    yaml.dump(env_config, file)

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Crew Agents

# COMMAND ----------

from crew import MlDocumentCrew
import mlflow

mlflow.set_experiment("/Workspace/Users/q.yu@databricks.com/mlflow_experiments/ml_document_agent")
mlflow.crewai.autolog()

ml_document_crew = MlDocumentCrew().crew()

# COMMAND ----------

with mlflow.start_span("ml_document_agent", span_type="AGENT") as span:
    inputs = {"catalog": catalog, "schema": schema, "model": model}
    result = ml_document_crew.kickoff(inputs=inputs)

# COMMAND ----------

from IPython.display import display, Markdown
import re

cleaned_result = re.sub(r'```markdown\n', '', result.raw)
display(Markdown(cleaned_result))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log a crewAI Model

# COMMAND ----------

from mlflow.models import infer_signature

signature = infer_signature(inputs, result)

# COMMAND ----------

import mlflow.pyfunc

def crewAI_ml_doc_crew(model_input):
    from crew import MlDocumentCrew

    ml_document_crew = MlDocumentCrew().crew()
    result = ml_document_crew.kickoff(model_input)
    return result

# COMMAND ----------

model_name = "crewai_ml_autodoc_model"
with mlflow.start_run():
    log_info = mlflow.pyfunc.log_model(artifact_path=model_name, 
                                       python_model=crewAI_ml_doc_crew,
                                       input_example=inputs,
                                       signature=signature,
                                       registered_model_name=f"{catalog}.{schema}.{model_name}")


# COMMAND ----------

