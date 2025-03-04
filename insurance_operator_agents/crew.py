from crewai_tools import DirectoryReadTool, FileReadTool
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from unitycatalog.ai.crewai.toolkit import UCFunctionToolkit
from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
import sys
import os
import yaml

databricks_cfg = yaml.safe_load(open('../config/databricks_config.yaml'))
llm_endpoint = databricks_cfg['llm_endpoint']
catalog = databricks_cfg['catalog']
schema = databricks_config['schema']

# Get UC Functions as tools
client = DatabricksFunctionClient()
set_uc_function_client(client)
functions = [f"{catalog}.{schema}.{func.name}" for func in client.list_functions(catalog=catalog,
                                                                                 schema=schema)
             if func.name.endswith('phone_number') or func.name == 'search_policy']

toolkit = UCFunctionToolkit(function_names=functions)
cs_profile_tool, find_transcript_tool, search_policy_tool = toolkit.tools
llm = LLM(model=llm_endpoint)

@CrewBase
class InsuranceOperatorCrew():
    """Insurance Operator Crew"""

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    @agent
    def transcript_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['transcript_analyzer'],
            tools=[find_transcript_tool],
            verbose=True
        )

    @agent
    def customer_service(self) -> Agent:
        return Agent(
            config=self.agents_config['customer_service'],
            tools=[cs_profile_tool, find_transcript_tool],
            verbose=True
        )

    @agent
    def policy_consultant(self) -> Agent:
        return Agent(
            config=self.agents_config['policy_consultant'],
            tools=[search_policy_tool],
            verbose=True
        )

    @task
    def process_claim(self) -> Task:
        return Task(
            config=self.tasks_config['process_claim'],
            agent=self.claims_adjuster
        )

    @task
    def underwrite_policy(self) -> Task:
        return Task(
            config=self.tasks_config['underwrite_policy'],
            agent=self.underwriter
        )
