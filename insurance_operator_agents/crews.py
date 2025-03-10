import yaml
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai.project import CrewBase, agent, task, crew
from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.crewai.toolkit import UCFunctionToolkit

databricks_cfg = yaml.safe_load(open('../global_config/databricks_config.yaml'))
llm_endpoint = databricks_cfg['llm_endpoint']
catalog = databricks_cfg['catalog']
schema = databricks_cfg['schema']

# Get UC Functions as tools
client = DatabricksFunctionClient()
set_uc_function_client(client)
uc_tools = [f"{catalog}.{schema}.{func.name}" for func in client.list_functions(catalog=catalog,
                                                                                schema=schema)
            if 'search' in func.name]

toolkit = UCFunctionToolkit(function_names=uc_tools)
tools = toolkit.tools
search_claim_details_tool, search_customer_policy_details_tool, search_policy_doc_tool = tools
llm = LLM(model=llm_endpoint)


# Create Crews
@CrewBase
class TriageCrew:
    """Insurance Operator Crew"""

    agents_config = './config/agents.yaml'
    tasks_config = './config/tasks.yaml'

    @agent
    def triage_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['triage_specialist'],
            verbose=True,
            llm=llm
        )

    @task
    def triage_task(self) -> Task:
        return Task(
            config=self.tasks_config['triage_task'],
            verbose=True,
        )

    @task
    def clarification_task(self) -> Task:
        return Task(
            config=self.tasks_config['clarification_task'],
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Triage Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks
        )


@CrewBase
class ClaimCrew:
    """Insurance Operator Crew"""

    agents_config = './config/agents.yaml'
    tasks_config = './config/tasks.yaml'

    @agent
    def claim_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['cliam_specialist'],
            tools=[search_claim_details_tool, search_customer_policy_details_tool],
            verbose=True,
            llm=llm
        )

    @task
    def claim_task(self) -> Task:
        return Task(
            config=self.tasks_config['claim_task'],
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Claim Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks
        )


@CrewBase
class PolicyCrew:
    """Insurance Operator Crew"""

    agents_config = './config/agents.yaml'
    tasks_config = './config/tasks.yaml'

    @agent
    def policy_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['policy_specialist'],
            tools=[search_policy_doc_tool],
            verbose=True,
            llm=llm
        )

    @task
    def policy_task(self) -> Task:
        return Task(
            config=self.tasks_config['policy_task'],
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Policy Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks
        )
