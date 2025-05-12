import yaml
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai.tools import BaseTool

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


class HumanInputTool(BaseTool):
    name: str = "Human interact"
    description: str = (
        "As user questions and gather user response"
    )

    def _run(self, argument: str) -> str:
        print(argument)
        res = input(f"{argument} \n")
        return res


human_input = HumanInputTool()


# Create Crews
class TriageCrew:
    def triage_agent(self) -> Agent:
        return Agent(
            role="triage agent",
            goal="analysis customer query and route to appropriate specialist, if you cannot "
                 "identify the query, ask user for clarification",
            backstory="The triage agent is responsible for analyzing the customer query and "
                      "routing it to the appropriate specialist. If the query cannot be "
                      "identified, ask the user for clarification",
            verbose=True,
            tools=[human_input],
            llm=llm
        )

    def traige_task(self) -> Task:
        return Task(
            description="Analyze the customer query: {query} route to appropriate "
                        "specialist",
            expected_output="return the word claim, policy, or unknown",
            agent=self.triage_agent(),
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[self.triage_agent()],
            tasks=[self.traige_task()],
            memory=True,
            verbose=True
        )


class ClaimCrew:
    def claim_agent(self) -> Agent:
        return Agent(
            role="claim agent",
            goal="Provide accurate and helpful information about customer claims",
            backstory=""" You are an insurance claims specialist. Your role is to:
                1. Process customer inquiries related to claims
                2. Look up customer profiles to retrieve claim information
                3. Provide status updates on existing claims
                4. Guide customers through the claims process""",
            verbose=True,
            tools=[search_claim_details_tool, search_customer_policy_details_tool],
            llm=llm
        )

    def claim_task(self) -> Task:
        return Task(
            description=(
                """Process this claim-related customer request based on: 
                query: {current_query}
                Previous conversation: {conversation_history}

                If you have a policy number,
                - use the search_policy_details_by_policy_number tool to retrieve customer profile 
                  and policy information.
                - use the search_claim_by_policy_number tool to retrieve existing claim information.
                
                Provide a helpful, detailed response about their claim.If you don't have their
                policy_number, explain that you need it to access their claim information."""
            ),
            expected_output="A helpful response to the customer's claim query",
            agent=self.claim_agent(),
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[self.claim_agent()],
            tasks=[self.claim_task()],
            memory=True,
            verbose=True
        )


class PolicyCrew:
    def policy_agent(self) -> Agent:
        return Agent(
            role="policy agent",
            goal="handle policy related queries",
            backstory="The policy agent is responsible for handling policy related queries",
            verbose=True,
            tools=[search_policy_doc_tool],
            llm=llm
        )

    def policy_task(self) -> Task:
        return Task(
            description="Answer the policy related query: {query}",
            expected_output="answer the policy related queries based on the policy document",
            agent=self.policy_agent(),
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[self.policy_agent()],
            tasks=[self.policy_task()],
            memory=True,
            verbose=True
        )
