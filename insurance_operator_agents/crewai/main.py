import asyncio
from dotenv import load_dotenv
from insurance_op_flow import InsuranceChatbotFlow
import os

load_dotenv('../.env')

DATABRICKS_HOST = os.getenv('host')
DATABRICKS_TOKEN = os.getenv('token')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def kickoff():
    """
    Run the flow.
    """
    insurance_operator_flow = InsuranceChatbotFlow()
    insurance_operator_flow.kickoff()
    print(f"flow state: {insurance_operator_flow.state}")


def plot():
    """
    Plot the flow.
    """
    insurance_operator_flow = InsuranceChatbotFlow()
    insurance_operator_flow.plot()


if __name__ == "__main__":
    kickoff()