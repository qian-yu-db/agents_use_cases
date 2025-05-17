from typing import Any, List, Optional, Dict, Generator
from mlflow.pyfunc import ChatAgent
from mlflow.entities import SpanType
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from openai import AsyncOpenAI
import os
import mlflow
from uuid import uuid4
import asyncio
from pydantic import BaseModel
from unitycatalog.ai.core.databricks import (
    DatabricksFunctionClient,
    FunctionExecutionResult,
)
from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from agents import function_tool, RunContextWrapper
from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insurance_chat_agent")

mlflow.openai.autolog()

class UserInfo(BaseModel):
    cust_id: str | None = None
    policy_no: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None


@function_tool
def search_claims_details_by_policy_no(wrapper: RunContextWrapper[UserInfo], policy_no: str) -> FunctionExecutionResult:
    logger.info("The 'search_claims_details_by_policy_no' tool was called")
    wrapper.context.policy_no = policy_no
    client = DatabricksFunctionClient()
    return client.execute_function(
        function_name="ai.insurance_agent.search_claims_details_by_policy_no",
        parameters={"input_policy_no": wrapper.context.policy_no},
    )


@function_tool
def policy_docs_vector_search(query: str) -> FunctionExecutionResult:
    logger.info("The 'policy_docs_vector_search' tool was called")
    client = DatabricksFunctionClient()
    return client.execute_function(
        function_name="ai.insurance_agent.policy_docs_vector_search",
        parameters={"query": query},
    )

set_tracing_disabled(disabled=False)

claims_detail_retrieval_agent = Agent[UserInfo](
    name="Claims Details Retrieval Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}"
        "You are a claims details retrieval agent. "
        "If you are speaking to a customer, you probably were transferred to you from the triage agent. "
        "Use the following routine to support the customer. \n"
        "# Routine: \n"
        "1. Identify the last question asked by the customer. \n"
        "2. Use the search tools to retrieve data about a claim. Do not rely on your own knowledge. \n"
        "3. If you cannot answer the question, transfer back to the triage agent. \n"
    ),
    tools=[
        search_claims_details_by_policy_no,
    ],
    model="gpt-4o",
    #model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
)

policy_qa_agent = Agent[UserInfo](
    name="Policy Q&A Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}"
        "You are an insurance policy Q&A agent. "
        "If you are speaking to a customer, you probably were transferred to you from the triage agent. "
        "Use the following routine to support the customer.\n"
        "# Routine: \n"
        "1. Identify the last question asked by the customer. \n"
        "2. Use the search tools to answer the question about their policy. Do not rely on your own knowledge. \n"
        "3. If you cannot answer the question, transfer back to the triage agent. \n"
    ),
    tools=[policy_docs_vector_search],
    model="gpt-4o",
    #model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
)

triage_agent = Agent[UserInfo](
    name="Triage agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}"
        "You are a helpful triaging agent. "
        "You can use your tools to delegate questions to other appropriate agents. "
        "If the customer does not have anymore questions, wish them a goodbye and a good rest of their day. "
    ),
    handoffs=[claims_detail_retrieval_agent, policy_qa_agent],
    model="gpt-4o",
    #model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
)

class InsuranceChatAgent(ChatAgent):
    def __init__(self, starting_agent: Agent):
        self.starting_agent = starting_agent
        self.conversation_state = {}

    def _get_or_create_conversation_state(self, conversation_id: str):
        """Get or create the state for a conversation"""
        if conversation_id not in self.conversation_state:
            self.conversation_state[conversation_id] = {
                "current_agent": self.starting_agent,
                "conversation_history": None
            }
        return self.conversation_state[conversation_id]

    def _get_latest_user_message(self, messages: List[ChatAgentMessage]) -> str:
        """Extract the most recent user messages as input text"""
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

    def _create_user_context(
            self,
            context: Optional[ChatContext] = None,
            custom_inputs: Optional[Dict[str, Any]] = None
        ) -> UserInfo:
        """Convert MLflow inputs to UserInfo object"""
        user_info = UserInfo()

        if context:
            conversation_id = getattr(context, "conversation_id", None)
            if conversation_id:
                user_info.conversation_id = conversation_id

            user_id = getattr(context, "user_id", None)
            if user_id:
                user_info.user_id = user_id

        return user_info

    @mlflow.trace(name="insurance_chat_agent", span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None
    ) -> ChatAgentResponse:
        user_info = self._create_user_context(context, custom_inputs)
        conversation_id = user_info.conversation_id

        # Get the state for this conversation
        state = self._get_or_create_conversation_state(conversation_id)
        current_agent = state["current_agent"]
        conversation_history = state["conversation_history"]


        # Get the latest user message
        latest_message = self._get_latest_user_message(messages)

        # Prepare the input for the agent
        if conversation_history is None:
            # First turn, just use the latest message
            agent_input = latest_message
        else:
            # Add the new user message to the conversation history
            conversation_history.append({
                "role": "user",
                "content": latest_message
            })
            agent_input = conversation_history

        # Run the agent using the current event loop if available, or create one if needed
        try:
            # Use existing event loop if it exists and is running
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # No event loop in thread, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Run the agent
            result = asyncio.run_coroutine_threadsafe(
                Runner.run(
                    starting_agent=current_agent,  # Use the current agent from state
                    input=agent_input,
                    context=user_info,
                ),
                loop
            ).result()
            
            # Update the state for the next turn
            # Store the updated conversation history from the result
            state["conversation_history"] = result.to_input_list()

            # Update the current agent based on which agent was last used
            if hasattr(result, "last_agent") and result.last_agent:
                state["current_agent"] = result.last_agent
                
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise

        # Convert the result to ChatAgentResponse format:
        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    role="assistant",
                    content=result.final_output,
                    id=str(uuid4())
                )
            ]
        )

    @mlflow.trace(name="insurance_change_agent_stream", span_type=SpanType.AGENT)
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None
    ) -> Generator[ChatAgentChunk, None, None]:
        response = self.predict(messages, context, custom_inputs)

        # Yield it as a single chunk
        for message in response.messages:
            yield ChatAgentChunk(delta=message)

AGENT = InsuranceChatAgent(starting_agent=triage_agent)
mlflow.models.set_model(AGENT)
