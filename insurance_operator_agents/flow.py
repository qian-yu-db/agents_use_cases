from typing import List, Optional

from crewai.flow.flow import Flow, listen, start, router, or_
from pydantic import BaseModel, Field

from crews import TriageCrew, ClaimCrew, PolicyCrew


class ChatMessage(BaseModel):
    role: str
    content: str


class InsuranceChatState(BaseModel):
    """Structured state for the insurance chatbot flow"""
    policy_number: Optional[str] = None
    current_query: str = ""
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    query_type: Optional[str] = None
    awaiting_user_input: bool = False
    current_prompt: Optional[str] = None
    response_to_user: Optional[str] = None


class InsuranceChatbotFlow(Flow[InsuranceChatState]):
    """Multi-turn insurance chatbot flow with three specialized agents"""

    @start()
    def initialize_chat(self):
        """Initialize the chatbot flow with a welcome message"""
        self.state.awaiting_user_input = True
        self.state.current_prompt = "Hello! I'm your insurance assistant. How can I help you today?"
        return self.state.current_prompt

    @listen(initialize_chat)
    def process_user_query(self):
        """Process the initial user query with the triage agent"""
        if not self.state.current_query:
            return "No query provided"

        # Add user message to conversation history
        self.state.conversation_history.append(
            ChatMessage(role="user", content=self.state.current_query)
        )

        # Execute triage task
        triage_crew = TriageCrew().crew()
        triage_result = triage_crew().kickoff(
            {"current_query": self.state.current_query,
             "conversation_history": self._format_conversation_history()}
        )

        # Store the triage result
        self.state.query_type = triage_result.strip().lower()

        return self.state.query_type

    @router(process_user_query)
    def route_to_specialist(self):
        """Route the query to the appropriate specialist based on triage result"""
        if self.state.query_type == "claim":
            return "handle_claim_query"
        elif self.state.query_type == "policy":
            return "handle_policy_query"
        else:
            return "request_clarification"

    @listen("handle_claim_query")
    def handle_claim_query(self):
        """Handle claim-related queries using the claims specialist"""
        # Check if we have policy number for claim lookups
        if not self.state.policy_number and "policy number" not in self.state.current_query.lower():
            self.state.awaiting_user_input = True
            self.state.current_prompt = ("To help with your claim, I'll need your policy number. "
                                         "What is your policy number?")
            self.state.response_to_user = self.state.current_prompt
            return self.state.response_to_user

        # process the claim query with the claims specialist
        claim_crew = ClaimCrew().crew()
        response = claim_crew().kickoff(
            {"current_query": self.state.current_query,
             "conversation_history": self._format_conversation_history(),
             "policy_number": self.state.policy_number}
        )

        # Store response and update conversation history
        self.state.response_to_user = response
        self.state.conversation_history.append(
            ChatMessage(role="assistant", content=response)
        )

        # Ready for next interaction
        self.state.awaiting_user_input = True
        return self.state.response_to_user

    @listen("handle_policy_query")
    def handle_policy_query(self):
        """Handle policy-related queries using the policy specialist"""
        # Create policy task
        policy_crew = PolicyCrew().crew()
        response = policy_crew().kickoff(
            {"current_query": self.state.current_query,
             "conversation_history": self._format_conversation_history()}
        )

        # Store response and update conversation history
        self.state.response_to_user = response
        self.state.conversation_history.append(
            ChatMessage(role="assistant", content=response)
        )

        # Ready for next interaction
        self.state.awaiting_user_input = True
        return self.state.response_to_user

    @listen("request_clarification")
    def request_clarification(self):
        """Handle general inquiries by requesting clarification"""

        clarification_crew = TriageCrew().crew()
        response = clarification_crew().kickoff(
            {"current_query": self.state.current_query,
             "conversation_history": self._format_conversation_history()}
        )

        # Store response and update conversation history
        self.state.response_to_user = response
        self.state.conversation_history.append(
            ChatMessage(role="assistant", content=response)
        )

        # Ready for next interaction
        self.state.awaiting_user_input = True
        return self.state.response_to_user

    @listen(or_(handle_claim_query, handle_policy_query, request_clarification))
    def continue_conversation(self):
        """Prepare for the next user input"""
        self.state.awaiting_user_input = True
        return "Awaiting next user input"

    def _format_conversation_history(self):
        """Format the conversation history for agent context"""
        if not self.state.conversation_history:
            return "No previous conversation."

        formatted = ""
        for message in self.state.conversation_history:
            formatted += f"{message.role.capitalize()}: {message.content}\n\n"

        return formatted.strip()
