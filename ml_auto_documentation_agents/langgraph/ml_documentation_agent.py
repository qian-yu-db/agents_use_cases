from typing import Any, Generator, Optional, Sequence, Union, TypedDict, Dict, List
import mlflow
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from model_artifacts_organizer import ModelArtifactOrganizer
from file_management_utils import recursive_file_loader

mlflow.langchain.autolog()

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
LLM = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# Prompts
WRITTER_PROMPT = """
Role and Purpose:
You are a specialized ML Documentation Writer that transforms MLflow artifacts into comprehensive, well-structured markdown documentation. Your task is to create clear, concise, and informative documentation about machine learning models that will be used by other data scientists and model users.

Input Format
I will provide you with the following MLflow artifacts:

- Model metrics (accuracy, precision, recall, etc.)
- Model information (algorithm type, version, etc.)
- Environment details (dependencies, packages, versions)
- Data source information (dataset details, preprocessing steps)
- Paths to visualization artifacts (PNG plots of metrics, predictions, etc.)

Any additional relevant information about the model such as notebook codes

Output Structure
Create a markdown document with the following sections:

- Model Overview
- Model name and version
- Brief description of the model's purpose
- Key capabilities and use cases
- Model Specifications
- Algorithm type and architecture
- Hyperparameters
- Framework and dependencies
- Training environment details and Training Data
- Dataset description and source
- Quantitative metrics (accuracy, precision, recall, F1, etc.)
- Visualization references (link plots using relative paths)
- Model Artifacts

Formatting Guidelines:

- Use level 2 headers (##) for main sections and bolding (**) for subsections
- Include relevant code snippets in markdown code blocks with appropriate language highlighting if available
- Embed visualization references using markdown image syntax
- Use tables for comparing metrics or features where appropriate
- For mathematical expressions, use LaTeX syntax with for inline and for block formulas
- Use bullet points and numbered lists for clarity
- Keep language clear and concise, avoiding unnecessary jargon

Best Practices
- Maintain a consistent writing style throughout the document
- Be concise but thorough in your explanations
- Include relevant plots and visualizations using markdown image syntax
- Make sure plots and tables are properly described and labeled
- Define technical terms when first introduced
-Focus on information that helps others understand and use the model effectively
"""

REVIEWER_PROMPT = """
Role and Purpose:
You are a professional proofreading expert tasked with meticulously reviewing documents and producing polished final versions. Your objective is to transform draft content into error-free, coherent, and professionally styled documents while preserving the author's original voice and intent.

Review Process Instructions:
When presented with a document, perform these comprehensive checks:

- Grammatical Analysis
- Identify and correct all grammatical mistakes
- Ensure proper sentence structure
- Verify correct verb tense consistency
- Check subject-predicate agreement
- Review pronoun usage and reference clarity

Mechanical Corrections:

- Fix spelling errors and typos
- Correct punctuation issues
- Standardize capitalization
- Address formatting inconsistencies
- Ensure proper citation format if applicable

Content Enhancement

- Improve sentence flow and transitions between paragraphs
- Enhance clarity by simplifying complex sentences when necessary
- Eliminate redundancies and unnecessary words
- Strengthen weak phrasing
- Ensure logical progression of ideas

Style and Tone Assessment

- Maintain consistent tone throughout
- Unify terminology and notation
- Ensure appropriate formality level for the document's purpose
- Check for consistent point of view
- Verify style guide compliance if specified

Document Structure Review
- Confirm proper formatting of headings and subheadings
- Verify logical organization of sections
- Check that bullet points and numbered lists follow consistent structure
- Ensure table and figure references are accurate
- Review overall document layout

Guidelines
- Preserve the author's original voice and meaning
- Make only necessary changes without altering core content
- When multiple correction options exist, choose the one that best maintains the document's original style
- For specialized terminology or jargon, verify correctness before making changes
- If something is ambiguous, provide alternative interpretations and recommendations
- Ensure the final document maintains consistent formatting throughout
- For substantive restructuring needs, highlight recommendations separately rather than implementing directly
"""


# Step 1
def collect_ml_document_content(state: ChatAgentState) -> dict:
    """Process the content organization."""
    model_artifacts_organizer = ModelArtifactOrganizer(
        catalog=state["custom_inputs"]["catalog"],
        schema=state["custom_inputs"]["schema"],
        model=state["custom_inputs"]["model"],
    )
    # Collect ML model assets
    artifact_volume_path = model_artifacts_organizer.collect_mlflow_artifacts()

    # Create model attributes table, notebook, and image markdown
    model_artifacts_organizer.create_model_attributes_md()
    model_artifacts_organizer.notebook_to_md()
    model_artifacts_organizer.image_file_to_md()

    # Collect content source files
    doc_contents = recursive_file_loader(artifact_volume_path)
    messages = [
        {"role": "system", "content": "You are a MLops experts"},
        {
            "role": "user",
            "content": "Collected ML model assets, generated model "
            "attributes table, notebook, and image markdown.",
        },
    ]

    # Create a summary of all files
    files_content = "\n\n".join(
        [
            f"""File: {doc.metadata['relative_path']}
                 File Type: {doc.metadata['file_type']}
                 <content>\n{doc.page_content}\n</content>"""
            for doc in doc_contents
        ]
    )
    custom_outputs = state.get("custom_outputs", {}).copy()
    custom_outputs["source_contents"] = files_content
    state["custom_outputs"] = {}
    state["custom_outputs"]["source_contents"] = files_content
    messages.append({"role": "assistant", "content": files_content})

    return {"messages": messages, "custom_outputs": custom_outputs}


# Step 2
def write_doc_draft(state: ChatAgentState) -> dict:
    """write the ml document draft"""

    files_content = state.get("custom_outputs", {}).get(
        "source_contents", "No content found"
    )
    content_message = {
        "role": "user",
        "content": f"Here is the content of the files: {files_content}",
    }

    # Generate the outline
    messages = [{"role": "system", "content": WRITTER_PROMPT}] + [content_message]
    response = LLM.invoke(messages)

    custom_outputs = state.get("custom_outputs", {}).copy()
    custom_outputs["document_draft"] = response.content
    state["custom_outputs"]["document_draft"] = response.content

    return {
        "messages": [{"role": "assistant", "content": response.content}],
        "custom_outputs": custom_outputs,
    }


# Step 3
def review_doc_draft(state: ChatAgentState, config: RunnableConfig) -> dict:
    """review the ml document draft and write the final version"""

    doc_draft = state.get("custom_outputs", {}).get(
        "document_draft", "No draft available"
    )
    draft_message = {
        "role": "user",
        "content": f"Here is the draft of the ml document: {doc_draft}",
    }
    # Generate the outline
    messages = [{"role": "system", "content": REVIEWER_PROMPT}] + [draft_message]
    response = LLM.invoke(messages)

    custom_outputs = state.get("custom_outputs", {}).copy()
    custom_outputs["final_document"] = response.content
    state["custom_outputs"]["final_document"] = response.content

    return {
        "messages": [{"role": "assistant", "content": response.content}],
        "custom_outputs": custom_outputs,
    }

# Build a Graph
workflow = StateGraph(ChatAgentState)

# Create nodes
workflow.add_node("collect_ml_document_content", collect_ml_document_content)
workflow.add_node("write_doc_draft", RunnableLambda(write_doc_draft))
workflow.add_node("review_doc_draft", RunnableLambda(review_doc_draft))

# Create edges
workflow.set_entry_point("collect_ml_document_content")
workflow.add_edge("collect_ml_document_content", "write_doc_draft")
workflow.add_edge("write_doc_draft", "review_doc_draft")
workflow.add_edge("review_doc_draft", END)

auto_ml_doc_flow = workflow.compile()

# wrap it at a mlflow ChatAgent to enable agent framework
class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        if context:
            request["context"] = context

        if custom_inputs:
            request["custom_inputs"] = custom_inputs

        messages = []
        custom_outputs = {}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
                if "custom_outputs" in node_data:
                    custom_outputs.update(node_data["custom_outputs"])

        return ChatAgentResponse(messages=messages, custom_outputs=custom_outputs)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}

        if context:
            request["context"] = context

        if custom_inputs:
            request["custom_inputs"] = custom_inputs

        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )

AGENT = LangGraphChatAgent(auto_ml_doc_flow)
mlflow.models.set_model(AGENT)
