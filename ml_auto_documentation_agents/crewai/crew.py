from crewai_tools import DirectoryReadTool, FileReadTool, FileWriterTool
from tools.custom_tools import ModelAssetCollectTool, ModelAttributesTableTool, Image2MarkdownTool, Notebook2MarkdownTool
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import sys
import os
import yaml

env_config = yaml.safe_load(open('config/env.yaml'))
llm_endpoint = env_config['llm_endpoint']
catalog = env_config['catalog']
schema = env_config['schema']
ml_model = env_config['ml_model']

llm = LLM(model=llm_endpoint)

ml_asset_collect_tool = ModelAssetCollectTool()
model_attributes_table_tool = ModelAttributesTableTool()
image_to_md_tool = Image2MarkdownTool()
notebook_to_md_tool = Notebook2MarkdownTool()
docs_tool = DirectoryReadTool(directory=f"/Volumes/{catalog}/{schema}/ml_documents/{ml_model}/")
file_read_tool = FileReadTool()
file_writer_tool = FileWriterTool()


@CrewBase
class MlDocumentCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def mlops_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['mlops_specialist'],
            tools=[ml_asset_collect_tool, 
                   model_attributes_table_tool, 
                   image_to_md_tool, 
                   notebook_to_md_tool],
            verbose=True,
            llm=llm
        )

    @agent
    def technical_document_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_document_planner'],
            tools=[docs_tool, file_read_tool],
            verbose=True,
            llm=llm
        )

    @agent
    def technical_document_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_document_writer'],
            verbose=True,
            tools = [file_read_tool],
            llm=llm
        )

    @agent 
    def technical_document_editor(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_document_editor'],
            verbose=True,
            tools=[file_read_tool, file_writer_tool],
            llm=llm
        )

    @task 
    def ml_document_content_organization_task(self) -> Task: 
        return Task(
            config=self.tasks_config['ml_document_content_organization_task']
        )

    @task 
    def ml_document_planning_outline_task(self) -> Task:
        return Task(
            config=self.tasks_config['ml_document_planning_outline_task']
        )

    @task
    def ml_document_write_task(self) -> Task:
        return Task(
            config=self.tasks_config['ml_document_write_task'],
            output_file=f"/Volumes/{catalog}/{schema}/ml_documents/{ml_model}/ml_document_draft.md"
        )

    @task 
    def ml_document_edit_publish_task(self) -> Task:
        return Task(
            config=self.tasks_config['ml_document_edit_publish_task'],
            output_file=f"/Volumes/{catalog}/{schema}/ml_documents/{ml_model}/ml_document_final.md"
        )

    @crew
    def crew(self) -> Crew: 
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False
        )
