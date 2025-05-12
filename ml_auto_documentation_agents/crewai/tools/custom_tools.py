from crewai.tools import BaseTool
from mlflow.tracking import MlflowClient
import json
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
from mlflow import MlflowClient
import mlflow
from json2html import json2html
from markdownify import markdownify as md
import nbformat
from nbconvert import MarkdownExporter

w = WorkspaceClient()

class ModelAssetCollectTool(BaseTool):
    name: str = "ml_asset_collect_tool"
    description: str = "Collect model assets use Mlflow API and save to a UC volume"

    def _run(self, catalog: str, uc_schema: str, model: str) -> str:
        mlflow.set_registry_uri("databricks-uc")
        client = MlflowClient()
        model_full_name = f"{catalog}.{uc_schema}.{model}"
        model_version = client.get_model_version_by_alias(name=model_full_name, alias="production")
        run_id = model_version.run_id
        volume_path = f"/Volumes/{catalog}/{uc_schema}/ml_documents"
        destination_path = self.create_artifact_store_folder(volume_path=volume_path, folder_name=model)
        mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/", dst_path=destination_path)
        return destination_path

    @staticmethod
    def create_artifact_store_folder(volume_path, folder_name):
        folder_path = os.path.join(volume_path, folder_name)

        # Check if the folder exists and create it if it does not
        if not os.path.isdir(folder_path):
            try:
                os.makedirs(folder_path)
                print(f"PASS: Folder `{folder_path}` created")
            except PermissionDenied:
                print(f"FAIL: No permissions to create folder `{folder_path}`")
                raise ValueError(f"No permissions to create folder `{folder_path}`")
        else:
            print(f"PASS: Folder `{folder_path}` already exists")
        return folder_path
        

class ModelAttributesTableTool(BaseTool):
    name: str = "model_attibute_to_markdown_tool"
    description: str = "find run_id based on a registered UC model and use mlflow client to write to a Markdown file with params and metrics"

    def _run(self, catalog: str, uc_schema: str, model: str) -> str:

        client = MlflowClient()
        model_full_name = f"{catalog}.{uc_schema}.{model}"
        dst_path = f"/Volumes/{catalog}/{uc_schema}/ml_documents/{model}"
        model_version = client.get_model_version_by_alias(name=model_full_name, alias="production")
        run_id = model_version.run_id
        run = client.get_run(run_id)

        # model properties
        model_flattened_json = json.dumps({
            **run.data.params,
            **run.data.metrics,
            **run.data.tags
        }, indent=4)
        model_html = json2html.convert(json.loads(model_flattened_json))

        # run properties
        run_info_dict = {key: value for key, value in run.info.__dict__.items()}
        run_info_html = json2html.convert(run_info_dict)

        # data source
        dataset_input = run.inputs.to_dictionary()['dataset_inputs'][0]
        data_source = dataset_input['dataset']
        data_source_html = json2html.convert(data_source)

        # consolidated information
        consolidated_md = (f"# model algorithm, model parameters, and model metrics table\n"
                        f"{md(model_html)}\n"
                        f"# model run information table\n"
                        f"{md(run_info_html)}\n"
                        f"# data source table\n"
                        f"{md(data_source_html)}")

        with open(f"{dst_path}/model_attribute_tables.md", "w") as file:
            print(f"writing model attributes tables to {dst_path}/model_attribute_tables.md")
            file.write(consolidated_information)

        return f"{dst_path}/model_attribute_tables.md"
    

class Image2MarkdownTool(BaseTool):
    name: str = "image_to_markdown_tool"
    description: str = "Get PNG files in a folder and create a markdown file with links to the PNG files using the file names as labels." 

    def _run(self, catalog: str, schema: str, model: str) -> str:
        folder_path = f"/Volumes/{catalog}/{schema}/ml_documents/{model}"
        png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
        markdown_content = "\n\n".join([f"![{os.path.splitext(f)[0]}]({os.path.join(folder_path, f)})" for f in png_files])
        output_file = f"{folder_path}/visuals.md"
    
        with open(output_file, 'w') as md_file:
            md_file.write(markdown_content)
        
        return output_file
        

class Notebook2MarkdownTool(BaseTool):
    name: str = "notebook_to_markdown_tool"
    description: str = "Get a notebook file (ipynb or py) and create a markdown file with the notebook content."

    def _run(self, catalog: str, schema: str, model: str) -> str:
        notebook_path = f"/Volumes/{catalog}/{schema}/ml_documents/{model}/notebooks"
        output_path = f"/Volumes/{catalog}/{schema}/ml_documents/{model}/"

        files = [f for f in os.listdir(notebook_path) if f.endswith(".ipynb") or f.endswith(".py")]
        for f in files:
            if f.endswith(".ipynb"):
                with open(f"{notebook_path}/{f}", 'r', encoding='utf-8') as nb_file:
                    notebook_content = nbformat.read(nb_file, as_version=4)
            else:
                with open(f"{notebook_path}/{f}", 'r', encoding='utf-8') as py_file:
                    notebook_content = py_file.read()
            
            # Convert to markdown
            markdown_exporter = MarkdownExporter()
            markdown_content, _ = markdown_exporter.from_notebook_node(notebook_content)
            
            # Write to markdown file
            output_file = f"{output_path}/{f.replace('.ipynb', '.md')}"
            print(output_file)
            with open(output_file, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)






