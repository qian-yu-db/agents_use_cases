import time
import yaml
import os
from databricks.sdk import WorkspaceClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("helper")


def get_shared_warehouse(name=None):
    w = WorkspaceClient()
    warehouses = w.warehouses.list()
    for wh in warehouses:
        if wh.name == name:
            return wh
    for wh in warehouses:
        if wh.name.lower() == "shared endpoint":
            return wh
    for wh in warehouses:
        if wh.name.lower() == "dbdemos-shared-endpoint":
            return wh
    # Try to fallback to an existing shared endpoint.
    for wh in warehouses:
        if "shared" in wh.name.lower():
            return wh
    for wh in warehouses:
        if "dbdemos" in wh.name.lower():
            return wh
    for wh in warehouses:
        if wh.num_clusters > 0:
            return wh
    raise Exception(
        "Couldn't find any Warehouse to use. Please create a wh first to run the demo and add the id here, or pass a name as parameter to the get_shared_warehouse(name='xxx') function"
    )


def use_and_create_db(catalog, dbName, cloud_storage_path=None):
    logger.info(f"USE CATALOG `{catalog}`")
    spark.sql(f"USE CATALOG `{catalog}`")
    spark.sql(f"""create database if not exists `{dbName}` """)


def index_exists(vsc, endpoint_name, index_full_name):
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get("status").get("ready", False)
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" not in str(e):
            logger.error(
                f"Unexpected error describing the index. This could be a permission "
                f"issue."
            )
            raise e
    return False


def endpoint_exists(vsc, vs_endpoint_name):
    try:
        return vs_endpoint_name in [
            e["name"] for e in vsc.list_endpoints().get("endpoints", [])
        ]
    except Exception as e:
        # Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            logger.info(
                "WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists"
            )
            return True
        else:
            raise e


def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
    for i in range(180):
        try:
            endpoint = vsc.get_endpoint(vs_endpoint_name)
        except Exception as e:
            # Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
            if "REQUEST_LIMIT_EXCEEDED" in str(e):
                logger.info(
                    "WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status"
                )
                return
            else:
                raise e
        status = endpoint.get("endpoint_status", endpoint.get("status"))[
            "state"
        ].upper()
        if "ONLINE" in status:
            return endpoint
        elif "PROVISIONING" in status or i < 6:
            if i % 20 == 0:
                logger.info(
                    f"Waiting for endpoint to be ready, this can take a few min... {endpoint}"
                )
            time.sleep(10)
        else:
            raise Exception(
                f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")'''
            )
    raise Exception(
        f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}"
    )


def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
    for i in range(180):
        idx = vsc.get_index(vs_endpoint_name, index_name).describe()
        index_status = idx.get("status", idx.get("index_status", {}))
        status = index_status.get(
            "detailed_state", index_status.get("status", "UNKNOWN")
        ).upper()
        url = index_status.get("index_url", index_status.get("url", "UNKNOWN"))
        if "ONLINE" in status:
            return
        if "UNKNOWN" in status:
            logger.info(
                f"Can't get the status - will assume index is ready {idx} - url: {url}"
            )
            return
        elif "PROVISIONING" in status:
            if i % 40 == 0:
                logger.info(
                    f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}"
                )
            time.sleep(10)
        else:
            raise Exception(
                f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}'''
            )
    raise Exception(
        f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}"
    )


def get_global_config(path=None):
    with open(path, "r") as f:
        global_config = yaml.safe_load(f)

    logger.info(global_config)
    return global_config


def is_running_in_databricks():
    """Detect if code is running in Databricks workspace"""
    # Check for Databricks environment variables
    databricks_env_vars = [
        "DB_CLUSTER_ID",
        "DB_IS_DRIVER",
        "DB_DRIVER_IP",
        "DATABRICKS_RUNTIME_VERSION",
    ]

    for var in databricks_env_vars:
        if var in os.environ:
            return True

    return False
