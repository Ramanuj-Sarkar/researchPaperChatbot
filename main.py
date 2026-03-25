# This is based on "Lab 6: Building a Research Paper Chatbot with Strands Agents"
# in the "Document AI: From OCR to Agentic Doc Extraction" course on DeepLearning.ai.
# More information can be found at:
# https://github.com/https-deeplearning-ai/sc-landingai
import boto3, os, json
from dotenv import load_dotenv
import pandas as pd
import lambda_helpers

load_dotenv()

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)


# Create clients
s3_client = session.client("s3")
lambda_client = session.client("lambda")
iam = session.client("iam")
logs = session.client("logs")
bedrock_agent_runtime = session.client("bedrock-agent-runtime")
bedrock_runtime = session.client("bedrock-runtime")

source_files = ["ade_s3_handler.py"]
requirements = ["landingai-ade", "typing-extensions"]

zip_path = lambda_helpers.create_deployment_package(
    source_files=source_files,
    requirements=requirements,
    output_zip="ade_lambda.zip",
    package_dir="ade_package"
)



role_arn = lambda_helpers.create_or_update_lambda_role(
    iam_client=iam,
    role_name="lambda-ade-exec-role",
    description="Execution role for LandingAI ADE Lambda"
)



env_vars = {
    "VISION_AGENT_API_KEY": os.getenv("VISION_AGENT_API_KEY"),
    "ADE_MODEL": "dpt-2-latest",
    "INPUT_FOLDER": "input/",
    "OUTPUT_FOLDER": "output/",
    "S3_BUCKET": os.getenv("S3_BUCKET"),
    "FORCE_REPROCESS": "false"  # Set to "true" to reprocess all files even if outputs exist
}

response = lambda_helpers.deploy_lambda_function(
    lambda_client=lambda_client,
    function_name="ade-s3-handler",
    zip_file="ade_lambda.zip",
    role_arn=role_arn,
    handler="ade_s3_handler.ade_handler",
    env_vars=env_vars,
    runtime="python3.10",
    timeout=900,
    memory_size=1024
)



















