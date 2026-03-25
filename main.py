# This is based on "Lab 6: Building a Research Paper Chatbot with Strands Agents"
# in the "Document AI: From OCR to Agentic Doc Extraction" course on DeepLearning.ai.
# More information can be found at:
# https://github.com/https-deeplearning-ai/sc-landingai
import boto3, os, json
from dotenv import load_dotenv
import pandas as pd
import lambda_helpers
from datetime import datetime
import strands
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from visual_grounding_helper import (
    extract_chunk_id_from_markdown,
    extract_chunk_image  # Using extract_chunk_image for cropped images
)

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

# Trigger on all files in input/ folder
lambda_helpers.setup_s3_trigger(
    s3_client=s3_client,
    lambda_client=lambda_client,
    bucket=os.getenv("S3_BUCKET"),
    prefix="input/",
    function_name="ade-s3-handler",
    suffix=None  # Optional: set to ".pdf" to only trigger on PDF files
)

# Upload medical documents to S3 input folder
local_folder = "medical/"

# Check if folder exists and upload
if os.path.exists(local_folder):
    count = lambda_helpers.upload_folder_to_s3(
        s3_client=s3_client,
        local_folder=local_folder,
        s3_prefix=f"input/{local_folder}",
        bucket=os.getenv("S3_BUCKET"),
        file_extensions=[".pdf", ".PDF"]
    )
    print(f"\n Waiting for automatic parsing to complete...")
    print("   (The Lambda function will automatically convert PDFs to markdown)")
else:
    print(f" Folder not found: {local_folder}")


'''stats = lambda_helpers.monitor_lambda_processing(
    logs_client=logs,
    s3_client=s3_client,
    bucket_name=os.getenv("S3_BUCKET")
)'''
# To stop monitoring, press Esc followed by double-clicking 'i'

# List all your knowledge bases
bedrock_agent = session.client("bedrock-agent")

print("All Knowledge Bases in your account:")
kb_response = bedrock_agent.list_knowledge_bases()

for kb in kb_response.get("knowledgeBaseSummaries", []):
    print(f"\nKnowledge Base: {kb['name']}")
    print(f"   ID: {kb['knowledgeBaseId']}")
    print(f"   Status: {kb['status']}")
    print(f"   Updated: {kb['updatedAt']}")

    # Get data sources for this knowledge base
    ds_response = bedrock_agent.list_data_sources(
        knowledgeBaseId=kb['knowledgeBaseId']
    )

    for ds in ds_response.get("dataSourceSummaries", []):
        print(f"   Data Source: {ds['name']}")
        print(f"      ID: {ds['dataSourceId']}")
        print(f"      Status: {ds['status']}")




response = bedrock_agent.start_ingestion_job(
    knowledgeBaseId=os.getenv("BEDROCK_KB_ID"),
    dataSourceId=os.getenv("DATA_SOURCE_ID")
)

job_id = response.get("ingestionJob", {}).get("ingestionJobId")
print("✅ Knowledge base sync initiated.")
print(f"   Job ID: {job_id}")


@strands.tool
def search_knowledge_base(query: str) -> str:
    """Search the Bedrock knowledge base for relevant medical documents with visual grounding."""
    try:
        # Ensure we have the required environment variables
        kb_id = os.getenv("BEDROCK_KB_ID")
        bucket = os.getenv("S3_BUCKET")
        if not kb_id:
            return "Error: Knowledge base ID not configured. Please set BEDROCK_KB_ID environment variable."

        # 1. Query the knowledge base using hybrid search

        # Create runtime client if needed
        bedrock_agent_runtime = session.client("bedrock-agent-runtime")

        # Query the Knowledge Base with 5 results as requested
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    "overrideSearchType": "HYBRID"  # Use hybrid search for better results
                }
            }
        )

        # Get results and sort by score (higher score = more relevant)
        raw_results = response.get("retrievalResults", [])
        sorted_results = sorted(raw_results, key=lambda x: x.get("score", 0), reverse=True)

        results = []
        seen_chunk_ids = set()  # Track seen chunk IDs to avoid duplicates

        # 2. For each result, get the location and check if this is a chunk JSON file from medical_chunks folder
        for result in sorted_results:
            content = result.get("content", {}).get("text", "")
            score = result.get("score", 0)
            location = result.get("location", {})

            # Get source file from S3 location
            s3_location = location.get("s3Location", {})
            source_uri = s3_location.get("uri", "")
            source_file = source_uri.split("/")[-1] if source_uri else "Unknown source"

            # Initialize variables
            chunk_id = None
            visual_info = None
            cropped_image_url = None
            chunk_type = "text"
            page = None
            bbox = None
            source_document = None

            # Check if this is a chunk JSON file from medical_chunks folder
            if source_file.endswith('.json') and 'chunks' in source_uri:
                try:
                    # 3. Get chunk data & extract the chunk metadata
                    # This is a chunk file - parse it directly to get all metadata
                    chunk_key = source_uri.replace(f"s3://{bucket}/", "")
                    chunk_response = s3_client.get_object(Bucket=bucket, Key=chunk_key)
                    chunk_data = json.loads(chunk_response['Body'].read().decode('utf-8'))

                    # Extract all metadata from chunk JSON
                    chunk_id = chunk_data.get('chunk_id', '')
                    chunk_type = chunk_data.get('chunk_type', 'text')
                    page = chunk_data.get('page', 0)
                    bbox = chunk_data.get('bbox', [0, 0, 1, 1])
                    source_document = chunk_data.get('source_document', '')

                    # The text might be in the chunk data or in the content
                    text = chunk_data.get('text', content)

                    # Skip if we've already seen this chunk ID
                    if chunk_id and chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk_id)

                    # 4. Generate cropped chunk image
                    if chunk_id and source_document:
                        source_pdf_key = f"input/medical/{source_document}.pdf"
                        try:
                            s3_client.head_object(Bucket=bucket, Key=source_pdf_key)
                            cropped_image_url = extract_chunk_image(
                                s3_client=s3_client,
                                bucket=bucket,
                                source_pdf_key=source_pdf_key,
                                bbox=bbox,
                                page_num=page,
                                chunk_id=chunk_id,
                                source_document=source_document,
                                highlight=True,
                                padding=10
                            )
                        except:
                            pass  # PDF not found

                except Exception as e:
                    # Fallback if can't parse chunk file
                    pass
            else:
                # Not a chunk file, try to extract chunk ID from markdown
                chunk_id = extract_chunk_id_from_markdown(content)

                # Skip if we've already seen this chunk ID
                if chunk_id and chunk_id in seen_chunk_ids:
                    continue
                if chunk_id:
                    seen_chunk_ids.add(chunk_id)

            # 5. Format result with all available information
            if cropped_image_url and chunk_id and page is not None:
                # Complete visual grounding available
                result_text = f"""
                **Source:** {source_document or source_file} (Relevance: {score:.2f})
                📄 **Chunk ID:** {chunk_id}
                📍 **Page:** {page}
                🏷️ **Chunk Type:** {chunk_type}
                🔍 **Cropped Chunk Image:** {cropped_image_url}

                **Content:**
                {content}"""
                results.append(result_text)
            elif chunk_id and page is not None:
                # Partial visual info (no image but has metadata)
                result_text = f"""
                **Source:** {source_document or source_file} (Relevance: {score:.2f})
                📄 **Chunk ID:** {chunk_id}
                📍 **Page:** {page}
                🏷️ **Chunk Type:** {chunk_type}
                📦 **Bbox:** {bbox if bbox else 'Not available'}

                **Content:**
                {content}"""
                results.append(result_text)
            else:
                # No visual grounding available - use content hash as unique ID
                content_hash = hash(content[:200])  # Hash first 200 chars for uniqueness
                if content_hash in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(content_hash)

                clean_source = source_file.replace('_grounding.json', '').replace('.json', '').replace('.md', '')
                result_text = f"""**Source:** {clean_source} (Relevance: {score:.2f})
                                **Content:**{content}"""
                results.append(result_text)

        if results:
            # Return only top 5 most relevant results with visual references
            return "\n\n---\n\n".join(results[:5])
        else:
            return f"No documents found for query: '{query}'. The knowledge base may be empty or still processing."

    except Exception as e:
        error_msg = str(e)
        if "ResourceNotFoundException" in error_msg:
            return f"Error: Knowledge base {kb_id} not found. Please verify the BEDROCK_KB_ID is correct."
        elif "ValidationException" in error_msg:
            return f"Error: Invalid query or configuration. Details: {error_msg}"
        else:
            return f"Error searching knowledge base: {error_msg}"


# Test the search function before creating agent
print("Testing knowledge base search function...")
test_result = search_knowledge_base("common cold symptoms")
print(f"Test result: {test_result[:200]}...")

if "Error" in test_result:
    print("\n Knowledge base search is not working. Checking configuration...")
    print(f"Current KB ID: {os.getenv('BEDROCK_KB_ID')}")
    print(f"Current Region: {os.getenv('AWS_REGION')}")
else:
    print("\n✅ Knowledge base search is working!")

print(test_result)

# Initialize memory client
memory_client = MemoryClient(region_name=os.getenv("AWS_REGION", "us-west-2"))

# Try to list existing memories first
try:
    existing_memories = memory_client.gmcp_client.list_memories()
    memory_list = existing_memories.get('memories', [])

    # Get all MedicalAgentMemory instances and use the most recent
    medical_memories = [m for m in memory_list if 'MedicalAgentMemory' in m.get('id', '')]

    if medical_memories:
        # Sort by creation date and take the most recent
        medical_memories.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        existing_medical_memory = medical_memories[0]
        MEMORY_ID = existing_medical_memory.get('id')
        print(f"Found {len(medical_memories)} existing MedicalAgentMemory instance(s)")
        print(f" Using most recent memory: {MEMORY_ID}")
        print(f"   Created: {existing_medical_memory.get('createdAt', 'N/A')}")
        print(f"   Status: {existing_medical_memory.get('status', 'N/A')}")
    else:
        # Only create if none exist
        raise Exception("No existing MedicalAgentMemory found, will create new one")

except Exception as e:
    # Create new memory only if none exists
    print(f" Creating new memory... (Reason: {e})")
    try:
        # Add timestamp to make name unique
        comprehensive_memory = memory_client.create_memory_and_wait(
            name=f"MedicalAgentMemory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Memory for medical document analysis with user preferences",
            strategies=[
                {
                    "summaryMemoryStrategy": {
                        "name": "SessionSummarizer",
                        "namespaces": ["/summaries/{actorId}/{sessionId}"]
                    }
                },
                {
                    "userPreferenceMemoryStrategy": {
                        "name": "PreferenceLearner",
                        "namespaces": ["/preferences/{actorId}"]
                    }
                },
                {
                    "semanticMemoryStrategy": {
                        "name": "FactExtractor",
                        "namespaces": ["/facts/{actorId}"]
                    }
                }
            ]
        )
        MEMORY_ID = comprehensive_memory.get('id')
        print(f" New memory created: {MEMORY_ID}")
    except Exception as create_error:
        print(f" Could not create memory: {create_error}")
        print("Continuing without memory functionality...")
        MEMORY_ID = None


# Set up memory configuration if memory exists
if MEMORY_ID:
    ACTOR_ID = f"medical_user_{datetime.now().strftime('%H%M%S')}"
    SESSION_ID = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"   Actor: {ACTOR_ID}")
    print(f"   Session: {SESSION_ID}")

    # Configure memory
    memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=SESSION_ID,
        actor_id=ACTOR_ID
    )

    # Create session manager
    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=memory_config,
        region_name=os.getenv("AWS_REGION", "us-west-2")
    )
else:
    session_manager = None
    print("Agent will run without memory")




















