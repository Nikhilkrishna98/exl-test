import os
import json
import csv
import email
from io import StringIO, BytesIO
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel
from dotenv import load_dotenv

import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from routing_agent import root_agent as routing_agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types


load_dotenv();  # Load environment variables from .env file
app = FastAPI(title="COLLETIONAI API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
CSV_FILE_PATH = os.environ.get('CSV_FILE_PATH')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

APP_NAME = "agent_testbed"

# NOTE: For Persistent DB
# DB_NAME = "api_orchestration_database.db"
# DB_URL = f"sqlite:///./{DB_NAME}"
# SESSION_SERVICE = DatabaseSessionService(db_url=DB_URL)

# NOTE: For InMemory DB
SESSION_SERVICE = InMemorySessionService()

ROUTING_AGENT_RUNNER = Runner(
    agent=routing_agent,
    app_name=APP_NAME,
    session_service=SESSION_SERVICE,
)

# Response Models
class CSVResponse(BaseModel):
    filename: str
    data: List[Dict[str, Any]]
    row_count: int

class EMLResponse(BaseModel):
    filename: str
    subject: str
    from_address: str
    to_address: str
    date: str
    body: str
    attachments: List[str]

class EMLListResponse(BaseModel):
    total_files: int
    emails: List[EMLResponse]

class ConfigResponse(BaseModel):
    bucket_name: str
    csv_file_path: str
    region: str

class TXTStatusResponse(BaseModel):
    file_path: str
    status: str  # 'completed' or 'inprogress'
    content: Optional[str] = None  # File content if requested

class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: str

class ChatResponse(BaseModel):
    response: str


# Initialize S3 client with environment variables
def get_s3_client():
    """Create S3 client using environment variables"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.environ.get('AWS_SESSION_TOKEN'),
            region_name=AWS_REGION
        )
        return s3_client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize S3 client: {str(e)}")
    
    
def startAgent():
    """Create S3 client using environment variables"""
    try:
       
        return "test012"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize S3 client: {str(e)}")
    
    

def validate_config():
    """Validate required configuration"""
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3_BUCKET_NAME environment variable not set")
    if not os.environ.get('AWS_ACCESS_KEY_ID'):
        raise HTTPException(status_code=500, detail="AWS_ACCESS_KEY_ID environment variable not set")
    if not os.environ.get('AWS_SECRET_ACCESS_KEY'):
        raise HTTPException(status_code=500, detail="AWS_SECRET_ACCESS_KEY environment variable not set")

def read_csv_from_s3(bucket_name: str, file_key: str) -> List[Dict[str, Any]]:
    """Read CSV file from S3 and convert to JSON"""
    s3_client = get_s3_client()
    
    try:
        # Get the file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        csv_content = response['Body'].read().decode('utf-8')
        
        # Parse CSV
        csv_reader = csv.DictReader(StringIO(csv_content))
        data = [row for row in csv_reader]
        return data
    except ClientError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")

def parse_eml_file(eml_content: bytes) -> Dict[str, Any]:
    """Parse EML file and extract information"""
    try:
        msg = email.message_from_bytes(eml_content)
        
        # Extract basic information
        email_data = {
            'subject': msg.get('Subject', ''),
            'from_address': msg.get('From', ''),
            'to_address': msg.get('To', ''),
            'date': msg.get('Date', ''),
            'body': '',
            'attachments': []
        }
        
        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                # Get body text
                if content_type == 'text/plain' and 'attachment' not in content_disposition:
                    try:
                        email_data['body'] = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        email_data['body'] = part.get_payload()
                
                # Get attachments
                if 'attachment' in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        email_data['attachments'].append(filename)
        else:
            try:
                email_data['body'] = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                email_data['body'] = msg.get_payload()
        
        return email_data
    except Exception as e:
        raise Exception(f"Error parsing EML: {str(e)}")

def find_eml_files_in_bucket(bucket_name: str, prefix: str = "") -> List[str]:
    """Find all EML files in S3 bucket"""
    s3_client = get_s3_client()
    eml_files = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        
        if prefix:
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        else:
            pages = paginator.paginate(Bucket=bucket_name)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].lower().endswith('.eml'):
                        eml_files.append(obj['Key'])
        
        return eml_files
    except ClientError as e:
        raise HTTPException(status_code=404, detail=f"Bucket not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

def check_file_exists(bucket_name: str, file_key: str) -> bool:
    s3_client = get_s3_client()
    try:
        s3_client.head_object(Bucket=bucket_name, Key=file_key)
        return True
    except ClientError:
        return False

def read_txt_from_s3(bucket_name: str, file_key: str) -> str:
    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        return response['Body'].read().decode('utf-8')
    except ClientError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading TXT: {str(e)}")

async def get_or_create_session(user_id: str, session_id: str):
    try:
        session = await SESSION_SERVICE.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        if session:
            return session
        return await SESSION_SERVICE.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception:
        return await SESSION_SERVICE.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )


async def run_agent(user_id: str, session_id: str, message: str) -> str:
    try:
        event_iterator = ROUTING_AGENT_RUNNER.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        )

        final_text = ""
        async for event in event_iterator:
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = "".join([p.text for p in event.content.parts if p.text])
                elif event.actions and event.actions.escalate:
                    final_text = f"Agent escalated: {event.error_message or 'No message.'}"
                break
        return final_text or "No response."
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# API Endpoints
@app.get("/")
def root():
    return {
        "message": "S3 File Reader API",
        "endpoints": {
            "config": "/config",
            "read_csv": "/read-csv",
            "scan_eml": "/scan-eml",
            "health": "/health"
        }
    }

@app.get("/config", response_model=ConfigResponse)
def get_config():
    """Get current configuration"""
    validate_config()
    return ConfigResponse(
        bucket_name=S3_BUCKET_NAME or "Not configured",
        csv_file_path=CSV_FILE_PATH or "Not configured",
        region=AWS_REGION
    )

@app.get("/read-csv", response_model=CSVResponse)
def read_csv_file(file_path: Optional[str] = Query(None, description="Override CSV file path")):
    """
    Read the configured CSV file from S3 and return as JSON
    
    Uses CSV_FILE_PATH from environment variables by default.
    Can be overridden with ?file_path=custom/path/file.csv
    """
    validate_config()
    
    # Use provided file_path or fall back to environment variable
    csv_path = file_path or CSV_FILE_PATH
    
    if not csv_path:
        raise HTTPException(
            status_code=400, 
            detail="CSV_FILE_PATH not configured and no file_path parameter provided"
        )
    
    data = read_csv_from_s3(S3_BUCKET_NAME, csv_path)
    
    return CSVResponse(
        filename=csv_path,
        data=data,
        row_count=len(data)
    )

@app.get("/scan-eml", response_model=EMLListResponse)
def scan_eml_files(
    limit: int = Query(100, description="Maximum number of EML files to process"),
    prefix: str = Query("", description="S3 prefix/folder to search in")
):
    """
    Scan the configured S3 bucket for EML files and extract their information
    
    Uses S3_BUCKET_NAME from environment variables.
    Optional parameters:
    - limit: Maximum number of files to process (default: 100)
    - prefix: S3 folder prefix to search in (default: root)
    """
    validate_config()
    
    # Find all EML files
    eml_files = find_eml_files_in_bucket(S3_BUCKET_NAME, prefix)
    
    if not eml_files:
        return EMLListResponse(total_files=0, emails=[])
    
    # Limit the number of files to process
    eml_files = eml_files[:limit]
    
    s3_client = get_s3_client()
    emails_data = []
    
    for file_key in eml_files:
        try:
            # Read EML file from S3
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
            eml_content = response['Body'].read()
            
            # Parse EML
            email_info = parse_eml_file(eml_content)
            email_info['filename'] = file_key
            
            emails_data.append(EMLResponse(**email_info))
        except Exception as e:
            print(f"Error processing {file_key}: {str(e)}")
            continue
    
    return EMLListResponse(
        total_files=len(emails_data),
        emails=emails_data
    )


@app.get("/get-agent-file", response_model=TXTStatusResponse)
def read_txt_file(
    file_path: str = Query(None, description="Override TXT file path"),
    jobid: str = Query(None, description="Job ID to check status"),
):
    excu_path = "Output/" + jobid + "/" + file_path
    print(excu_path)
    file_exists = check_file_exists(S3_BUCKET_NAME, excu_path)
    content = None
    if file_exists:
        content = read_txt_from_s3(S3_BUCKET_NAME, excu_path)

    return TXTStatusResponse(
        file_path=file_path,
        status="completed" if file_exists else "inprogress",
        content=content
    )


@app.get("/agent")
def get_config():
    """Get current configuration"""
    cc = startAgent()
    return {"jobid": cc}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        validate_config()
        return {"status": "healthy", "bucket": S3_BUCKET_NAME}
    except HTTPException as e:
        return {"status": "unhealthy", "error": e.detail}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    # make sure session exists
    await get_or_create_session(payload.user_id, payload.session_id)

    # run agent
    response = await run_agent(payload.user_id, payload.session_id, payload.message)
    return ChatResponse(response=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)