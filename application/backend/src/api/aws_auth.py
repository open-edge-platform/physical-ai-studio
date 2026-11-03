"""AWS named profile helpers for provider setup on the backend host."""

import asyncio

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from schemas.aws_auth import AwsAuthCapabilities, AwsCredentialSelection, AwsIdentity
from schemas.operational_issue import OperationalIssue
from services import aws_auth_service

router = APIRouter(prefix="/providers/aws", responses={400: {"model": OperationalIssue}})


@router.get("/auth", response_model=AwsAuthCapabilities)
async def aws_auth_capabilities() -> AwsAuthCapabilities | JSONResponse:
    """List available profile names without authenticating."""
    try:
        return await aws_auth_service.get_capabilities()
    except (BotoCoreError, ClientError) as error:
        return JSONResponse(status_code=400, content=aws_auth_service.operational_aws_error(error).model_dump())


@router.post("/credentials/check", response_model=AwsIdentity)
async def check_aws_credentials(selection: AwsCredentialSelection) -> AwsIdentity | JSONResponse:
    """Verify the selected AWS named profile without requiring the AWS CLI."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(aws_auth_service.check_credentials, selection), timeout=20)
    except (BotoCoreError, ClientError, TimeoutError) as error:
        return JSONResponse(
            status_code=400, content=aws_auth_service.operational_aws_error(error, selection.profile_name).model_dump()
        )
