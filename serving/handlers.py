import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from serving.errors import format_errors

logger = logging.getLogger(__name__)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    problems = format_errors(exc.errors())
    logger.warning("Rejected %s %s with %d problem(s)", request.method, request.url.path, len(problems))
    return JSONResponse(status_code=400, content={"detail": problems})
