import json
import os
import time
import logging
from typing import Dict, Any, Optional
import jwt
import requests
from jwt import PyJWKClient
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError, PyJWKClientError
from botocore.exceptions import ClientError
import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# Configuration
USER_POOL_ID = os.environ.get('USER_POOL_ID')
USER_POOL_CLIENT_ID = os.environ.get('USER_POOL_CLIENT_ID')
AWS_REGION = os.environ.get('AWS_REGION')

if not all([USER_POOL_ID, USER_POOL_CLIENT_ID, AWS_REGION]):
    raise ValueError("Missing required environment variables: USER_POOL_ID, USER_POOL_CLIENT_ID, AWS_REGION")

JWKS_URL = f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{USER_POOL_ID}/.well-known/jwks.json"
ISSUER = f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{USER_POOL_ID}"

# Initialize JWKS client with caching
jwks_client = PyJWKClient(JWKS_URL, cache_keys=True, max_cached_keys=10)

# Initialize DynamoDB for user permissions (if needed)
dynamodb = boto3.resource('dynamodb')

class AuthorizationError(Exception):
    """Custom exception for authorization errors"""
    pass

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda Authorizer for API Gateway
    Validates JWT tokens and implements RBAC with tenant isolation
    """
    
    correlation_id = context.aws_request_id
    method_arn = event.get('methodArn', '')
    
    logger.info("Authorization request received", extra={
        'correlationId': correlation_id,
        'methodArn': method_arn
    })
    
    try:
        # Extract and validate token
        token = extract_token(event.get('headers', {}))
        claims = validate_token(token)
        
        # Extract user context
        user_id = claims['sub']
        tenant_id = claims.get('custom:tenant_id')
        user_role = claims.get('custom:role', 'user')
        email = claims.get('email')
        
        logger.info("Token validated successfully", extra={
            'correlationId': correlation_id,
            'userId': user_id,
            'tenantId': tenant_id,
            'userRole': user_role
        })
        
        # Validate tenant access
        if not validate_tenant_access(user_id, tenant_id):
            logger.warning("Tenant access denied", extra={
                'correlationId': correlation_id,
                'userId': user_id,
                'tenantId': tenant_id
            })
            return generate_policy('user', 'Deny', method_arn)
        
        # Check resource permissions
        if not check_resource_permission(user_role, method_arn, user_id):
            logger.warning("Resource permission denied", extra={
                'correlationId': correlation_id,
                'userId': user_id,
                'userRole': user_role,
                'resource': method_arn
            })
            return generate_policy('user', 'Deny', method_arn)
        
        # Generate allow policy with user context
        policy = generate_policy('user', 'Allow', method_arn, {
            'userId': user_id,
            'tenantId': tenant_id or 'default',
            'userRole': user_role,
            'email': email
        })
        
        logger.info("Authorization successful", extra={
            'correlationId': correlation_id,
            'userId': user_id,
            'tenantId': tenant_id
        })
        
        return policy
        
    except ExpiredSignatureError:
        logger.warning("Token expired", extra={'correlationId': correlation_id})
        return generate_policy('user', 'Deny', method_arn)
    except InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}", extra={'correlationId': correlation_id})
        return generate_policy('user', 'Deny', method_arn)
    except AuthorizationError as e:
        logger.warning(f"Authorization error: {str(e)}", extra={'correlationId': correlation_id})
        return generate_policy('user', 'Deny', method_arn)
    except Exception as e:
        logger.error(f"Unexpected authorization error: {str(e)}", exc_info=True, extra={
            'correlationId': correlation_id
        })
        return generate_policy('user', 'Deny', method_arn)

def validate_token(token: str) -> Dict[str, Any]:
    """
    Validate JWT token against Cognito User Pool
    """
    try:
        # Get signing key from JWKS
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Decode and validate token
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=USER_POOL_CLIENT_ID,
            issuer=ISSUER,
            options={
                "verify_exp": True,
                "verify_aud": True,
                "verify_iss": True,
                "verify_signature": True
            }
        )
        
        # Additional token validation
        current_time = int(time.time())
        
        # Check token use
        if claims.get('token_use') != 'access':
            raise InvalidTokenError("Invalid token use")
        
        # Check not before time
        if 'nbf' in claims and current_time < claims['nbf']:
            raise InvalidTokenError("Token not yet valid")
        
        # Validate required custom claims for multi-tenancy
        if 'custom:tenant_id' not in claims:
            logger.warning("Token missing tenant_id claim")
            # For backward compatibility, allow tokens without tenant_id
            # but assign a default tenant
            claims['custom:tenant_id'] = 'default'
        
        if 'custom:role' not in claims:
            logger.warning("Token missing role claim")
            claims['custom:role'] = 'user'  # Default role
        
        return claims
        
    except PyJWKClientError as e:
        logger.error(f"JWKS client error: {str(e)}")
        raise InvalidTokenError("Unable to validate token signature")
    except jwt.InvalidSignatureError:
        raise InvalidTokenError("Invalid token signature")
    except jwt.ExpiredSignatureError:
        raise ExpiredSignatureError("Token has expired")
    except jwt.InvalidAudienceError:
        raise InvalidTokenError("Invalid token audience")
    except jwt.InvalidIssuerError:
        raise InvalidTokenError("Invalid token issuer")
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        raise InvalidTokenError("Token validation failed")

def validate_tenant_access(user_id: str, tenant_id: Optional[str]) -> bool:
    """
    Validate that user has access to the specified tenant
    In production, this would query a user-tenant mapping table
    """
    if not tenant_id:
        # Allow access to default tenant if no tenant specified
        return True
    
    try:
        # For demonstration, we'll assume user_id contains tenant info
        # In production, you'd query a user permissions table
        
        # Example: user_tenant_123_20241201120000
        if tenant_id in user_id or tenant_id == 'default':
            return True
        
        # Alternative: Query DynamoDB table for user-tenant mappings
        # user_permissions_table = dynamodb.Table('UserPermissions')
        # response = user_permissions_table.get_item(
        #     Key={'userId': user_id, 'tenantId': tenant_id}
        # )
        # return 'Item' in response
        
        return True  # For demo purposes, allow all access
        
    except Exception as e:
        logger.error(f"Error validating tenant access: {str(e)}")
        return False

def check_resource_permission(user_role: str, method_arn: str, user_id: str) -> bool:
    """
    Check if user role has permission to access the requested resource
    Implements the RBAC matrix from the documentation
    """
    try:
        # Parse method ARN to extract HTTP method and resource
        # Format: arn:aws:execute-api:region:account:api-id/stage/METHOD/resource
        arn_parts = method_arn.split('/')
        if len(arn_parts) < 3:
            return False
        
        http_method = arn_parts[-2]
        resource_path = '/' + '/'.join(arn_parts[-1:])
        
        # RBAC Permission Matrix based on documentation
        permissions = {
            'admin': {
                'patterns': ['*'],  # Admins have access to everything
                'description': 'Full system access'
            },
            'moderator': {
                'patterns': [
                    'GET:/users',
                    'GET:/users/*',
                    'POST:/users',
                    'PUT:/users/*',
                    'GET:/orders',
                    'GET:/orders/*',
                    'POST:/orders',
                    'PUT:/orders/*',
                    'GET:/payments/*'
                ],
                'description': 'Read/write access to users and orders, read-only payments'
            },
            'user': {
                'patterns': [
                    'GET:/users/{userId}',  # Can only access own user data
                    'PUT:/users/{userId}',  # Can only update own user data
                    'POST:/orders',         # Can create orders
                    'GET:/orders/*',        # Can view own orders (filtered by backend)
                    'GET:/payments/*'       # Can view own payments (filtered by backend)
                ],
                'description': 'Limited access to own data and order creation'
            },
            'guest': {
                'patterns': [
                    'GET:/public/*'
                ],
                'description': 'Public read-only access'
            }
        }
        
        user_permissions = permissions.get(user_role, {}).get('patterns', [])
        
        # Check for wildcard admin access
        if '*' in user_permissions:
            return True
        
        # Check specific permissions
        for pattern in user_permissions:
            if match_permission_pattern(pattern, http_method, resource_path, user_id):
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking resource permission: {str(e)}")
        return False

def match_permission_pattern(pattern: str, http_method: str, resource_path: str, user_id: str) -> bool:
    """
    Match a permission pattern against the actual request
    Handles special cases like {userId} substitution
    """
    try:
        # Split pattern into method and path
        if ':' not in pattern:
            return False
        
        pattern_method, pattern_path = pattern.split(':', 1)
        
        # Check HTTP method match
        if pattern_method != '*' and pattern_method != http_method:
            return False
        
        # Handle user-specific resource access
        if '{userId}' in pattern_path:
            # Replace {userId} with actual user ID for comparison
            expected_path = pattern_path.replace('{userId}', user_id)
            return resource_path == expected_path
        
        # Handle wildcard paths
        if pattern_path.endswith('/*'):
            pattern_prefix = pattern_path[:-2]
            return resource_path.startswith(pattern_prefix)
        
        # Exact path match
        return pattern_path == resource_path
        
    except Exception as e:
        logger.error(f"Error matching permission pattern: {str(e)}")
        return False

def extract_token(headers: Dict[str, str]) -> str:
    """
    Extract JWT token from Authorization header
    """
    # Handle case-insensitive headers
    auth_header = None
    for key, value in headers.items():
        if key.lower() == 'authorization':
            auth_header = value
            break
    
    if not auth_header:
        raise InvalidTokenError("Missing Authorization header")
    
    if not auth_header.startswith('Bearer '):
        raise InvalidTokenError("Invalid Authorization header format")
    
    token = auth_header[7:]  # Remove 'Bearer ' prefix
    if not token:
        raise InvalidTokenError("Empty token")
    
    return token

def generate_policy(principal_id: str, effect: str, resource: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate IAM policy for API Gateway
    """
    policy = {
        'principalId': principal_id,
        'policyDocument': {
            'Version': '2012-10-17',
            'Statement': [
                {
                    'Action': 'execute-api:Invoke',
                    'Effect': effect,
                    'Resource': resource
                }
            ]
        }
    }
    
    # Add context for downstream Lambda functions
    if context:
        # API Gateway has limitations on context values
        # Convert all values to strings and limit length
        sanitized_context = {}
        for key, value in context.items():
            if value is not None:
                str_value = str(value)
                # API Gateway context values must be strings and <= 1024 chars
                sanitized_context[key] = str_value[:1024] if len(str_value) > 1024 else str_value
        
        policy['context'] = sanitized_context
    
    return policy

# Health check function for monitoring
def health_check() -> Dict[str, Any]:
    """
    Health check for the authorizer
    """
    try:
        # Test JWKS endpoint connectivity
        response = requests.get(JWKS_URL, timeout=5)
        response.raise_for_status()
        
        return {
            'status': 'healthy',
            'jwks_endpoint': 'accessible',
            'timestamp': time.time()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }
