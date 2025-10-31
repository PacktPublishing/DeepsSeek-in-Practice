import json
import boto3
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all
from botocore.exceptions import ClientError
import re

# Initialize AWS services
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ.get('USERS_TABLE', 'enterprise-api-system-users-dev'))
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# Patch all AWS SDK calls for X-Ray tracing
patch_all()

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class AuthorizationError(Exception):
    """Custom exception for authorization errors"""
    pass

class DatabaseError(Exception):
    """Custom exception for database errors"""
    pass

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for User Service operations
    Implements JWT authentication, tenant isolation, and CRUD operations
    """
    
    # Add correlation ID for request tracing
    correlation_id = context.aws_request_id
    logger.info(f"Processing request", extra={
        'correlationId': correlation_id,
        'httpMethod': event.get('httpMethod'),
        'path': event.get('path')
    })
    
    try:
        # Extract request information
        http_method = event.get('httpMethod')
        path = event.get('path', '')
        
        # Extract user context from authorizer
        authorizer_context = event.get('requestContext', {}).get('authorizer', {})
        tenant_id = authorizer_context.get('tenantId')
        user_id = authorizer_context.get('userId')
        user_role = authorizer_context.get('userRole')
        
        if not tenant_id or not user_id:
            return error_response(401, "Missing authentication context")
        
        # Route requests based on HTTP method and path
        if http_method == 'POST' and path == '/users':
            return create_user(event, tenant_id, user_id, correlation_id)
        elif http_method == 'GET' and path.startswith('/users/'):
            return get_user(event, tenant_id, user_id, user_role, correlation_id)
        elif http_method == 'PUT' and path.startswith('/users/'):
            return update_user(event, tenant_id, user_id, user_role, correlation_id)
        elif http_method == 'DELETE' and path.startswith('/users/'):
            return delete_user(event, tenant_id, user_id, user_role, correlation_id)
        elif http_method == 'GET' and path == '/users':
            return list_users(event, tenant_id, user_role, correlation_id)
        else:
            return error_response(405, "Method Not Allowed")
            
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}", extra={'correlationId': correlation_id})
        return error_response(400, str(e))
    except AuthorizationError as e:
        logger.warning(f"Authorization error: {str(e)}", extra={'correlationId': correlation_id})
        return error_response(403, str(e))
    except DatabaseError as e:
        logger.error(f"Database error: {str(e)}", extra={'correlationId': correlation_id})
        return error_response(500, "Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True, extra={'correlationId': correlation_id})
        return error_response(500, "Internal Server Error")

@xray_recorder.capture('create_user')
def create_user(event: Dict[str, Any], tenant_id: str, requesting_user_id: str, correlation_id: str) -> Dict[str, Any]:
    """Create a new user with tenant isolation and validation"""
    
    try:
        body = json.loads(event.get('body', '{}'))
    except json.JSONDecodeError:
        raise ValidationError("Invalid JSON in request body")
    
    # Validate required fields
    validate_user_data(body)
    
    # Generate unique user ID with tenant prefix
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    new_user_id = f"user_{tenant_id}_{timestamp}"
    
    # Prepare user data with security and audit fields
    user_data = {
        'userId': new_user_id,
        'tenantId': tenant_id,
        'email': body['email'].lower().strip(),
        'firstName': body['firstName'].strip(),
        'lastName': body['lastName'].strip(),
        'status': 'active',
        'createdAt': datetime.utcnow().isoformat(),
        'createdBy': requesting_user_id,
        'updatedAt': datetime.utcnow().isoformat(),
        'version': 1
    }
    
    # Add optional fields if provided
    if 'phoneNumber' in body:
        user_data['phoneNumber'] = validate_phone_number(body['phoneNumber'])
    
    if 'department' in body:
        user_data['department'] = body['department'].strip()
    
    # Add metadata for audit trail
    xray_recorder.put_metadata('user_creation', {
        'newUserId': new_user_id,
        'tenantId': tenant_id,
        'createdBy': requesting_user_id
    })
    
    try:
        # Check for duplicate email within tenant
        existing_user = check_email_exists(body['email'], tenant_id)
        if existing_user:
            raise ValidationError("Email already exists in this tenant")
        
        # Create user in DynamoDB
        table.put_item(
            Item=user_data,
            ConditionExpression='attribute_not_exists(userId)'
        )
        
        logger.info(f"User created successfully", extra={
            'correlationId': correlation_id,
            'userId': new_user_id,
            'tenantId': tenant_id
        })
        
        # Return user data without sensitive information
        response_data = {k: v for k, v in user_data.items() 
                        if k not in ['createdBy', 'version']}
        
        return success_response(response_data, 201)
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            raise ValidationError("User ID already exists")
        else:
            raise DatabaseError(f"Failed to create user: {e.response['Error']['Message']}")

@xray_recorder.capture('get_user')
def get_user(event: Dict[str, Any], tenant_id: str, requesting_user_id: str, 
             user_role: str, correlation_id: str) -> Dict[str, Any]:
    """Retrieve user with tenant isolation and RBAC"""
    
    target_user_id = event.get('pathParameters', {}).get('userId')
    if not target_user_id:
        raise ValidationError("Missing userId in path")
    
    # Check authorization - users can only access their own data unless admin/moderator
    if user_role not in ['admin', 'moderator'] and target_user_id != requesting_user_id:
        raise AuthorizationError("Insufficient permissions to access this user")
    
    # Validate user belongs to same tenant
    if not target_user_id.startswith(f"user_{tenant_id}_"):
        raise AuthorizationError("User not found in your tenant")
    
    try:
        response = table.get_item(
            Key={'userId': target_user_id},
            ProjectionExpression='userId, tenantId, email, firstName, lastName, phoneNumber, department, #status, createdAt, updatedAt',
            ExpressionAttributeNames={'#status': 'status'}
        )
        
        if 'Item' not in response:
            return error_response(404, "User not found")
        
        user_data = response['Item']
        
        # Additional tenant validation
        if user_data.get('tenantId') != tenant_id:
            raise AuthorizationError("User not found in your tenant")
        
        logger.info(f"User retrieved successfully", extra={
            'correlationId': correlation_id,
            'targetUserId': target_user_id,
            'requestingUserId': requesting_user_id
        })
        
        return success_response(user_data)
        
    except ClientError as e:
        raise DatabaseError(f"Failed to retrieve user: {e.response['Error']['Message']}")

@xray_recorder.capture('update_user')
def update_user(event: Dict[str, Any], tenant_id: str, requesting_user_id: str, 
                user_role: str, correlation_id: str) -> Dict[str, Any]:
    """Update user with validation and audit trail"""
    
    target_user_id = event.get('pathParameters', {}).get('userId')
    if not target_user_id:
        raise ValidationError("Missing userId in path")
    
    # Authorization check
    if user_role not in ['admin', 'moderator'] and target_user_id != requesting_user_id:
        raise AuthorizationError("Insufficient permissions to update this user")
    
    try:
        body = json.loads(event.get('body', '{}'))
    except json.JSONDecodeError:
        raise ValidationError("Invalid JSON in request body")
    
    # Validate tenant isolation
    if not target_user_id.startswith(f"user_{tenant_id}_"):
        raise AuthorizationError("User not found in your tenant")
    
    # Build update expression dynamically
    update_expression = "SET updatedAt = :updatedAt, updatedBy = :updatedBy"
    expression_values = {
        ':updatedAt': datetime.utcnow().isoformat(),
        ':updatedBy': requesting_user_id,
        ':tenantId': tenant_id
    }
    
    # Add updateable fields
    updateable_fields = ['firstName', 'lastName', 'phoneNumber', 'department']
    for field in updateable_fields:
        if field in body:
            if field == 'phoneNumber':
                value = validate_phone_number(body[field])
            else:
                value = body[field].strip()
            update_expression += f", {field} = :{field}"
            expression_values[f':{field}'] = value
    
    # Only admins can update status
    if 'status' in body and user_role == 'admin':
        if body['status'] in ['active', 'inactive', 'suspended']:
            update_expression += ", #status = :status"
            expression_values[':status'] = body['status']
    
    try:
        response = table.update_item(
            Key={'userId': target_user_id},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values,
            ExpressionAttributeNames={'#status': 'status'} if 'status' in body else {},
            ConditionExpression='tenantId = :tenantId',
            ReturnValues='ALL_NEW'
        )
        
        logger.info(f"User updated successfully", extra={
            'correlationId': correlation_id,
            'targetUserId': target_user_id,
            'updatedBy': requesting_user_id
        })
        
        return success_response(response['Attributes'])
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            return error_response(404, "User not found")
        else:
            raise DatabaseError(f"Failed to update user: {e.response['Error']['Message']}")

@xray_recorder.capture('delete_user')
def delete_user(event: Dict[str, Any], tenant_id: str, requesting_user_id: str, 
                user_role: str, correlation_id: str) -> Dict[str, Any]:
    """Soft delete user (GDPR compliant)"""
    
    if user_role != 'admin':
        raise AuthorizationError("Only administrators can delete users")
    
    target_user_id = event.get('pathParameters', {}).get('userId')
    if not target_user_id:
        raise ValidationError("Missing userId in path")
    
    # Validate tenant isolation
    if not target_user_id.startswith(f"user_{tenant_id}_"):
        raise AuthorizationError("User not found in your tenant")
    
    try:
        # Soft delete by updating status
        response = table.update_item(
            Key={'userId': target_user_id},
            UpdateExpression="SET #status = :status, updatedAt = :updatedAt, updatedBy = :updatedBy, deletedAt = :deletedAt",
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': 'deleted',
                ':updatedAt': datetime.utcnow().isoformat(),
                ':updatedBy': requesting_user_id,
                ':deletedAt': datetime.utcnow().isoformat(),
                ':tenantId': tenant_id
            },
            ConditionExpression='tenantId = :tenantId AND #status <> :status',
            ReturnValues='ALL_NEW'
        )
        
        logger.info(f"User deleted successfully", extra={
            'correlationId': correlation_id,
            'targetUserId': target_user_id,
            'deletedBy': requesting_user_id
        })
        
        return success_response({'message': 'User deleted successfully'})
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            return error_response(404, "User not found or already deleted")
        else:
            raise DatabaseError(f"Failed to delete user: {e.response['Error']['Message']}")

@xray_recorder.capture('list_users')
def list_users(event: Dict[str, Any], tenant_id: str, user_role: str, correlation_id: str) -> Dict[str, Any]:
    """List users with pagination and tenant isolation"""
    
    query_params = event.get('queryStringParameters') or {}
    limit = min(int(query_params.get('limit', '20')), 100)  # Max 100 items
    last_key = query_params.get('lastKey')
    
    try:
        scan_kwargs = {
            'IndexName': 'TenantIndex',
            'KeyConditionExpression': 'tenantId = :tenantId',
            'ExpressionAttributeValues': {':tenantId': tenant_id},
            'ProjectionExpression': 'userId, email, firstName, lastName, #status, createdAt',
            'ExpressionAttributeNames': {'#status': 'status'},
            'Limit': limit
        }
        
        if last_key:
            scan_kwargs['ExclusiveStartKey'] = {'userId': last_key, 'tenantId': tenant_id}
        
        response = table.query(**scan_kwargs)
        
        result = {
            'users': response['Items'],
            'count': len(response['Items'])
        }
        
        if 'LastEvaluatedKey' in response:
            result['lastKey'] = response['LastEvaluatedKey']['userId']
        
        return success_response(result)
        
    except ClientError as e:
        raise DatabaseError(f"Failed to list users: {e.response['Error']['Message']}")

# Validation functions
def validate_user_data(data: Dict[str, Any]) -> None:
    """Validate user input data"""
    required_fields = ['email', 'firstName', 'lastName']
    
    for field in required_fields:
        if field not in data or not data[field]:
            raise ValidationError(f"Missing required field: {field}")
    
    # Email validation
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, data['email']):
        raise ValidationError("Invalid email format")
    
    # Name validation
    name_regex = r'^[a-zA-Z\s\-\'\.]{1,50}$'
    if not re.match(name_regex, data['firstName']):
        raise ValidationError("Invalid first name format")
    
    if not re.match(name_regex, data['lastName']):
        raise ValidationError("Invalid last name format")

def validate_phone_number(phone: str) -> str:
    """Validate and format phone number"""
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    # Check if it's a valid international format
    if len(digits_only) < 10 or len(digits_only) > 15:
        raise ValidationError("Invalid phone number format")
    
    # Return with + prefix for international format
    return f"+{digits_only}"

def check_email_exists(email: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    """Check if email already exists in tenant"""
    try:
        response = table.query(
            IndexName='EmailIndex',
            KeyConditionExpression='email = :email',
            FilterExpression='tenantId = :tenantId AND #status <> :deleted_status',
            ExpressionAttributeValues={
                ':email': email.lower().strip(),
                ':tenantId': tenant_id,
                ':deleted_status': 'deleted'
            },
            ExpressionAttributeNames={'#status': 'status'},
            Limit=1
        )
        return response['Items'][0] if response['Items'] else None
    except ClientError:
        return None

# Response helper functions
def success_response(data: Any, status_code: int = 200) -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY'
        },
        'body': json.dumps({
            'success': True,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }, default=str)
    }

def error_response(status_code: int, message: str) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY'
        },
        'body': json.dumps({
            'success': False,
            'error': {
                'code': status_code,
                'message': message
            },
            'timestamp': datetime.utcnow().isoformat()
        })
    }
