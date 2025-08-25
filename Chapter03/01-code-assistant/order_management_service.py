"""
Order Management Service (OMS) - Production-Ready Implementation
Implements DeepSeek R1's recommendations for state machine patterns, SAGA transactions, and regulatory compliance
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import structlog
from prometheus_client import Counter, Histogram, Gauge, Enum as PrometheusEnum
import json
import os
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, Column, String, DateTime, Numeric, Integer, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import hashlib
import hmac

# Structured logging
logger = structlog.get_logger()

# Prometheus metrics
ORDER_SUBMISSIONS = Counter('order_submissions_total', 'Total order submissions', ['status', 'order_type'])
ORDER_EXECUTIONS = Counter('order_executions_total', 'Total order executions', ['status', 'venue'])
ORDER_LATENCY = Histogram('order_processing_latency_seconds', 'Order processing latency')
ACTIVE_ORDERS = Gauge('active_orders_count', 'Number of active orders')
ORDER_STATE_TRANSITIONS = Counter('order_state_transitions_total', 'Order state transitions', ['from_state', 'to_state'])

# Database setup
Base = declarative_base()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/trading_db")

class OrderState(str, Enum):
    PENDING_VALIDATION = "pending_validation"
    RISK_CHECKING = "risk_checking" 
    PENDING_EXECUTION = "pending_execution"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit" 
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class ExecutionVenue(str, Enum):
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    ARCA = "arca"
    DARK_POOL = "dark_pool"

class OrderRequest(BaseModel):
    symbol: str = Field(..., pattern=r'^[A-Z]{1,5}$')
    quantity: int = Field(..., gt=0, le=1000000)
    order_type: OrderType
    side: OrderSide
    price: Optional[Decimal] = Field(None, gt=0)
    stop_price: Optional[Decimal] = Field(None, gt=0)
    time_in_force: str = Field(default="DAY", pattern=r'^(DAY|GTC|IOC|FOK)$')
    strategy_id: str = Field(..., description="Strategy identifier")
    client_order_id: Optional[str] = Field(None, description="Client-provided order ID")
    
    @validator('price')
    def price_precision(cls, v):
        if v is not None:
            return v.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        return v
    
    @validator('stop_price') 
    def stop_price_precision(cls, v):
        if v is not None:
            return v.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        return v

class OrderResponse(BaseModel):
    order_id: str
    client_order_id: Optional[str]
    state: OrderState
    symbol: str
    quantity: int
    filled_quantity: int
    remaining_quantity: int
    average_price: Optional[Decimal]
    estimated_commission: Decimal
    created_at: datetime
    updated_at: datetime
    rejection_reason: Optional[str] = None
    uti: str  # Unique Trade Identifier for MiFID II
    risk_assessment_id: Optional[str] = None

class Execution(BaseModel):
    execution_id: str
    order_id: str
    symbol: str
    quantity: int
    price: Decimal
    venue: ExecutionVenue
    executed_at: datetime
    commission: Decimal
    uti: str
    contra_party: Optional[str] = None

# Database Models
class OrderRecord(Base):
    __tablename__ = "orders"
    
    order_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_order_id = Column(String(50), nullable=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    filled_quantity = Column(Integer, default=0)
    order_type = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    price = Column(Numeric(12, 4), nullable=True)
    stop_price = Column(Numeric(12, 4), nullable=True)
    state = Column(String(30), nullable=False, index=True)
    time_in_force = Column(String(10), nullable=False)
    strategy_id = Column(String(50), nullable=False, index=True)
    average_price = Column(Numeric(12, 4), nullable=True)
    estimated_commission = Column(Numeric(8, 2), default=0)
    uti = Column(String(100), nullable=False, unique=True)
    risk_assessment_id = Column(String(100), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(50), nullable=False)
    audit_hash = Column(String(64), nullable=False)  # For immutable audit trail

class ExecutionRecord(Base):
    __tablename__ = "executions"
    
    execution_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    symbol = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(12, 4), nullable=False)
    venue = Column(String(20), nullable=False)
    executed_at = Column(DateTime, nullable=False)
    commission = Column(Numeric(8, 2), nullable=False)
    uti = Column(String(100), nullable=False, unique=True)
    contra_party = Column(String(50), nullable=True)
    audit_hash = Column(String(64), nullable=False)

@dataclass
class OrderStateMachine:
    """State machine for order lifecycle management"""
    
    VALID_TRANSITIONS = {
        OrderState.PENDING_VALIDATION: [OrderState.RISK_CHECKING, OrderState.REJECTED],
        OrderState.RISK_CHECKING: [OrderState.PENDING_EXECUTION, OrderState.REJECTED],
        OrderState.PENDING_EXECUTION: [OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED],
        OrderState.PARTIALLY_FILLED: [OrderState.FILLED, OrderState.CANCELLED],
        OrderState.FILLED: [],  # Terminal state
        OrderState.CANCELLED: [],  # Terminal state
        OrderState.REJECTED: [],  # Terminal state
        OrderState.FAILED: []  # Terminal state
    }
    
    @classmethod
    def can_transition(cls, from_state: OrderState, to_state: OrderState) -> bool:
        """Check if state transition is valid"""
        return to_state in cls.VALID_TRANSITIONS.get(from_state, [])
    
    @classmethod
    def get_valid_next_states(cls, current_state: OrderState) -> List[OrderState]:
        """Get list of valid next states"""
        return cls.VALID_TRANSITIONS.get(current_state, [])

class OrderManagementService:
    def __init__(self):
        self.redis_client = None
        self.http_client = None
        self.db_engine = None
        self.session_factory = None
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Service endpoints
        self.market_data_url = os.getenv("MARKET_DATA_SERVICE_URL", "http://localhost:8001")
        self.risk_engine_url = os.getenv("RISK_ENGINE_SERVICE_URL", "http://localhost:8002")
        
        # Order tracking
        self.active_orders: Dict[str, OrderRecord] = {}
        self._lock = threading.RLock()
        
        # Idempotency tracking
        self.idempotency_cache = {}
        
    async def startup(self):
        """Initialize OMS"""
        try:
            # Redis connection
            self.redis_client = redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                decode_responses=True
            )
            
            # HTTP client for service communication
            timeout = httpx.Timeout(30.0, connect=10.0)
            self.http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
            
            # Database connection
            self.db_engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=30)
            Base.metadata.create_all(self.db_engine)
            self.session_factory = sessionmaker(bind=self.db_engine)
            
            # Load active orders from database
            await self._load_active_orders()
            
            logger.info("Order Management Service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize OMS", error=str(e))
            raise

    async def shutdown(self):
        """Cleanup OMS"""
        if self.redis_client:
            await self.redis_client.close()
        if self.http_client:
            await self.http_client.aclose()
        if self.db_engine:
            self.db_engine.dispose()
        
        self.executor.shutdown(wait=True)
        logger.info("OMS shutdown complete")

    async def _load_active_orders(self):
        """Load active orders from database on startup"""
        try:
            with self.session_factory() as session:
                active_orders = session.query(OrderRecord).filter(
                    OrderRecord.state.in_([
                        OrderState.PENDING_VALIDATION.value,
                        OrderState.RISK_CHECKING.value,
                        OrderState.PENDING_EXECUTION.value,
                        OrderState.PARTIALLY_FILLED.value
                    ])
                ).all()
                
                with self._lock:
                    for order in active_orders:
                        self.active_orders[str(order.order_id)] = order
                
                logger.info(f"Loaded {len(active_orders)} active orders from database")
                
        except Exception as e:
            logger.error("Failed to load active orders", error=str(e))

    def generate_uti(self, order_id: str, timestamp: datetime) -> str:
        """Generate Unique Trade Identifier for MiFID II compliance"""
        # Format: FIRM_ID + TIMESTAMP + ORDER_ID_HASH
        firm_id = os.getenv("FIRM_ID", "FIRM001")
        timestamp_str = timestamp.strftime("%Y%m%d%H%M%S%f")
        order_hash = hashlib.sha256(order_id.encode()).hexdigest()[:8]
        return f"{firm_id}_{timestamp_str}_{order_hash}"

    def calculate_audit_hash(self, order_data: dict) -> str:
        """Calculate audit hash for immutable record integrity"""
        secret_key = os.getenv("AUDIT_SECRET_KEY", "default_secret").encode()
        data_string = json.dumps(order_data, sort_keys=True, default=str)
        return hmac.new(secret_key, data_string.encode(), hashlib.sha256).hexdigest()

    async def check_idempotency(self, client_order_id: str, user_id: str) -> Optional[str]:
        """Check for duplicate order submissions"""
        if not client_order_id:
            return None
            
        idempotency_key = f"idempotency:{user_id}:{client_order_id}"
        existing_order_id = await self.redis_client.get(idempotency_key)
        
        if existing_order_id:
            logger.info("Duplicate order detected", 
                       client_order_id=client_order_id, 
                       existing_order_id=existing_order_id)
            return existing_order_id
        
        return None

    async def set_idempotency(self, client_order_id: str, user_id: str, order_id: str):
        """Set idempotency key for order"""
        if client_order_id:
            idempotency_key = f"idempotency:{user_id}:{client_order_id}"
            await self.redis_client.setex(idempotency_key, 86400, order_id)  # 24 hour TTL

    async def get_market_data(self, symbol: str) -> Optional[dict]:
        """Get current market data from Market Data Service"""
        try:
            headers = {"Authorization": f"Bearer {os.getenv('API_TOKEN', 'market-data-token')}"}
            response = await self.http_client.get(
                f"{self.market_data_url}/quote/{symbol}",
                headers=headers
            )
            response.raise_for_status()
            
            market_data = response.json()
            if market_data.get("success") and market_data.get("data"):
                return market_data["data"]
            return None
            
        except Exception as e:
            logger.error("Failed to get market data", symbol=symbol, error=str(e))
            return None

    async def assess_risk(self, order_request: OrderRequest, current_price: Decimal) -> Optional[dict]:
        """Assess risk using Risk Engine Service"""
        try:
            headers = {"Authorization": f"Bearer {os.getenv('API_TOKEN', 'risk-engine-token')}"}
            risk_request = {
                "symbol": order_request.symbol,
                "current_price": float(current_price),
                "intended_quantity": order_request.quantity if order_request.side == OrderSide.BUY else -order_request.quantity,
                "trade_type": order_request.side.value,
                "strategy_id": order_request.strategy_id
            }
            
            response = await self.http_client.post(
                f"{self.risk_engine_url}/assess-risk",
                headers=headers,
                json=risk_request
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error("Failed to assess risk", symbol=order_request.symbol, error=str(e))
            return None

    async def transition_order_state(self, order_id: str, new_state: OrderState, reason: Optional[str] = None) -> bool:
        """Safely transition order state with validation"""
        try:
            with self._lock:
                order = self.active_orders.get(order_id)
                if not order:
                    logger.error("Order not found for state transition", order_id=order_id)
                    return False
                
                current_state = OrderState(order.state)
                
                if not OrderStateMachine.can_transition(current_state, new_state):
                    logger.error("Invalid state transition attempted",
                               order_id=order_id,
                               from_state=current_state.value,
                               to_state=new_state.value)
                    return False
                
                # Update order state
                old_state = order.state
                order.state = new_state.value
                order.updated_at = datetime.utcnow()
                
                if reason and new_state == OrderState.REJECTED:
                    order.rejection_reason = reason
                
                # Update in database
                with self.session_factory() as session:
                    db_order = session.query(OrderRecord).filter(
                        OrderRecord.order_id == order.order_id
                    ).first()
                    if db_order:
                        db_order.state = new_state.value
                        db_order.updated_at = datetime.utcnow()
                        if reason and new_state == OrderState.REJECTED:
                            db_order.rejection_reason = reason
                        session.commit()
                
                # Update metrics
                ORDER_STATE_TRANSITIONS.labels(from_state=old_state, to_state=new_state.value).inc()
                
                # Remove from active orders if terminal state
                if new_state in [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.FAILED]:
                    self.active_orders.pop(order_id, None)
                    ACTIVE_ORDERS.dec()
                
                logger.info("Order state transition completed",
                           order_id=order_id,
                           from_state=old_state,
                           to_state=new_state.value,
                           reason=reason)
                
                return True
                
        except Exception as e:
            logger.error("Failed to transition order state",
                        order_id=order_id,
                        to_state=new_state.value,
                        error=str(e))
            return False

    async def submit_order(self, order_request: OrderRequest, user_id: str) -> OrderResponse:
        """Submit new order with full SAGA transaction pattern"""
        start_time = datetime.utcnow()
        order_id = str(uuid.uuid4())
        
        # Step 1: Check idempotency
        if order_request.client_order_id:
            existing_order_id = await self.check_idempotency(order_request.client_order_id, user_id)
            if existing_order_id:
                ORDER_SUBMISSIONS.labels(status="duplicate", order_type=order_request.order_type.value).inc()
                raise HTTPException(status_code=409, detail=f"Order already exists: {existing_order_id}")
        
        try:
            # Step 2: Generate UTI and audit hash
            uti = self.generate_uti(order_id, start_time)
            
            # Step 3: Create order record
            order_data = {
                "order_id": order_id,
                "client_order_id": order_request.client_order_id,
                "symbol": order_request.symbol,
                "quantity": order_request.quantity,
                "order_type": order_request.order_type.value,
                "side": order_request.side.value,
                "price": float(order_request.price) if order_request.price else None,
                "stop_price": float(order_request.stop_price) if order_request.stop_price else None,
                "time_in_force": order_request.time_in_force,
                "strategy_id": order_request.strategy_id,
                "uti": uti,
                "created_by": user_id,
                "created_at": start_time.isoformat()
            }
            
            audit_hash = self.calculate_audit_hash(order_data)
            
            order_record = OrderRecord(
                order_id=uuid.UUID(order_id),
                client_order_id=order_request.client_order_id,
                symbol=order_request.symbol,
                quantity=order_request.quantity,
                order_type=order_request.order_type.value,
                side=order_request.side.value,
                price=order_request.price,
                stop_price=order_request.stop_price,
                state=OrderState.PENDING_VALIDATION.value,
                time_in_force=order_request.time_in_force,
                strategy_id=order_request.strategy_id,
                estimated_commission=Decimal('9.99'),  # Simplified commission
                uti=uti,
                created_at=start_time,
                created_by=user_id,
                audit_hash=audit_hash
            )
            
            # Step 4: Persist to database
            with self.session_factory() as session:
                session.add(order_record)
                session.commit()
                session.refresh(order_record)
            
            # Step 5: Add to active orders
            with self._lock:
                self.active_orders[order_id] = order_record
            
            ACTIVE_ORDERS.inc()
            
            # Step 6: Set idempotency key
            await self.set_idempotency(order_request.client_order_id, user_id, order_id)
            
            # Step 7: Start async processing
            asyncio.create_task(self._process_order_async(order_id, order_request))
            
            ORDER_SUBMISSIONS.labels(status="accepted", order_type=order_request.order_type.value).inc()
            
            logger.info("Order submitted successfully",
                       order_id=order_id,
                       symbol=order_request.symbol,
                       quantity=order_request.quantity,
                       uti=uti)
            
            return OrderResponse(
                order_id=order_id,
                client_order_id=order_request.client_order_id,
                state=OrderState.PENDING_VALIDATION,
                symbol=order_request.symbol,
                quantity=order_request.quantity,
                filled_quantity=0,
                remaining_quantity=order_request.quantity,
                average_price=None,
                estimated_commission=Decimal('9.99'),
                created_at=start_time,
                updated_at=start_time,
                uti=uti
            )
            
        except Exception as e:
            # SAGA Compensation: Rollback order creation
            try:
                with self.session_factory() as session:
                    session.query(OrderRecord).filter(OrderRecord.order_id == order_id).delete()
                    session.commit()
                
                with self._lock:
                    self.active_orders.pop(order_id, None)
                
            except Exception as rollback_error:
                logger.error("Failed to rollback order creation",
                           order_id=order_id,
                           rollback_error=str(rollback_error))
            
            ORDER_SUBMISSIONS.labels(status="failed", order_type=order_request.order_type.value).inc()
            logger.error("Order submission failed", order_id=order_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Order submission failed: {str(e)}")

    async def _process_order_async(self, order_id: str, order_request: OrderRequest):
        """Async order processing pipeline"""
        try:
            # Step 1: Market Data Validation
            await self.transition_order_state(order_id, OrderState.PENDING_VALIDATION)
            
            market_data = await self.get_market_data(order_request.symbol)
            if not market_data:
                await self.transition_order_state(order_id, OrderState.REJECTED, "Market data unavailable")
                return
            
            current_price = Decimal(str(market_data["price"]))
            
            # Step 2: Risk Assessment
            await self.transition_order_state(order_id, OrderState.RISK_CHECKING)
            
            risk_assessment = await self.assess_risk(order_request, current_price)
            if not risk_assessment or not risk_assessment.get("approved"):
                reason = "Risk assessment failed"
                if risk_assessment and risk_assessment.get("risk_factors"):
                    reason = f"Risk violation: {'; '.join(risk_assessment['risk_factors'])}"
                await self.transition_order_state(order_id, OrderState.REJECTED, reason)
                return
            
            # Update risk assessment ID
            with self._lock:
                order = self.active_orders.get(order_id)
                if order:
                    order.risk_assessment_id = risk_assessment.get("assessment_id", "unknown")
            
            # Step 3: Ready for Execution
            await self.transition_order_state(order_id, OrderState.PENDING_EXECUTION)
            
            # Step 4: Simulate execution (in production, this would route to exchanges)
            await self._simulate_execution(order_id, order_request, current_price)
            
        except Exception as e:
            logger.error("Order processing failed", order_id=order_id, error=str(e))
            await self.transition_order_state(order_id, OrderState.FAILED, f"Processing error: {str(e)}")

    async def _simulate_execution(self, order_id: str, order_request: OrderRequest, current_price: Decimal):
        """Simulate order execution (replace with real exchange connectivity)"""
        try:
            # Simulate execution delay
            await asyncio.sleep(0.5)
            
            # Simulate partial vs full fills
            import random
            if order_request.quantity > 1000:
                # Large orders might get partial fills
                fill_quantity = int(order_request.quantity * random.uniform(0.3, 0.8))
                await self.transition_order_state(order_id, OrderState.PARTIALLY_FILLED)
                
                # Simulate additional fills
                await asyncio.sleep(1.0)
                remaining = order_request.quantity - fill_quantity
                await self._record_execution(order_id, order_request.symbol, remaining, current_price, ExecutionVenue.NYSE)
                await self.transition_order_state(order_id, OrderState.FILLED)
            else:
                # Small orders get immediate full fills
                await self._record_execution(order_id, order_request.symbol, order_request.quantity, current_price, ExecutionVenue.NASDAQ)
                await self.transition_order_state(order_id, OrderState.FILLED)
            
            ORDER_EXECUTIONS.labels(status="filled", venue="simulated").inc()
            
        except Exception as e:
            logger.error("Execution simulation failed", order_id=order_id, error=str(e))
            await self.transition_order_state(order_id, OrderState.FAILED, f"Execution failed: {str(e)}")

    async def _record_execution(self, order_id: str, symbol: str, quantity: int, price: Decimal, venue: ExecutionVenue):
        """Record trade execution"""
        try:
            execution_id = str(uuid.uuid4())
            executed_at = datetime.utcnow()
            uti = self.generate_uti(execution_id, executed_at)
            commission = Decimal('0.005') * quantity  # $0.005 per share
            
            execution_record = ExecutionRecord(
                execution_id=uuid.UUID(execution_id),
                order_id=uuid.UUID(order_id),
                symbol=symbol,
                quantity=quantity,
                price=price,
                venue=venue.value,
                executed_at=executed_at,
                commission=commission,
                uti=uti,
                audit_hash=self.calculate_audit_hash({
                    "execution_id": execution_id,
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": float(price),
                    "executed_at": executed_at.isoformat()
                })
            )
            
            with self.session_factory() as session:
                session.add(execution_record)
                
                # Update order record
                order_record = session.query(OrderRecord).filter(
                    OrderRecord.order_id == order_id
                ).first()
                if order_record:
                    order_record.filled_quantity += quantity
                    order_record.average_price = price  # Simplified - should be weighted average
                
                session.commit()
            
            logger.info("Execution recorded",
                       execution_id=execution_id,
                       order_id=order_id,
                       symbol=symbol,
                       quantity=quantity,
                       price=float(price),
                       venue=venue.value)
            
        except Exception as e:
            logger.error("Failed to record execution",
                        order_id=order_id,
                        error=str(e))

# Global OMS instance
oms = OrderManagementService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await oms.startup()
    yield
    # Shutdown
    await oms.shutdown()

# FastAPI application
app = FastAPI(
    title="Order Management Service",
    description="Production-ready OMS with state machine and SAGA patterns",
    version="1.0.0",
    lifespan=lifespan
)

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """JWT token validation - simplified for demo"""
    if credentials.credentials != os.getenv("API_TOKEN", "oms-token"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"user_id": "trader_001", "permissions": ["trade", "view"]}

@app.post("/orders", response_model=OrderResponse)
async def submit_order_endpoint(
    order_request: OrderRequest,
    user: dict = Depends(get_current_user)
):
    """
    Submit new order for execution
    
    - **symbol**: Stock symbol
    - **quantity**: Number of shares
    - **order_type**: market, limit, stop, stop_limit
    - **side**: buy or sell
    - **price**: Limit price (required for limit orders)
    - **strategy_id**: Strategy identifier for tracking
    """
    with ORDER_LATENCY.time():
        return await oms.submit_order(order_request, user["user_id"])

@app.get("/orders/{order_id}")
async def get_order(
    order_id: str,
    user: dict = Depends(get_current_user)
):
    """Get order details by ID"""
    try:
        with oms.session_factory() as session:
            order = session.query(OrderRecord).filter(
                OrderRecord.order_id == order_id
            ).first()
            
            if not order:
                raise HTTPException(status_code=404, detail="Order not found")
            
            return {
                "order_id": str(order.order_id),
                "client_order_id": order.client_order_id,
                "state": order.state,
                "symbol": order.symbol,
                "quantity": order.quantity,
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": order.quantity - order.filled_quantity,
                "order_type": order.order_type,
                "side": order.side,
                "price": float(order.price) if order.price else None,
                "average_price": float(order.average_price) if order.average_price else None,
                "uti": order.uti,
                "created_at": order.created_at,
                "updated_at": order.updated_at,
                "rejection_reason": order.rejection_reason
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get order", order_id=order_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_orders": len(oms.active_orders),
        "database_connected": oms.db_engine is not None,
        "redis_connected": oms.redis_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
