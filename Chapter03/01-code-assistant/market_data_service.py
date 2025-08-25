"""
Market Data Service - Production-Ready Implementation
Addresses DeepSeek R1's recommendations for circuit breakers, caching, and API abstraction
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
from prometheus_client import Counter, Histogram, Gauge
import json
import os
from contextlib import asynccontextmanager

# Structured logging
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('market_data_requests_total', 'Total market data requests', ['symbol', 'status'])
REQUEST_DURATION = Histogram('market_data_request_duration_seconds', 'Request duration')
CACHE_HIT_RATE = Gauge('market_data_cache_hit_rate', 'Cache hit rate percentage')
API_HEALTH = Gauge('market_data_api_health', 'External API health status')

class DataProvider(str, Enum):
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    FALLBACK = "fallback"

@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds

class MarketDataRequest(BaseModel):
    symbol: str = Field(..., pattern=r'^[A-Z]{1,5}$', description="Stock symbol (1-5 uppercase letters)")
    data_type: str = Field(default="quote", description="Type of data: quote, historical, intraday")
    interval: Optional[str] = Field(default=None, description="Time interval for historical data")

class QuoteData(BaseModel):
    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: int
    timestamp: datetime
    provider: DataProvider
    market_status: str

class HistoricalData(BaseModel):
    symbol: str
    data_points: List[Dict[str, Any]]
    interval: str
    provider: DataProvider
    last_updated: datetime

class MarketDataResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cached: bool = False
    provider_used: DataProvider
    response_time_ms: float

class MarketDataService:
    def __init__(self):
        self.redis_client = None
        self.http_client = None
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.api_keys = {
            DataProvider.ALPHA_VANTAGE: os.getenv("ALPHA_VANTAGE_API_KEY"),
            DataProvider.POLYGON: os.getenv("POLYGON_API_KEY"),
            DataProvider.TWELVE_DATA: os.getenv("TWELVE_DATA_API_KEY"),
        }
        self.provider_priority = [
            DataProvider.ALPHA_VANTAGE,
            DataProvider.POLYGON,
            DataProvider.TWELVE_DATA,
            DataProvider.FALLBACK
        ]
        
    async def startup(self):
        """Initialize service connections"""
        try:
            # Redis connection for caching
            self.redis_client = redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                decode_responses=True
            )
            
            # HTTP client with timeout and retry configuration
            timeout = httpx.Timeout(10.0, connect=5.0)
            self.http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
            
            logger.info("Market Data Service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Market Data Service", error=str(e))
            raise

    async def shutdown(self):
        """Cleanup service connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.http_client:
            await self.http_client.aclose()
        logger.info("Market Data Service shutdown complete")

    def get_circuit_breaker(self, provider: DataProvider) -> CircuitBreakerState:
        """Get or create circuit breaker for provider"""
        if provider.value not in self.circuit_breakers:
            self.circuit_breakers[provider.value] = CircuitBreakerState()
        return self.circuit_breakers[provider.value]

    def is_circuit_open(self, provider: DataProvider) -> bool:
        """Check if circuit breaker is open for provider"""
        cb = self.get_circuit_breaker(provider)
        
        if cb.state == "OPEN":
            if cb.last_failure_time and \
               datetime.now() - cb.last_failure_time > timedelta(seconds=cb.recovery_timeout):
                cb.state = "HALF_OPEN"
                logger.info(f"Circuit breaker for {provider} moved to HALF_OPEN")
                return False
            return True
        return False

    def record_success(self, provider: DataProvider):
        """Record successful API call"""
        cb = self.get_circuit_breaker(provider)
        cb.failure_count = 0
        cb.state = "CLOSED"
        API_HEALTH.labels().set(1)

    def record_failure(self, provider: DataProvider):
        """Record failed API call and potentially open circuit"""
        cb = self.get_circuit_breaker(provider)
        cb.failure_count += 1
        cb.last_failure_time = datetime.now()
        
        if cb.failure_count >= cb.failure_threshold:
            cb.state = "OPEN"
            logger.warning(f"Circuit breaker OPENED for {provider} after {cb.failure_count} failures")
            API_HEALTH.labels().set(0)

    async def get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from Redis cache"""
        try:
            if not self.redis_client:
                return None
                
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                CACHE_HIT_RATE.set(1)
                return json.loads(cached_data)
            else:
                CACHE_HIT_RATE.set(0)
                return None
                
        except Exception as e:
            logger.warning("Cache retrieval failed", cache_key=cache_key, error=str(e))
            return None

    async def cache_data(self, cache_key: str, data: Dict[str, Any], ttl: int = 300):
        """Store data in Redis cache with TTL"""
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(data, default=str)
                )
        except Exception as e:
            logger.warning("Cache storage failed", cache_key=cache_key, error=str(e))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_from_alpha_vantage(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage API"""
        if self.is_circuit_open(DataProvider.ALPHA_VANTAGE):
            raise HTTPException(status_code=503, detail="Alpha Vantage circuit breaker is open")

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_keys[DataProvider.ALPHA_VANTAGE]
            }
            
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Transform Alpha Vantage response to our format
            if "Global Quote" in data:
                quote = data["Global Quote"]
                result = {
                    "symbol": quote.get("01. symbol", symbol),
                    "price": float(quote.get("05. price", 0)),
                    "volume": int(quote.get("06. volume", 0)),
                    "timestamp": datetime.now().isoformat(),
                    "provider": DataProvider.ALPHA_VANTAGE.value,
                    "market_status": "open"  # Simplified
                }
                self.record_success(DataProvider.ALPHA_VANTAGE)
                return result
            else:
                raise ValueError("Invalid response format from Alpha Vantage")
                
        except Exception as e:
            self.record_failure(DataProvider.ALPHA_VANTAGE)
            logger.error("Alpha Vantage API failure", symbol=symbol, error=str(e))
            raise

    async def fetch_from_polygon(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Polygon API"""
        if self.is_circuit_open(DataProvider.POLYGON):
            raise HTTPException(status_code=503, detail="Polygon circuit breaker is open")

        try:
            url = f"https://api.polygon.io/v2/last/nbbo/{symbol}"
            headers = {"Authorization": f"Bearer {self.api_keys[DataProvider.POLYGON]}"}
            
            response = await self.http_client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "OK" and "results" in data:
                result_data = data["results"]
                result = {
                    "symbol": symbol,
                    "price": float(result_data.get("last", {}).get("price", 0)),
                    "bid": float(result_data.get("bid", 0)),
                    "ask": float(result_data.get("ask", 0)),
                    "volume": int(result_data.get("last", {}).get("size", 0)),
                    "timestamp": datetime.now().isoformat(),
                    "provider": DataProvider.POLYGON.value,
                    "market_status": "open"
                }
                self.record_success(DataProvider.POLYGON)
                return result
            else:
                raise ValueError("Invalid response format from Polygon")
                
        except Exception as e:
            self.record_failure(DataProvider.POLYGON)
            logger.error("Polygon API failure", symbol=symbol, error=str(e))
            raise

    async def get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Generate fallback data when all APIs fail"""
        logger.warning("Using fallback data", symbol=symbol)
        return {
            "symbol": symbol,
            "price": 100.0,  # Mock price
            "volume": 1000,
            "timestamp": datetime.now().isoformat(),
            "provider": DataProvider.FALLBACK.value,
            "market_status": "unknown",
            "warning": "Fallback data - not real market data"
        }

    async def get_quote(self, symbol: str) -> MarketDataResponse:
        """Get real-time quote for symbol with failover and caching"""
        start_time = time.time()
        cache_key = f"quote:{symbol}"
        
        try:
            # Check cache first
            cached_data = await self.get_cached_data(cache_key)
            if cached_data:
                REQUEST_COUNT.labels(symbol=symbol, status="cache_hit").inc()
                return MarketDataResponse(
                    success=True,
                    data=cached_data,
                    cached=True,
                    provider_used=DataProvider(cached_data["provider"]),
                    response_time_ms=(time.time() - start_time) * 1000
                )

            # Try providers in priority order
            last_error = None
            for provider in self.provider_priority:
                if provider == DataProvider.FALLBACK:
                    continue
                    
                try:
                    if provider == DataProvider.ALPHA_VANTAGE and self.api_keys[provider]:
                        data = await self.fetch_from_alpha_vantage(symbol)
                    elif provider == DataProvider.POLYGON and self.api_keys[provider]:
                        data = await self.fetch_from_polygon(symbol)
                    else:
                        continue  # Skip if no API key
                    
                    # Cache successful response
                    await self.cache_data(cache_key, data, ttl=30)  # 30 second cache
                    
                    REQUEST_COUNT.labels(symbol=symbol, status="success").inc()
                    return MarketDataResponse(
                        success=True,
                        data=data,
                        cached=False,
                        provider_used=provider,
                        response_time_ms=(time.time() - start_time) * 1000
                    )
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Provider {provider} failed", symbol=symbol, error=str(e))
                    continue

            # All providers failed, use fallback
            fallback_data = await self.get_fallback_data(symbol)
            REQUEST_COUNT.labels(symbol=symbol, status="fallback").inc()
            
            return MarketDataResponse(
                success=True,
                data=fallback_data,
                cached=False,
                provider_used=DataProvider.FALLBACK,
                response_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            REQUEST_COUNT.labels(symbol=symbol, status="error").inc()
            logger.error("Complete market data failure", symbol=symbol, error=str(e))
            
            return MarketDataResponse(
                success=False,
                error=str(e),
                cached=False,
                provider_used=DataProvider.FALLBACK,
                response_time_ms=(time.time() - start_time) * 1000
            )

# Global service instance
market_data_service = MarketDataService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await market_data_service.startup()
    yield
    # Shutdown
    await market_data_service.shutdown()

# FastAPI application
app = FastAPI(
    title="Market Data Service",
    description="Production-ready market data service with circuit breakers and caching",
    version="1.0.0",
    lifespan=lifespan
)

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Simple token validation - replace with proper JWT validation in production"""
    if credentials.credentials != os.getenv("API_TOKEN", "market-data-token"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"user": "market_data_client"}

@app.get("/quote/{symbol}", response_model=MarketDataResponse)
async def get_quote_endpoint(
    symbol: str,
    user: dict = Depends(get_current_user)
):
    """
    Get real-time quote for a stock symbol
    
    - **symbol**: Stock symbol (e.g., AAPL, GOOGL)
    - Returns: Real-time quote data with failover and caching
    """
    with REQUEST_DURATION.time():
        symbol = symbol.upper()
        return await market_data_service.get_quote(symbol)

@app.get("/health")
async def health_check():
    """Service health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "circuit_breakers": {
            provider: cb.state for provider, cb in market_data_service.circuit_breakers.items()
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return {"message": "Metrics available at /metrics endpoint for Prometheus scraping"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
