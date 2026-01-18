"""
Cache Manager Service - Intelligent query result caching.
"""
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import asyncio

from app.core.config import settings


@dataclass
class CacheEntry:
    """A cached query result."""
    key: str
    query_hash: str
    result: Any
    created_at: float
    expires_at: float
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    tables_involved: List[str] = field(default_factory=list)


@dataclass
class CacheStats:
    """Statistics about cache performance."""
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    hit_rate: float
    eviction_count: int
    avg_entry_age_seconds: float


class CacheManager:
    """
    Intelligent cache manager for query results.
    
    Features:
    - LRU eviction policy
    - TTL-based expiration
    - Table-based invalidation
    - Size-based limits
    - Cache warming
    """
    
    def __init__(
        self,
        max_size_mb: int = 100,
        default_ttl_seconds: Optional[int] = None,
        enabled: Optional[bool] = None
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl_seconds or settings.CACHE_TTL_SECONDS
        self.enabled = enabled if enabled is not None else settings.ENABLE_QUERY_CACHE
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._table_keys: Dict[str, set] = {}  # table -> set of cache keys
        self._current_size = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def _compute_key(self, query: str, params: Optional[Dict] = None) -> str:
        """Compute cache key from query and parameters."""
        key_data = query
        if params:
            key_data += json.dumps(params, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        try:
            return len(json.dumps(data).encode())
        except (TypeError, ValueError):
            return 1000  # Default estimate
    
    async def get(
        self, 
        query: str, 
        params: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Any]]:
        """
        Get cached result for a query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Tuple of (cache_hit, result)
        """
        if not self.enabled:
            return False, None
        
        key = self._compute_key(query, params)
        
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return False, None
            
            entry = self._cache[key]
            
            # Check expiration
            if time.time() > entry.expires_at:
                await self._remove_entry(key)
                self._misses += 1
                return False, None
            
            # Update access info
            entry.hit_count += 1
            entry.last_accessed = time.time()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._hits += 1
            return True, entry.result
    
    async def set(
        self,
        query: str,
        result: Any,
        params: Optional[Dict] = None,
        ttl_seconds: Optional[int] = None,
        tables: Optional[List[str]] = None
    ) -> bool:
        """
        Cache a query result.
        
        Args:
            query: SQL query
            result: Query result to cache
            params: Query parameters
            ttl_seconds: Time to live in seconds
            tables: Tables involved in the query
            
        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False
        
        key = self._compute_key(query, params)
        ttl = ttl_seconds or self.default_ttl
        size = self._estimate_size(result)
        
        # Don't cache if result is too large (>10% of max cache size)
        if size > self.max_size_bytes * 0.1:
            return False
        
        async with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                await self._remove_entry(key)
            
            # Evict entries if needed
            while self._current_size + size > self.max_size_bytes and self._cache:
                await self._evict_lru()
            
            # Create new entry
            now = time.time()
            entry = CacheEntry(
                key=key,
                query_hash=hashlib.md5(query.encode()).hexdigest(),
                result=result,
                created_at=now,
                expires_at=now + ttl,
                size_bytes=size,
                tables_involved=tables or []
            )
            
            self._cache[key] = entry
            self._current_size += size
            
            # Track table associations
            for table in entry.tables_involved:
                if table not in self._table_keys:
                    self._table_keys[table] = set()
                self._table_keys[table].add(key)
        
        return True
    
    async def invalidate_query(self, query: str, params: Optional[Dict] = None) -> bool:
        """Invalidate a specific cached query."""
        key = self._compute_key(query, params)
        
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
        return False
    
    async def invalidate_table(self, table_name: str) -> int:
        """
        Invalidate all cached queries involving a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            if table_name not in self._table_keys:
                return 0
            
            keys_to_remove = list(self._table_keys[table_name])
            count = 0
            
            for key in keys_to_remove:
                if key in self._cache:
                    await self._remove_entry(key)
                    count += 1
            
            return count
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cached queries matching a pattern.
        
        Args:
            pattern: Pattern to match in query hash
            
        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if pattern in entry.query_hash
            ]
            
            for key in keys_to_remove:
                await self._remove_entry(key)
            
            return len(keys_to_remove)
    
    async def clear(self) -> int:
        """Clear all cached entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._table_keys.clear()
            self._current_size = 0
            return count
    
    async def _remove_entry(self, key: str):
        """Remove a cache entry."""
        if key not in self._cache:
            return
        
        entry = self._cache[key]
        self._current_size -= entry.size_bytes
        
        # Remove from table tracking
        for table in entry.tables_involved:
            if table in self._table_keys:
                self._table_keys[table].discard(key)
                if not self._table_keys[table]:
                    del self._table_keys[table]
        
        del self._cache[key]
    
    async def _evict_lru(self):
        """Evict the least recently used entry."""
        if not self._cache:
            return
        
        # Get first item (least recently used)
        key = next(iter(self._cache))
        await self._remove_entry(key)
        self._evictions += 1
    
    async def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry.expires_at
            ]
            
            for key in expired_keys:
                await self._remove_entry(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        avg_age = 0
        if self._cache:
            now = time.time()
            total_age = sum(now - e.created_at for e in self._cache.values())
            avg_age = total_age / len(self._cache)
        
        return CacheStats(
            total_entries=len(self._cache),
            total_size_bytes=self._current_size,
            hit_count=self._hits,
            miss_count=self._misses,
            hit_rate=hit_rate,
            eviction_count=self._evictions,
            avg_entry_age_seconds=avg_age
        )
    
    def get_hot_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed cached queries."""
        sorted_entries = sorted(
            self._cache.values(),
            key=lambda e: e.hit_count,
            reverse=True
        )
        
        return [
            {
                "query_hash": entry.query_hash,
                "hit_count": entry.hit_count,
                "size_bytes": entry.size_bytes,
                "tables": entry.tables_involved,
                "age_seconds": time.time() - entry.created_at,
                "ttl_remaining": max(0, entry.expires_at - time.time())
            }
            for entry in sorted_entries[:limit]
        ]
    
    async def warm_cache(
        self,
        queries: List[Tuple[str, Any, Dict]],
        ttl_seconds: Optional[int] = None
    ) -> int:
        """
        Pre-warm the cache with query results.
        
        Args:
            queries: List of (query, result, params) tuples
            ttl_seconds: TTL for warmed entries
            
        Returns:
            Number of entries added
        """
        count = 0
        for query, result, params in queries:
            if await self.set(query, result, params, ttl_seconds):
                count += 1
        return count
    
    def should_cache(
        self,
        query: str,
        execution_time_ms: float,
        result_size: int
    ) -> Tuple[bool, str]:
        """
        Determine if a query result should be cached.
        
        Args:
            query: The SQL query
            execution_time_ms: How long the query took
            result_size: Size of the result
            
        Returns:
            Tuple of (should_cache, reason)
        """
        if not self.enabled:
            return False, "Caching disabled"
        
        # Don't cache very fast queries (not worth the overhead)
        if execution_time_ms < 10:
            return False, "Query too fast to benefit from caching"
        
        # Don't cache very large results
        if result_size > self.max_size_bytes * 0.1:
            return False, "Result too large"
        
        # Don't cache if query contains certain keywords
        query_upper = query.upper()
        if any(kw in query_upper for kw in ['NOW()', 'CURRENT_', 'RANDOM()', 'RAND()']):
            return False, "Query contains non-deterministic function"
        
        # Don't cache writes
        if any(kw in query_upper for kw in ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE']):
            return False, "Write queries should not be cached"
        
        return True, "Query eligible for caching"
