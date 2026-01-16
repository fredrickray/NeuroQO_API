#!/usr/bin/env python3
"""
Standalone migration script to fix invalid optimization_rules_applied values.

Run with:
    python scripts/fix_optimization_types.py

No alembic dependency required.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.core.database import engine


# Mapping of invalid values to valid OptimizationType values
OPTIMIZATION_TYPE_FIXES = {
    'add_limit': 'limit_addition',
    'add_index': 'index_suggestion',
    'rewrite_query': 'query_rewrite',
    'optimize_select': 'select_optimization',
    'optimize_join': 'join_optimization',
    'eliminate_subquery': 'subquery_elimination',
    'pushdown_predicate': 'predicate_pushdown',
    'remove_distinct': 'distinct_removal',
    'convert_or_to_in': 'or_to_in',
    'convert_not_in': 'not_in_to_exists',
    'optimize_wildcard': 'wildcard_optimization',
}


async def fix_optimization_types():
    """Fix invalid optimization type values in the database."""
    
    print("=" * 60)
    print("Fix Invalid Optimization Types Migration")
    print("=" * 60)
    
    async with engine.begin() as conn:
        # Get all optimization results with non-null optimization_rules_applied
        result = await conn.execute(
            text("SELECT id, optimization_rules_applied FROM optimization_results WHERE optimization_rules_applied IS NOT NULL")
        )
        rows = result.fetchall()
        
        print(f"\nFound {len(rows)} optimization results to check...")
        
        fixed_count = 0
        
        for row in rows:
            opt_id = row[0]
            rules_applied = row[1]
            
            if rules_applied is None:
                continue
            
            # Parse JSON if it's a string
            if isinstance(rules_applied, str):
                try:
                    rules_list = json.loads(rules_applied)
                except json.JSONDecodeError:
                    print(f"  [SKIP] ID {opt_id}: Invalid JSON")
                    continue
            else:
                rules_list = rules_applied
            
            if not isinstance(rules_list, list):
                print(f"  [SKIP] ID {opt_id}: Not a list")
                continue
            
            # Fix invalid values
            updated = False
            fixed_rules = []
            changes = []
            
            for rule in rules_list:
                if rule in OPTIMIZATION_TYPE_FIXES:
                    new_rule = OPTIMIZATION_TYPE_FIXES[rule]
                    fixed_rules.append(new_rule)
                    changes.append(f"'{rule}' -> '{new_rule}'")
                    updated = True
                else:
                    fixed_rules.append(rule)
            
            # Update if any values were fixed
            if updated:
                await conn.execute(
                    text("UPDATE optimization_results SET optimization_rules_applied = :rules WHERE id = :id"),
                    {"rules": json.dumps(fixed_rules), "id": opt_id}
                )
                fixed_count += 1
                print(f"  [FIXED] ID {opt_id}: {', '.join(changes)}")
        
        print(f"\n{'=' * 60}")
        print(f"Migration complete!")
        print(f"  - Records checked: {len(rows)}")
        print(f"  - Records fixed: {fixed_count}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(fix_optimization_types())
