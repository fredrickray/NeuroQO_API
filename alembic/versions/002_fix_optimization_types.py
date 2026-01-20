"""Fix invalid optimization_rules_applied values

Revision ID: 002_fix_optimization_types
Revises: 001_update_experiment_schema
Create Date: 2026-01-20

This migration fixes invalid optimization type values stored in the 
optimization_results table. It maps legacy/incorrect values to their
correct OptimizationType enum values.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session


# revision identifiers, used by Alembic.
revision = '002_fix_optimization_types'
down_revision = '001_update_experiment_schema'
branch_labels = None
depends_on = None


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


def upgrade():
    """
    Fix invalid optimization_rules_applied values in optimization_results table.
    The optimization_rules_applied column stores a JSON array of optimization types.
    """
    bind = op.get_bind()
    session = Session(bind=bind)
    
    try:
        # Get all optimization results with non-null optimization_rules_applied
        result = session.execute(
            sa.text("SELECT id, optimization_rules_applied FROM optimization_results WHERE optimization_rules_applied IS NOT NULL")
        )
        
        for row in result:
            opt_id = row[0]
            rules_applied = row[1]
            
            if rules_applied is None:
                continue
            
            # Parse JSON if it's a string
            import json
            if isinstance(rules_applied, str):
                try:
                    rules_list = json.loads(rules_applied)
                except json.JSONDecodeError:
                    continue
            else:
                rules_list = rules_applied
            
            if not isinstance(rules_list, list):
                continue
            
            # Fix invalid values
            updated = False
            fixed_rules = []
            for rule in rules_list:
                if rule in OPTIMIZATION_TYPE_FIXES:
                    fixed_rules.append(OPTIMIZATION_TYPE_FIXES[rule])
                    updated = True
                else:
                    fixed_rules.append(rule)
            
            # Update if any values were fixed
            if updated:
                session.execute(
                    sa.text("UPDATE optimization_results SET optimization_rules_applied = :rules WHERE id = :id"),
                    {"rules": json.dumps(fixed_rules), "id": opt_id}
                )
        
        session.commit()
        print(f"Migration complete: Fixed invalid optimization type values")
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def downgrade():
    """
    Reverse the optimization type fixes (convert back to old values).
    Note: This is a best-effort reversal.
    """
    bind = op.get_bind()
    session = Session(bind=bind)
    
    # Reverse mapping
    REVERSE_FIXES = {v: k for k, v in OPTIMIZATION_TYPE_FIXES.items()}
    
    try:
        result = session.execute(
            sa.text("SELECT id, optimization_rules_applied FROM optimization_results WHERE optimization_rules_applied IS NOT NULL")
        )
        
        for row in result:
            opt_id = row[0]
            rules_applied = row[1]
            
            if rules_applied is None:
                continue
            
            import json
            if isinstance(rules_applied, str):
                try:
                    rules_list = json.loads(rules_applied)
                except json.JSONDecodeError:
                    continue
            else:
                rules_list = rules_applied
            
            if not isinstance(rules_list, list):
                continue
            
            # Reverse fix
            updated = False
            reversed_rules = []
            for rule in rules_list:
                if rule in REVERSE_FIXES:
                    reversed_rules.append(REVERSE_FIXES[rule])
                    updated = True
                else:
                    reversed_rules.append(rule)
            
            if updated:
                session.execute(
                    sa.text("UPDATE optimization_results SET optimization_rules_applied = :rules WHERE id = :id"),
                    {"rules": json.dumps(reversed_rules), "id": opt_id}
                )
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
