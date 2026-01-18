"""Update experiment schema fields

Revision ID: 001_update_experiment_schema
Revises: 
Create Date: 2025-12-18 12:49:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_update_experiment_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Upgrade the database schema to match the new experiment model.
    
    Changes:
    1. Rename columns in experiments table
    2. Update experiment_metrics table
    3. Update enum values
    """
    
    # === EXPERIMENTS TABLE ===
    
    # Rename configuration columns
    op.alter_column('experiments', 'baseline_config', 
                    new_column_name='control_config',
                    existing_type=sa.JSON(),
                    existing_nullable=True)
    
    op.alter_column('experiments', 'variant_config', 
                    new_column_name='treatment_config',
                    existing_type=sa.JSON(),
                    existing_nullable=True)
    
    # Rename timestamp columns
    op.alter_column('experiments', 'start_time', 
                    new_column_name='started_at',
                    existing_type=sa.DateTime(),
                    existing_nullable=True)
    
    op.alter_column('experiments', 'end_time', 
                    new_column_name='ended_at',
                    existing_type=sa.DateTime(),
                    existing_nullable=True)
    
    # Rename traffic percentage column
    op.alter_column('experiments', 'baseline_percentage', 
                    new_column_name='traffic_percentage',
                    existing_type=sa.Integer(),
                    existing_nullable=True)
    
    # Drop duration_hours column (moving to sample-based completion)
    op.drop_column('experiments', 'duration_hours')
    
    # Add required_sample_size column
    op.add_column('experiments', 
                  sa.Column('required_sample_size', sa.Integer(), 
                           nullable=True, server_default='1000'))
    
    # Update ExperimentStatus enum to include 'aborted' instead of 'cancelled'
    # For PostgreSQL
    op.execute("ALTER TYPE experimentstatus RENAME VALUE 'cancelled' TO 'aborted'")
    
    # === EXPERIMENT_METRICS TABLE ===
    
    # Drop old columns
    op.drop_column('experiment_metrics', 'query_hash')
    op.drop_column('experiment_metrics', 'cpu_time_ms')
    op.drop_column('experiment_metrics', 'io_reads')
    
    # Add new columns
    op.add_column('experiment_metrics', 
                  sa.Column('query_log_id', sa.Integer(), nullable=True))
    
    op.add_column('experiment_metrics', 
                  sa.Column('cache_hit', sa.Boolean(), 
                           nullable=True, server_default='false'))
    
    op.add_column('experiment_metrics', 
                  sa.Column('error_occurred', sa.Boolean(), 
                           nullable=True, server_default='false'))
    
    op.add_column('experiment_metrics', 
                  sa.Column('additional_metrics', sa.JSON(), nullable=True))
    
    # Update variant values from 'baseline'/'variant' to 'control'/'treatment'
    op.execute("""
        UPDATE experiment_metrics 
        SET variant = 'control' 
        WHERE variant = 'baseline'
    """)
    
    op.execute("""
        UPDATE experiment_metrics 
        SET variant = 'treatment' 
        WHERE variant = 'variant'
    """)
    
    # Update winner values in experiments table
    op.execute("""
        UPDATE experiments 
        SET winner = 'control' 
        WHERE winner = 'baseline'
    """)
    
    op.execute("""
        UPDATE experiments 
        SET winner = 'treatment' 
        WHERE winner = 'variant'
    """)


def downgrade() -> None:
    """
    Downgrade the database schema to the previous version.
    """
    
    # === EXPERIMENT_METRICS TABLE ===
    
    # Revert variant values
    op.execute("""
        UPDATE experiment_metrics 
        SET variant = 'baseline' 
        WHERE variant = 'control'
    """)
    
    op.execute("""
        UPDATE experiment_metrics 
        SET variant = 'variant' 
        WHERE variant = 'treatment'
    """)
    
    # Revert winner values
    op.execute("""
        UPDATE experiments 
        SET winner = 'baseline' 
        WHERE winner = 'control'
    """)
    
    op.execute("""
        UPDATE experiments 
        SET winner = 'variant' 
        WHERE winner = 'treatment'
    """)
    
    # Drop new columns
    op.drop_column('experiment_metrics', 'additional_metrics')
    op.drop_column('experiment_metrics', 'error_occurred')
    op.drop_column('experiment_metrics', 'cache_hit')
    op.drop_column('experiment_metrics', 'query_log_id')
    
    # Add back old columns
    op.add_column('experiment_metrics', 
                  sa.Column('query_hash', sa.String(64), nullable=True))
    op.add_column('experiment_metrics', 
                  sa.Column('cpu_time_ms', sa.Float(), nullable=True))
    op.add_column('experiment_metrics', 
                  sa.Column('io_reads', sa.Integer(), nullable=True))
    
    # === EXPERIMENTS TABLE ===
    
    # Revert enum
    op.execute("ALTER TYPE experimentstatus RENAME VALUE 'aborted' TO 'cancelled'")
    
    # Drop required_sample_size
    op.drop_column('experiments', 'required_sample_size')
    
    # Add back duration_hours
    op.add_column('experiments', 
                  sa.Column('duration_hours', sa.Integer(), 
                           nullable=True, server_default='24'))
    
    # Rename columns back
    op.alter_column('experiments', 'traffic_percentage', 
                    new_column_name='baseline_percentage',
                    existing_type=sa.Integer(),
                    existing_nullable=True)
    
    op.alter_column('experiments', 'ended_at', 
                    new_column_name='end_time',
                    existing_type=sa.DateTime(),
                    existing_nullable=True)
    
    op.alter_column('experiments', 'started_at', 
                    new_column_name='start_time',
                    existing_type=sa.DateTime(),
                    existing_nullable=True)
    
    op.alter_column('experiments', 'treatment_config', 
                    new_column_name='variant_config',
                    existing_type=sa.JSON(),
                    existing_nullable=True)
    
    op.alter_column('experiments', 'control_config', 
                    new_column_name='baseline_config',
                    existing_type=sa.JSON(),
                    existing_nullable=True)
