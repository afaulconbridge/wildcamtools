# Persistence Module

Robust, normalized persistence layer for tracking video processing pipelines, frame analysis, and multi-level labeling.

## Overview

This module provides a database abstraction layer using SQLModel (SQLAlchemy + Pydantic). It supports:
- **SQLite** for local development
- **PostgreSQL** for production deployments
- **Database-agnostic** design via connection string configuration

## Database Management

### Engine Creation

```python
from wildcamtools.lib.persistence import create_engine_and_tables, PersistenceManager

# SQLite (local development)
engine = create_engine_and_tables("sqlite:///wildcam.db")

# PostgreSQL (production)
engine = create_engine_and_tables(
    "postgresql+psycopg://user:password@localhost:5432/wildcam"
)
```

### Using PersistenceManager

```python
from wildcamtools.lib.persistence import PersistenceManager

pm = PersistenceManager(engine)

# Create or lookup resolution
resolution = pm.get_or_create_resolution(1920, 1080)

# Create video record
video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)

# Create processing pipeline
step = pm.create_processing_step(resolution.id, "motion_detect", {"threshold": 0.5})
node = pm.create_processing_node(step.id, parent_node_id=None, run_id=1)

# Record frame versions
frame = pm.create_frame_version(
    node_id=node.id,
    video_id=video.id,
    frame_no=42,
    is_kept=True,
    motion_proportion=0.75,
    crop_coords=(100, 100, 800, 600),
)

# Create labels
frame_set = pm.create_frame_set("event_001")
ai_label = pm.create_ai_label(frame_set.id, "deer, night")
human_label = pm.create_human_label(video.id, "winter footage")
```

### Session Management

The `PersistenceManager` handles session lifecycle automatically. All operations:
- Create and close their own database sessions
- Rollback on exceptions
- Validate inputs before database operations

## Configuration

### Connection Strings

| Database | Connection String Format | Example |
|----------|-------------------------|---------|
| SQLite | `sqlite:///path/to/db.db` | `sqlite:///wildcam.db` |
| PostgreSQL | `postgresql+psycopg://user:pass@host:port/dbname` | `postgresql+psycopg://user:pass@localhost:5432/wildcam` |

### Environment Variables

Recommended environment variable configuration:

```bash
WILDCAM_DATABASE_URL=sqlite:///wildcam.db
```

### Best Practices

1. **Development**: Use SQLite with relative paths
2. **Production**: Use PostgreSQL with connection pooling
3. **Secrets**: Store database credentials in environment variables, not code

## Migration

Schema migrations are handled by `create_engine_and_tables()` which auto-creates all tables on connection.

For production deployments with controlled schema changes, configure Alembic migrations as needed.

## Backup

### SQLite Backup

```bash
# Simple file copy (while database is not in use)
cp wildcam.db wildcam.backup.db

# Online backup using sqlite3
sqlite3 wildcam.db ".backup 'wildcam.backup.db'"
```

### PostgreSQL Backup

```bash
# Full database backup
pg_dump -U user -h localhost wildcam > backup.sql

# Restore from backup
psql -U user -h localhost wildcam < backup.sql

# Continuous archiving (WAL-based)
# Configure postgresql.conf:
#   wal_level = replica
#   archive_mode = on
#   archive_command = 'cp %p /backup/%f'
```

### Backup Best Practices

1. **Frequency**: Daily backups for production, weekly for development
2. **Retention**: Keep 7 daily backups, 4 weekly backups
3. **Verification**: Regularly test restore procedures
4. **Offsite**: Store backups in separate location from primary database

## Data Model

### Entity Relationships

```
Resolution (1) ── (N) Video
     │                │
     │                ├─ (N) FrameVersion
     │                │
     └─ (N) ProcessingStep ── (N) ProcessingNode ── (N) FrameVersion
                                    │
                                    └─ (self-referential parent/children)

FrameSet (1) ── (N) FrameVersion
    │
    └─ (N) AILabel

Video (1) ── (N) HumanLabel
```

### Key Tables

| Table | Purpose |
|-------|---------|
| `resolution` | Normalized video resolutions (width x height) |
| `video` | Video file metadata with resolution reference |
| `processing_step` | Pipeline step definitions with config |
| `processing_node` | Instantiated steps with run tracking |
| `frame_version` | Individual frames with crop/motion data |
| `frame_set` | Grouped frames for event labeling |
| `ai_label` | AI-generated labels for frame sets |
| `human_label` | Human labels for videos |

## Error Handling

All `PersistenceManager` methods:
- Validate inputs (positive IDs, non-empty strings, valid ranges)
- Rollback transactions on failure
- Raise `ValueError` for invalid inputs
- Re-raise database exceptions with context

```python
try:
    pm.create_video("", 1)  # Will raise ValueError
except ValueError as e:
    logger.error("Invalid input: %s", e)
```

## Type Safety

- Python 3.13+ type hints throughout
- SQLModel combines Pydantic validation with SQLAlchemy ORM
- All public APIs have complete type annotations
