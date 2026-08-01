CREATE TABLE IF NOT EXISTS active_nodes (
    node_id TEXT PRIMARY KEY,
    session_token TEXT,
    jwt_token TEXT,
    hardware_json TEXT,
    benchmark_json TEXT,
    status TEXT DEFAULT 'idle',
    contributing INTEGER DEFAULT 1,
    current_task_id TEXT,
    last_heartbeat_ts INTEGER,
    jobs_completed INTEGER DEFAULT 0,
    jobs_failed INTEGER DEFAULT 0,
    reliability_score REAL DEFAULT 1.0,
    created_ts INTEGER DEFAULT (strftime('%s', 'now')),
    updated_ts INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    job_type TEXT DEFAULT 'fine_tune',
    base_model TEXT,
    dataset_ref TEXT,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER DEFAULT 0,
    priority_tier TEXT DEFAULT 'P1',
    total_votes INTEGER DEFAULT 0,
    vote_weight REAL DEFAULT 1.0,
    submitter_id TEXT DEFAULT 'distribai',
    org TEXT DEFAULT 'DistribAI',
    created_ts INTEGER NOT NULL,
    updated_ts INTEGER NOT NULL,
    started_ts INTEGER,
    completed_ts INTEGER,
    steps INTEGER DEFAULT 100,
    batch_size INTEGER DEFAULT 32,
    queue_position INTEGER,
    estimated_start_hours REAL,
    active_nodes INTEGER DEFAULT 0,
    progress_pct REAL DEFAULT 0,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 100,
    attempts INTEGER DEFAULT 0,
    latest_task_id TEXT,
    latest_reason TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    assignee_node_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    weight_blob_url TEXT,
    batch_blob_url TEXT,
    hparams_json TEXT,
    deadline_ts INTEGER,
    weight_version TEXT,
    steps INTEGER DEFAULT 1,
    step_offset INTEGER DEFAULT 0,
    attempt_count INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    benchmark_score REAL,
    reliability_score REAL,
    created_ts INTEGER DEFAULT (strftime('%s', 'now')),
    updated_ts INTEGER DEFAULT (strftime('%s', 'now')),
    started_ts INTEGER,
    completed_ts INTEGER,
    gradient_blob_url TEXT,
    output_json TEXT,
    last_error TEXT,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS credit_ledger (
    tx_id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT,
    tx_type TEXT,
    amount REAL,
    balance_after REAL,
    tx_hash TEXT,
    prev_hash TEXT,
    ts INTEGER
);

CREATE TABLE IF NOT EXISTS vote_transactions (
    vote_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    voter_id TEXT NOT NULL,
    credits INTEGER NOT NULL,
    created_ts INTEGER NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);
