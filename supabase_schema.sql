-- Run this in Supabase SQL Editor once.
create table if not exists raw_gorgias (
  id bigint primary key,
  status text,
  channel text,
  via text,
  subject text,
  customer_email text,
  customer_name text,
  created_datetime timestamptz,
  updated_datetime timestamptz,
  last_message_datetime timestamptz,
  tags jsonb,
  raw jsonb,
  inserted_at timestamptz default now(),

  -- AI enrichment fields
  message_for_ai text,
  ai_labels jsonb,
  ai_sentiment text,
  ai_priority integer,
  ai_summary text,
  embedding jsonb,
  ai_updated_at timestamptz
);

-- Clustering analysis tables
create table if not exists ticket_clusters (
  id serial primary key,
  cluster_id integer not null,
  theme_name text not null,
  summary text,
  severity integer check (severity between 1 and 5),
  action_items text,
  ticket_count integer not null,
  avg_sentiment real,
  sentiment_trend text check (sentiment_trend in ('positive', 'neutral', 'negative')),
  centroid_embedding jsonb,
  created_at timestamptz default now()
);

create table if not exists ticket_cluster_assignments (
  id serial primary key,
  ticket_id bigint not null references raw_gorgias(id) on delete cascade,
  cluster_id integer not null,
  assigned_at timestamptz default now(),
  unique(ticket_id, cluster_id)
);

create table if not exists cluster_trends (
  id serial primary key,
  analysis_date timestamptz not null,
  total_tickets integer not null,
  total_clusters integer not null,
  analysis_period_days integer not null,
  cluster_distribution jsonb,
  created_at timestamptz default now()
);

-- Indexes for performance
create index if not exists idx_raw_gorgias_created_datetime on raw_gorgias(created_datetime);
create index if not exists idx_raw_gorgias_embedding on raw_gorgias using gin(embedding);
create index if not exists idx_ticket_cluster_assignments_ticket_id on ticket_cluster_assignments(ticket_id);
create index if not exists idx_ticket_cluster_assignments_cluster_id on ticket_cluster_assignments(cluster_id);
create index if not exists idx_ticket_clusters_created_at on ticket_clusters(created_at);
create index if not exists idx_cluster_trends_analysis_date on cluster_trends(analysis_date);
