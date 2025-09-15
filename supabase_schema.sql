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
  inserted_at timestamptz default now()
);
