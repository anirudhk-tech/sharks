-- Supabase PostGIS Setup
-- Run this SQL in your Supabase SQL Editor (Dashboard > SQL Editor > New Query)

-- 1. Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- 2. Create a table for map locations
CREATE TABLE IF NOT EXISTS locations (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  geom GEOMETRY(Point, 4326) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Create a spatial index for better query performance
CREATE INDEX IF NOT EXISTS idx_locations_geom ON locations USING GIST (geom);

-- 4. Enable Row Level Security (recommended for production)
ALTER TABLE locations ENABLE ROW LEVEL SECURITY;

-- 5. Create policies to allow public read and write access
-- Note: Adjust this based on your security requirements
DROP POLICY IF EXISTS "Allow public read access" ON locations;
CREATE POLICY "Allow public read access" 
  ON locations FOR SELECT 
  TO public 
  USING (true);

DROP POLICY IF EXISTS "Allow public insert access" ON locations;
CREATE POLICY "Allow public insert access" 
  ON locations FOR INSERT 
  TO public 
  WITH CHECK (true);

-- 6. Insert sample data (San Francisco area)
INSERT INTO locations (name, description, geom) VALUES
  ('Golden Gate Bridge', 'Famous suspension bridge connecting San Francisco to Marin County', ST_SetSRID(ST_MakePoint(-122.4783, 37.8199), 4326)),
  ('Alcatraz Island', 'Historic island prison and popular tourist destination', ST_SetSRID(ST_MakePoint(-122.4230, 37.8267), 4326)),
  ('Pier 39', 'Popular waterfront marketplace and tourist attraction', ST_SetSRID(ST_MakePoint(-122.4102, 37.8087), 4326)),
  ('Coit Tower', 'Art Deco tower with 360-degree views of the city', ST_SetSRID(ST_MakePoint(-122.4058, 37.8024), 4326)),
  ('Ferry Building', 'Historic transportation hub and gourmet food marketplace', ST_SetSRID(ST_MakePoint(-122.3933, 37.7955), 4326))
ON CONFLICT DO NOTHING;

-- Verify the data was inserted correctly
SELECT 
  id, 
  name, 
  description, 
  ST_X(geom) as longitude, 
  ST_Y(geom) as latitude 
FROM locations;
