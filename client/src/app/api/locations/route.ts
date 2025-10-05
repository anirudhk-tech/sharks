import { NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';

interface Location {
  id: number;
  name: string;
  description: string;
  geom: {
    type: 'Point';
    coordinates: [number, number];
  };
}

interface GeoJSONFeature {
  type: 'Feature';
  properties: {
    id: number;
    name: string;
    description: string;
  };
  geometry: {
    type: 'Point';
    coordinates: [number, number];
  };
}

/**
 * GET /api/locations
 * Fetch all locations from Supabase PostGIS database
 * Returns GeoJSON FeatureCollection
 */
export async function GET() {
  try {
    // Query locations from PostGIS database
    const { data, error } = await supabase
      .from('locations')
      .select('id, name, description, geom')
      .order('created_at', { ascending: false });

    if (error) {
      throw new Error(`Database error: ${error.message}`);
    }

    // Convert PostGIS geometry to GeoJSON format
    const features: GeoJSONFeature[] = (data as Location[])?.map((location) => ({
      type: 'Feature',
      properties: {
        id: location.id,
        name: location.name,
        description: location.description,
      },
      geometry: location.geom,
    })) || [];

    return NextResponse.json({
      type: 'FeatureCollection',
      features,
    });
  } catch (error) {
    console.error('Error fetching locations:', error);
    return NextResponse.json(
      { 
        error: 'Failed to fetch locations',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
