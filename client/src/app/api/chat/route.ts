import { streamText, UIMessage, convertToModelMessages, tool, stepCountIs } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { z } from 'zod';
import { supabase } from '@/lib/supabase';

export const maxDuration = 30;

// Initialize OpenAI provider
const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Use GPT-5 as the model
const model = openai('gpt-5');

interface ChatRequest {
  messages: UIMessage[];
}

// Geocoding tool to fetch coordinates for locations
const geocodingTool = tool({
  description: 'Fetch coordinates (latitude and longitude) for a given location or address',
  inputSchema: z.object({
    location: z.string().describe('The location, address, or landmark to get coordinates for (e.g., "White House", "Times Square", "1600 Pennsylvania Avenue")'),
  }),
  execute: async ({ location }) => {
    try {
      // Using a free geocoding service (Nominatim from OpenStreetMap)
      const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(location)}&limit=1&addressdetails=1`;
      
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'AI-Chatbot/1.0', // Required by Nominatim
        },
        signal: AbortSignal.timeout(10000), // 10 second timeout
      });

      if (!response.ok) {
        throw new Error(`Geocoding service error: ${response.status}`);
      }

      const data = await response.json();

      if (!data || data.length === 0) {
        return {
          success: false,
          error: `No coordinates found for location: ${location}`,
          location,
        };
      }

      const result = data[0];
      const coordinates = {
        success: true,
        location: result.display_name,
        latitude: parseFloat(result.lat),
        longitude: parseFloat(result.lon),
        address: {
          country: result.address?.country || 'Unknown',
          state: result.address?.state || 'Unknown',
          city: result.address?.city || result.address?.town || result.address?.village || 'Unknown',
          postcode: result.address?.postcode || 'Unknown',
        },
        boundingBox: result.boundingbox ? {
          north: parseFloat(result.boundingbox[1]),
          south: parseFloat(result.boundingbox[0]),
          east: parseFloat(result.boundingbox[3]),
          west: parseFloat(result.boundingbox[2]),
        } : undefined,
      };
      
      return coordinates;
    } catch (error) {
      return {
        success: false,
        error: `Failed to fetch coordinates: ${error instanceof Error ? error.message : 'Unknown error'}`,
        location,
      };
    }
  },
});

// Add pin to map tool - saves location to Supabase database
const addMapPinTool = tool({
  description: 'Add a pin/marker to the map by saving a location to the database. Use this when users want to save, bookmark, or add a location to the map.',
  inputSchema: z.object({
    name: z.string().describe('Name or title of the location (e.g., "My Favorite Restaurant", "Meeting Point")'),
    description: z.string().optional().describe('Optional description of the location'),
    latitude: z.number().describe('Latitude coordinate of the location'),
    longitude: z.number().describe('Longitude coordinate of the location'),
  }),
  execute: async ({ name, description, latitude, longitude }) => {
    try {
      // Validate coordinates
      if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
        return {
          success: false,
          error: 'Invalid coordinates: Latitude must be between -90 and 90, longitude between -180 and 180',
        };
      }

      // Try to insert using PostGIS SQL function first
      let { data, error } = await supabase
        .rpc('insert_location_with_geom', {
          location_name: name,
          longitude: longitude,
          latitude: latitude,
          location_description: description || null
        });

      // If RPC function doesn't exist, fall back to raw insert
      if (error && error.message.includes('function insert_location_with_geom')) {
        console.log('RPC function not found, trying direct insert...');
        
        const insertResult = await supabase
          .from('locations')
          .insert([
            {
              name,
              description: description || null,
              geom: `ST_SetSRID(ST_MakePoint(${longitude}, ${latitude}), 4326)`
            }
          ])
          .select('id, name, description, created_at')
          .single();

        data = insertResult.data;
        error = insertResult.error;
      }

      if (error) {
        return {
          success: false,
          error: `Failed to save location: ${error.message}`,
          details: error,
        };
      }

      // Handle RPC response (array) vs direct insert response (object)
      const locationData = Array.isArray(data) ? data[0] : data;

      const result = {
        success: true,
        message: `Successfully added "${name}" to the map!`,
        location: {
          id: locationData.id,
          name: locationData.name,
          description: locationData.description,
          latitude,
          longitude,
          createdAt: locationData.created_at,
        },
        // Signal to refresh the map
        refreshMap: true,
      };

      // Broadcast real-time event to all connected clients
      try {
        const baseUrl = process.env.NEXTAUTH_URL || process.env.VERCEL_URL || 'http://localhost:3000';
        await fetch(`${baseUrl}/api/map-events`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: 'pin-added',
            data: result.location,
          }),
        });
      } catch (error) {
        // SSE broadcast failed, but pin was still added successfully
        console.warn('Failed to broadcast map event:', error);
      }

      return result;
    } catch (error) {
      return {
        success: false,
        error: `Failed to add pin to map: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  },
});

// Show location on map tool - centers/moves the map to a specific location
const showLocationOnMapTool = tool({
  description: 'Move and center the map to show a specific location. Use this when users ask to "show", "navigate to", or "go to" a location on the map.',
  inputSchema: z.object({
    name: z.string().describe('Name of the location being shown'),
    latitude: z.number().describe('Latitude coordinate of the location'),
    longitude: z.number().describe('Longitude coordinate of the location'),
    zoom: z.number().optional().describe('Optional zoom level (8-18, default: 14). Use higher values for closer zoom.'),
  }),
  execute: async ({ name, latitude, longitude, zoom = 14 }) => {
    try {
      // Validate coordinates
      if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
        return {
          success: false,
          error: 'Invalid coordinates: Latitude must be between -90 and 90, longitude between -180 and 180',
        };
      }

      // Validate zoom level
      const clampedZoom = Math.max(1, Math.min(20, zoom));

      console.log('Showing location on map:', { name, latitude, longitude, zoom: clampedZoom });

      const showResult = await supabase
      .from('zoom')
      .insert([
        {
          name,
          lat: latitude,
          long: longitude,
          zoom: clampedZoom,
        }
      ])
      .select('id, name, lat, long, zoom, created_at')
      .single();

        // Broadcast real-time event to all connected clients
        try {
          const baseUrl = process.env.NEXTAUTH_URL || process.env.VERCEL_URL || 'http://localhost:3000';
          await fetch(`${baseUrl}/api/map-events`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              type: 'show-location-on-map',
              data: showResult.data,
            }),
          });
        } catch (error) {
          // SSE broadcast failed, but pin was still added successfully
          console.warn('Failed to broadcast map event:', error);
        }

      return {
        success: true,
        message: `Showing ${name} on the map`,
        action: 'move-map',
        location: {
          name,
          latitude,
          longitude,
          zoom: clampedZoom
        },
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to show location on map: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  },
});

/**
 * POST /api/chat
 * Stream AI chat responses using GPT-5
 */
export async function POST(req: Request) {
  try {
    const { messages }: ChatRequest = await req.json();

    const result = streamText({
      model,
      messages: convertToModelMessages(messages),
      system: `You are a helpful assistant that can answer questions and help with tasks. 

AVAILABLE TOOLS:
- getCoordinates: Fetch latitude and longitude coordinates for any location, address, or landmark (e.g., "White House", "Eiffel Tower", "123 Main St, New York"). This tool provides detailed address information, coordinates, and bounding boxes.
- addMapPin: Add a pin/marker to the map by saving a location to the database. Use this when users want to save, bookmark, or add a location to the map. Requires a name, optional description, and coordinates (latitude/longitude).
- showLocationOnMap: Move and center the map to show a specific location. Use this when users ask to "show me", "navigate to", "go to", or "find" a location on the map. This will smoothly move the map to center on the requested location.

USAGE PATTERNS:
- For location queries: "Where is X?" → use getCoordinates
- For showing locations: "Show me the White House" → use getCoordinates + showLocationOnMap  
- For saving locations: "Add X to my map" → use getCoordinates + addMapPin
- For navigation: "Go to Times Square" → use getCoordinates + showLocationOnMap

You can combine tools as needed. Always get coordinates first with getCoordinates, then use the appropriate action tool.`,
      tools: {
        getCoordinates: geocodingTool,
        addMapPin: addMapPinTool,
        showLocationOnMap: showLocationOnMapTool,
      },
      providerOptions: {
        openai: {
          reasoningEffort: 'low',
          reasoningSummary: 'auto', // Enable reasoning summaries
        },
      },
      stopWhen: stepCountIs(5), // stop after a maximum of 5 steps if tools were called
    });

    // Stream response with sources and reasoning
    return result.toUIMessageStreamResponse({
      sendSources: true,
      sendReasoning: true,
    });
  } catch (error) {
    console.error('Error in chat API:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
}
