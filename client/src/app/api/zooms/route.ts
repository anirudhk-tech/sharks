import { NextRequest } from 'next/server';

// Simple in-memory store for SSE connections
const connections = new Set<ReadableStreamDefaultController>();

/**
 * GET /api/map-events
 * Server-Sent Events endpoint for real-time map updates
 */
export async function GET(request: NextRequest) {
  // Create a readable stream for SSE
  const stream = new ReadableStream({
    start(controller) {
      // Add this connection to our set
      connections.add(controller);
      
      // Send initial connection message
      controller.enqueue(`data: ${JSON.stringify({ type: 'connected', timestamp: Date.now() })}\n\n`);
      
      // Clean up when connection closes
      request.signal.addEventListener('abort', () => {
        connections.delete(controller);
        try {
          controller.close();
        } catch (e) {
          // Connection already closed
        }
      });
    },
  });

  // Return SSE response
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
    },
  });
}

/**
 * POST /api/map-events
 * Broadcast map update events to all connected clients
 */
export async function POST(request: NextRequest) {
  try {
    const { type, data } = await request.json();
    
    // Broadcast to all connected clients
    const message = `data: ${JSON.stringify({ type, data, timestamp: Date.now() })}\n\n`;
    
    connections.forEach((controller) => {
      try {
        controller.enqueue(message);
      } catch (e) {
        // Remove dead connections
        connections.delete(controller);
      }
    });
    
    return new Response(JSON.stringify({ success: true, sent: connections.size }), {
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({ error: 'Failed to broadcast event' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
