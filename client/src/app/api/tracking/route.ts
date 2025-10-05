import { NextRequest, NextResponse } from 'next/server';
import sharkData from '@/data/sharkData.json';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const sharkId = searchParams.get('sharkId');

    // Get real-time tracking data
    let trackingData = sharkData.realTimeTracking?.sharks || [];

    // Filter by shark species if specified
    if (sharkId) {
      trackingData = trackingData.filter(shark => shark.sharkId === sharkId);
    }

    // Simulate real-time updates by modifying the data slightly
    const updatedData = trackingData.map(shark => ({
      ...shark,
      currentPosition: {
        lat: shark.currentPosition.lat + (Math.random() - 0.5) * 0.01,
        lng: shark.currentPosition.lng + (Math.random() - 0.5) * 0.01
      },
      jawMotion: {
        current: Math.random(),
        previous: shark.jawMotion.current,
        trend: Math.random() > 0.5 ? 'increasing' : 'decreasing'
      },
      lastUpdate: new Date().toISOString()
    }));

    return NextResponse.json({
      sharks: updatedData,
      timestamp: new Date().toISOString(),
      count: updatedData.length
    });
  } catch (error) {
    console.error('Error fetching tracking data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch tracking data' },
      { status: 500 }
    );
  }
}
