import { NextRequest, NextResponse } from 'next/server';
import sharkData from '@/data/sharkData.json';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const year = searchParams.get('year');
    const month = searchParams.get('month');
    const sharkId = searchParams.get('sharkId');

    // If specific time parameters are provided, filter the data
    if (year && month && sharkId) {
      const shark = sharkData.sharks.find(s => s.id === sharkId);
      if (!shark) {
        return NextResponse.json({ error: 'Shark not found' }, { status: 404 });
      }

      const yearData = shark.zones[year];
      if (!yearData) {
        return NextResponse.json({ error: 'Year data not found' }, { status: 404 });
      }

      const monthZones = yearData[month];
      if (!monthZones) {
        return NextResponse.json({ error: 'Month data not found' }, { status: 404 });
      }

      return NextResponse.json({
        shark: {
          id: shark.id,
          name: shark.name,
          scientificName: shark.scientificName,
          optimalSST: shark.optimalSST
        },
        zones: monthZones,
        year: parseInt(year),
        month: parseInt(month)
      });
    }

    // Return all shark data if no specific parameters
    return NextResponse.json(sharkData);
  } catch (error) {
    console.error('Error fetching shark data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch shark data' },
      { status: 500 }
    );
  }
}
